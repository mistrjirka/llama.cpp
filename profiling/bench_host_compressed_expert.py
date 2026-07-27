#!/usr/bin/env python3
"""End-to-end host compressed expert -> pinned staging -> CUDA benchmark."""

from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path
import statistics
import time

from gguf import GGUFReader
import lz4.frame
import zstandard


CUDA_MEMCPY_HOST_TO_DEVICE = 1
CUDA_HOST_ALLOC_PORTABLE = 1


class Cuda:
    def __init__(self) -> None:
        self.lib = ctypes.CDLL("libcudart.so")
        self.lib.cudaSetDevice.argtypes = [ctypes.c_int]
        self.lib.cudaSetDevice.restype = ctypes.c_int
        self.lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.lib.cudaMalloc.restype = ctypes.c_int
        self.lib.cudaFree.argtypes = [ctypes.c_void_p]
        self.lib.cudaFree.restype = ctypes.c_int
        self.lib.cudaHostAlloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint]
        self.lib.cudaHostAlloc.restype = ctypes.c_int
        self.lib.cudaFreeHost.argtypes = [ctypes.c_void_p]
        self.lib.cudaFreeHost.restype = ctypes.c_int
        self.lib.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        self.lib.cudaMemcpy.restype = ctypes.c_int

    @staticmethod
    def check(code: int, operation: str) -> None:
        if code != 0:
            raise RuntimeError(f"{operation} failed with CUDA error {code}")

    def set_device(self, device: int) -> None:
        self.check(self.lib.cudaSetDevice(device), "cudaSetDevice")

    def malloc(self, size: int) -> ctypes.c_void_p:
        pointer = ctypes.c_void_p()
        self.check(self.lib.cudaMalloc(ctypes.byref(pointer), size), "cudaMalloc")
        return pointer

    def host_alloc(self, size: int) -> ctypes.c_void_p:
        pointer = ctypes.c_void_p()
        self.check(
            self.lib.cudaHostAlloc(ctypes.byref(pointer), size, CUDA_HOST_ALLOC_PORTABLE),
            "cudaHostAlloc",
        )
        return pointer

    def h2d(self, destination: ctypes.c_void_p, source: ctypes.c_void_p, size: int) -> None:
        self.check(
            self.lib.cudaMemcpy(destination, source, size, CUDA_MEMCPY_HOST_TO_DEVICE),
            "cudaMemcpy",
        )


def load_tensors(model_dir: Path):
    tensors = {}
    readers = []
    for shard in sorted(model_dir.glob("*.gguf")):
        reader = GGUFReader(str(shard), "r")
        readers.append(reader)
        for tensor in reader.tensors:
            tensors[tensor.name] = tensor
    return readers, tensors


def bundle(tensors, layer: int, expert: int) -> bytes:
    return b"".join(
        tensors[f"blk.{layer}.ffn_{component}_exps.weight"].data[expert].tobytes(order="C")
        for component in ("gate", "up", "down")
    )


def metrics(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "mean_ms": statistics.mean(ordered),
        "p50_ms": statistics.median(ordered),
        "p95_ms": ordered[int(0.95 * (len(ordered) - 1))],
        "p99_ms": ordered[int(0.99 * (len(ordered) - 1))],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def run_case(cuda: Cuda, raw: bytes, iterations: int) -> dict[str, object]:
    size = len(raw)
    device = cuda.malloc(size)
    pinned = cuda.host_alloc(size)
    pageable = ctypes.create_string_buffer(raw)
    ctypes.memmove(pinned, raw, size)

    zstd_compressor = zstandard.ZstdCompressor(level=1)
    zstd_decompressor = zstandard.ZstdDecompressor()
    zstd_payload = zstd_compressor.compress(raw)
    lz4_payload = lz4.frame.compress(raw, compression_level=0, block_linked=True)

    def direct_pageable() -> None:
        cuda.h2d(device, ctypes.cast(pageable, ctypes.c_void_p), size)

    def direct_pinned() -> None:
        cuda.h2d(device, pinned, size)

    def zstd_then_copy() -> None:
        restored = zstd_decompressor.decompress(zstd_payload, max_output_size=size)
        ctypes.memmove(pinned, restored, size)
        cuda.h2d(device, pinned, size)

    def lz4_then_copy() -> None:
        restored = lz4.frame.decompress(lz4_payload)
        ctypes.memmove(pinned, restored, size)
        cuda.h2d(device, pinned, size)

    operations = {
        "direct_pageable_h2d": direct_pageable,
        "direct_pinned_h2d": direct_pinned,
        "zstd1_decompress_pinned_h2d": zstd_then_copy,
        "lz4fast_decompress_pinned_h2d": lz4_then_copy,
    }
    result = {
        "raw_bytes": size,
        "zstd_bytes": len(zstd_payload),
        "lz4_bytes": len(lz4_payload),
        "zstd_ratio": len(zstd_payload) / size,
        "lz4_ratio": len(lz4_payload) / size,
        "operations": {},
    }
    for name, operation in operations.items():
        for _ in range(10):
            operation()
        samples = []
        for _ in range(iterations):
            start = time.perf_counter_ns()
            operation()
            samples.append((time.perf_counter_ns() - start) / 1e6)
        result["operations"][name] = metrics(samples)

    cuda.lib.cudaFreeHost(pinned)
    cuda.lib.cudaFree(device)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    readers, tensors = load_tensors(args.model_dir)
    del readers  # tensors retain their mmap references
    cuda = Cuda()
    cuda.set_device(0)
    cases = {
        "early_compressible_layer0_expert1": bundle(tensors, 0, 1),
        "middle_typical_layer30_expert0": bundle(tensors, 30, 0),
        "large_layer46_expert0": bundle(tensors, 46, 0),
    }
    result = {
        name: run_case(cuda, raw, args.iterations)
        for name, raw in cases.items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")

    for name, case in result.items():
        print(
            f"{name}: raw={case['raw_bytes']/2**20:.3f} MiB "
            f"zstd={case['zstd_ratio']*100:.2f}% lz4={case['lz4_ratio']*100:.2f}%"
        )
        for operation, values in case["operations"].items():
            print(
                f"  {operation:36s} p50={values['p50_ms']:.4f} ms "
                f"p95={values['p95_ms']:.4f} ms mean={values['mean_ms']:.4f} ms"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
