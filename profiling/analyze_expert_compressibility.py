#!/usr/bin/env python3
"""Measure lossless compression of individual quantized MoE expert bundles.

Run with the repository-local analysis venv and PYTHONPATH=gguf-py.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import time

from gguf import GGUFReader
import lz4.frame
import zstandard


def load_static_ranking(path: Path) -> dict[int, list[int]]:
    ranking: dict[int, list[int]] = {}
    with path.open() as stream:
        for line in stream:
            values = list(map(int, line.split()))
            if values:
                ranking[values[0]] = values[1:]
    return ranking


def load_tensors(model_dir: Path) -> dict[tuple[int, str], object]:
    result: dict[tuple[int, str], object] = {}
    for shard in sorted(model_dir.glob("*.gguf")):
        reader = GGUFReader(str(shard), "r")
        # Keep the reader reachable through each tensor's memmap object.
        for tensor in reader.tensors:
            name = tensor.name
            if not name.startswith("blk.") or "_exps.weight" not in name:
                continue
            pieces = name.split(".")
            layer = int(pieces[1])
            component = None
            for candidate in ("down", "gate", "up"):
                if f"ffn_{candidate}_exps.weight" in name:
                    component = candidate
                    break
            if component is not None:
                result[(layer, component)] = tensor
    return result


def selected_experts(layer: int, ranking: list[int], resident_count: int, count_each: int) -> list[tuple[str, int]]:
    resident = ranking[:resident_count]
    nonresident = ranking[resident_count:]
    selected: list[tuple[str, int]] = []
    selected.extend(("resident", expert) for expert in resident[:count_each])
    # Spread the cold sample through the tail rather than taking only its front.
    if nonresident:
        indices = [round(index * (len(nonresident) - 1) / max(1, count_each - 1)) for index in range(count_each)]
        selected.extend(("nonresident", nonresident[index]) for index in indices)
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--static-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-class", type=int, default=2)
    args = parser.parse_args()

    rankings = load_static_ranking(args.static_map)
    tensors = load_tensors(args.model_dir)
    layers = sorted({layer for layer, _ in tensors})
    if not layers:
        raise RuntimeError("no expert tensors found")

    codecs = {
        "zstd1": (
            lambda payload: zstandard.ZstdCompressor(level=1).compress(payload),
            lambda payload: zstandard.ZstdDecompressor().decompress(payload),
        ),
        "zstd3": (
            lambda payload: zstandard.ZstdCompressor(level=3).compress(payload),
            lambda payload: zstandard.ZstdDecompressor().decompress(payload),
        ),
        "zstd6": (
            lambda payload: zstandard.ZstdCompressor(level=6).compress(payload),
            lambda payload: zstandard.ZstdDecompressor().decompress(payload),
        ),
        "lz4fast": (
            lambda payload: lz4.frame.compress(payload, compression_level=0, block_linked=True),
            lambda payload: lz4.frame.decompress(payload),
        ),
        "lz4hc": (
            lambda payload: lz4.frame.compress(payload, compression_level=9, block_linked=True),
            lambda payload: lz4.frame.decompress(payload),
        ),
    }

    aggregate = {
        codec: {
            "raw_bytes": 0,
            "compressed_bytes": 0,
            "compress_seconds": 0.0,
            "decompress_seconds": 0.0,
            "ratios": [],
        }
        for codec in codecs
    }
    by_class = defaultdict(lambda: defaultdict(lambda: {"raw": 0, "compressed": 0}))
    by_type = defaultdict(lambda: defaultdict(lambda: {"raw": 0, "compressed": 0}))
    samples: list[dict[str, object]] = []

    for layer in layers:
        ranking = rankings.get(layer, list(range(256)))
        resident_count = 72 if layer == 46 else 98
        for cache_class, expert in selected_experts(
            layer, ranking, resident_count, args.samples_per_class
        ):
            bundle_record: dict[str, object] = {
                "layer": layer,
                "expert": expert,
                "cache_class": cache_class,
                "components": {},
            }
            for component in ("down", "gate", "up"):
                tensor = tensors[(layer, component)]
                raw = tensor.data[expert].tobytes(order="C")
                digest = hashlib.blake2b(raw, digest_size=8).hexdigest()
                component_record = {
                    "type": tensor.tensor_type.name,
                    "raw_bytes": len(raw),
                    "digest": digest,
                    "codecs": {},
                }
                for codec_name, (compress, decompress) in codecs.items():
                    start = time.perf_counter()
                    compressed = compress(raw)
                    compress_seconds = time.perf_counter() - start
                    start = time.perf_counter()
                    restored = decompress(compressed)
                    decompress_seconds = time.perf_counter() - start
                    if restored != raw:
                        raise RuntimeError(
                            f"round-trip mismatch layer={layer} expert={expert} component={component} codec={codec_name}"
                        )
                    raw_bytes = len(raw)
                    compressed_bytes = len(compressed)
                    ratio = compressed_bytes / raw_bytes
                    component_record["codecs"][codec_name] = {
                        "compressed_bytes": compressed_bytes,
                        "ratio": ratio,
                        "compress_seconds": compress_seconds,
                        "decompress_seconds": decompress_seconds,
                    }
                    stats = aggregate[codec_name]
                    stats["raw_bytes"] += raw_bytes
                    stats["compressed_bytes"] += compressed_bytes
                    stats["compress_seconds"] += compress_seconds
                    stats["decompress_seconds"] += decompress_seconds
                    stats["ratios"].append(ratio)
                    by_class[cache_class][codec_name]["raw"] += raw_bytes
                    by_class[cache_class][codec_name]["compressed"] += compressed_bytes
                    by_type[tensor.tensor_type.name][codec_name]["raw"] += raw_bytes
                    by_type[tensor.tensor_type.name][codec_name]["compressed"] += compressed_bytes
                bundle_record["components"][component] = component_record
            samples.append(bundle_record)

    summary: dict[str, object] = {
        "model_dir": str(args.model_dir),
        "static_map": str(args.static_map),
        "layers": layers,
        "sample_bundles": len(samples),
        "sample_components": len(samples) * 3,
        "codecs": {},
        "by_cache_class": {},
        "by_quantization_type": {},
        "samples": samples,
    }
    for codec_name, stats in aggregate.items():
        raw_bytes = stats["raw_bytes"]
        compressed_bytes = stats["compressed_bytes"]
        summary["codecs"][codec_name] = {
            "raw_bytes": raw_bytes,
            "compressed_bytes": compressed_bytes,
            "compressed_ratio": compressed_bytes / raw_bytes,
            "space_saving": 1.0 - compressed_bytes / raw_bytes,
            "compress_gib_s": raw_bytes / stats["compress_seconds"] / 2**30,
            "decompress_gib_s": raw_bytes / stats["decompress_seconds"] / 2**30,
            "median_component_ratio": statistics.median(stats["ratios"]),
            "min_component_ratio": min(stats["ratios"]),
            "max_component_ratio": max(stats["ratios"]),
        }
    for cache_class, values in by_class.items():
        summary["by_cache_class"][cache_class] = {
            codec: {
                "compressed_ratio": item["compressed"] / item["raw"],
                "space_saving": 1.0 - item["compressed"] / item["raw"],
            }
            for codec, item in values.items()
        }
    for quantization, values in by_type.items():
        summary["by_quantization_type"][quantization] = {
            codec: {
                "compressed_ratio": item["compressed"] / item["raw"],
                "space_saving": 1.0 - item["compressed"] / item["raw"],
                "raw_bytes": item["raw"],
            }
            for codec, item in values.items()
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")

    print(
        f"sampled {summary['sample_bundles']} expert bundles / "
        f"{summary['sample_components']} components"
    )
    print("codec      ratio    saving   compress GiB/s  decompress GiB/s  component range")
    for codec_name, stats in summary["codecs"].items():
        print(
            f"{codec_name:9s} {stats['compressed_ratio']*100:7.2f}% "
            f"{stats['space_saving']*100:7.2f}% {stats['compress_gib_s']:14.3f} "
            f"{stats['decompress_gib_s']:16.3f} "
            f"{stats['min_component_ratio']*100:6.2f}-{stats['max_component_ratio']*100:6.2f}%"
        )
    print("\nby cache class (zstd1 / lz4fast):")
    for cache_class, values in summary["by_cache_class"].items():
        print(
            cache_class,
            f"zstd1={values['zstd1']['compressed_ratio']*100:.2f}%",
            f"lz4fast={values['lz4fast']['compressed_ratio']*100:.2f}%",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
