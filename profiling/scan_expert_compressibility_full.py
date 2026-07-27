#!/usr/bin/env python3
"""Full lossless-compressibility scan of every MoE expert component."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

from gguf import GGUFReader
import lz4.frame
import zstandard


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    tensors = {}
    for shard in sorted(args.model_dir.glob("*.gguf")):
        reader = GGUFReader(str(shard), "r")
        for tensor in reader.tensors:
            if tensor.name.startswith("blk.") and "_exps.weight" in tensor.name:
                tensors[tensor.name] = tensor

    codecs = {
        "zstd1": lambda payload: zstandard.ZstdCompressor(level=1).compress(payload),
        "lz4fast": lambda payload: lz4.frame.compress(payload, compression_level=0, block_linked=True),
    }
    aggregate = {
        name: {"raw": 0, "compressed": 0, "seconds": 0.0, "components_smaller": 0}
        for name in codecs
    }
    per_layer = defaultdict(
        lambda: {
            name: {"raw": 0, "compressed": 0, "components_smaller": 0}
            for name in codecs
        }
    )
    per_type = defaultdict(
        lambda: {
            name: {"raw": 0, "compressed": 0, "components_smaller": 0}
            for name in codecs
        }
    )
    ratio_histogram = {
        name: {"lt_0.1": 0, "0.1_0.5": 0, "0.5_0.9": 0, "0.9_0.99": 0, "0.99_1.0": 0, "ge_1.0": 0}
        for name in codecs
    }

    components = 0
    bundles = 0
    for layer in range(48):
        layer_tensors = []
        for component in ("down", "gate", "up"):
            name = f"blk.{layer}.ffn_{component}_exps.weight"
            if name not in tensors:
                raise RuntimeError(f"missing {name}")
            layer_tensors.append(tensors[name])
        for expert in range(256):
            bundles += 1
            for tensor in layer_tensors:
                raw = tensor.data[expert].tobytes(order="C")
                components += 1
                for codec_name, compress in codecs.items():
                    start = time.perf_counter()
                    encoded = compress(raw)
                    elapsed = time.perf_counter() - start
                    raw_size = len(raw)
                    encoded_size = len(encoded)
                    ratio = encoded_size / raw_size
                    aggregate[codec_name]["raw"] += raw_size
                    aggregate[codec_name]["compressed"] += encoded_size
                    aggregate[codec_name]["seconds"] += elapsed
                    aggregate[codec_name]["components_smaller"] += encoded_size < raw_size
                    per_layer[layer][codec_name]["raw"] += raw_size
                    per_layer[layer][codec_name]["compressed"] += encoded_size
                    per_layer[layer][codec_name]["components_smaller"] += encoded_size < raw_size
                    per_type[tensor.tensor_type.name][codec_name]["raw"] += raw_size
                    per_type[tensor.tensor_type.name][codec_name]["compressed"] += encoded_size
                    per_type[tensor.tensor_type.name][codec_name]["components_smaller"] += encoded_size < raw_size
                    if ratio < 0.1:
                        bucket = "lt_0.1"
                    elif ratio < 0.5:
                        bucket = "0.1_0.5"
                    elif ratio < 0.9:
                        bucket = "0.5_0.9"
                    elif ratio < 0.99:
                        bucket = "0.9_0.99"
                    elif ratio < 1.0:
                        bucket = "0.99_1.0"
                    else:
                        bucket = "ge_1.0"
                    ratio_histogram[codec_name][bucket] += 1

    def finalize(values):
        raw = values["raw"]
        compressed = values["compressed"]
        result = {
            "raw_bytes": raw,
            "compressed_bytes": compressed,
            "compressed_ratio": compressed / raw,
            "space_saving": 1.0 - compressed / raw,
            "components_smaller": values["components_smaller"],
        }
        if "seconds" in values:
            result["compress_seconds"] = values["seconds"]
            result["compress_gib_s"] = raw / values["seconds"] / 2**30
        return result

    result = {
        "model_dir": str(args.model_dir),
        "bundles": bundles,
        "components": components,
        "aggregate": {name: finalize(values) for name, values in aggregate.items()},
        "per_layer": {
            str(layer): {name: finalize(values) for name, values in codecs_data.items()}
            for layer, codecs_data in per_layer.items()
        },
        "per_type": {
            quant: {name: finalize(values) for name, values in codecs_data.items()}
            for quant, codecs_data in per_type.items()
        },
        "ratio_histogram": ratio_histogram,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")

    print(f"bundles={bundles} components={components}")
    for name, values in result["aggregate"].items():
        print(
            f"{name}: ratio={values['compressed_ratio']*100:.2f}% "
            f"saving={values['space_saving']*100:.2f}% "
            f"speed={values['compress_gib_s']:.3f} GiB/s "
            f"smaller={values['components_smaller']}/{components}"
        )
    print("per-layer zstd1 ratio:")
    for layer in range(48):
        ratio = result["per_layer"][str(layer)]["zstd1"]["compressed_ratio"]
        print(f"{layer:2d} {ratio*100:7.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
