#!/usr/bin/env python3
"""Measure collision cost of bounded hashed L-2/L-1 interaction tables."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def load(path: Path):
    by_step = defaultdict(dict)
    with path.open() as stream:
        for line in stream:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("event") == "route" and record.get("phase") == "decode":
                by_step[int(record["step"])][int(record["layer"])] = tuple(
                    dict.fromkeys(map(int, record["selected"]))
                )
    steps = sorted(by_step)
    layers = sorted(set.intersection(*(set(by_step[step]) for step in steps)))
    return layers, [{layer: by_step[step][layer] for layer in layers} for step in steps]


def bucket(expert2: int, expert1: int, buckets: int) -> int:
    value = (expert2 * 0x9E3779B1) & 0xFFFFFFFF
    value ^= ((expert1 * 0x85EBCA6B) + 0x27D4EB2D) & 0xFFFFFFFF
    value ^= value >> 16
    return value & (buckets - 1)


def evaluate(layers, tokens, buckets: int, weight: float):
    co = {target: [Counter() for _ in range(256)] for target in layers[2:]}
    source_n = {target: Counter() for target in layers[2:]}
    joint = {target: [Counter() for _ in range(buckets)] for target in layers[2:]}
    joint_n = {target: [0] * buckets for target in layers[2:]}
    marginal = {target: Counter() for target in layers[2:]}
    metrics = {k: Counter() for k in (1, 4, 8, 16)}
    occupied_pairs = {target: defaultdict(set) for target in layers[2:]}

    for token in tokens:
        for index in range(2, len(layers)):
            l2, l1, target = layers[index - 2], layers[index - 1], layers[index]
            source2, source1, actual = token[l2], token[l1], set(token[target])
            scores = [0.0] * 256
            for source in source1:
                n = source_n[target][source]
                if n:
                    for expert, count in co[target][source].items():
                        scores[expert] += count / n
            seen = set()
            for expert2 in source2:
                for expert1 in source1:
                    b = bucket(expert2, expert1, buckets)
                    if b in seen or not joint_n[target][b]:
                        continue
                    seen.add(b)
                    for expert, count in joint[target][b].items():
                        scores[expert] += weight * count / joint_n[target][b]
            ranked = sorted(range(256), key=lambda expert: (-scores[expert], -marginal[target][expert], expert))
            for k, metric in metrics.items():
                hits = len(actual.intersection(ranked[:k]))
                metric["hits"] += hits
                metric["routes"] += len(actual)
                metric["events"] += 1
                metric["any"] += hits > 0
                metric["complete"] += hits == len(actual)

        for index in range(2, len(layers)):
            l2, l1, target = layers[index - 2], layers[index - 1], layers[index]
            source2, source1, actual = token[l2], token[l1], token[target]
            marginal[target].update(actual)
            for source in source1:
                source_n[target][source] += 1
                co[target][source].update(actual)
            seen = set()
            for expert2 in source2:
                for expert1 in source1:
                    b = bucket(expert2, expert1, buckets)
                    occupied_pairs[target][b].add((expert2, expert1))
                    if b in seen:
                        continue
                    seen.add(b)
                    joint_n[target][b] += 1
                    joint[target][b].update(actual)

    result = {
        "buckets": buckets,
        "weight": weight,
        "metrics": {
            str(k): {
                "route_recall": m["hits"] / m["routes"],
                "any_rate": m["any"] / m["events"],
                "complete_rate": m["complete"] / m["events"],
            }
            for k, m in metrics.items()
        },
        "collision": {
            "mean_pairs_per_occupied_bucket": sum(
                len(pairs) for target in occupied_pairs.values() for pairs in target.values()
            ) / max(1, sum(len(target) for target in occupied_pairs.values())),
            "max_pairs_in_bucket": max(
                len(pairs) for target in occupied_pairs.values() for pairs in target.values()
            ),
        },
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    layers, tokens = load(args.trace)
    result = {"tokens": len(tokens), "configs": []}
    for buckets in (128, 256, 512, 1024, 2048):
        for weight in (0.25, 1.0):
            result["configs"].append(evaluate(layers, tokens, buckets, weight))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("buckets weight top1-hit k8-recall k8-complete pairs/bucket max-pairs")
    for item in result["configs"]:
        metrics = item["metrics"]
        collision = item["collision"]
        print(
            f"{item['buckets']:7d} {item['weight']:6.2f} "
            f"{metrics['1']['any_rate']*100:8.2f}% {metrics['8']['route_recall']*100:9.2f}% "
            f"{metrics['8']['complete_rate']*100:11.2f}% "
            f"{collision['mean_pairs_per_occupied_bucket']:12.2f} {collision['max_pairs_in_bucket']:9d}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
