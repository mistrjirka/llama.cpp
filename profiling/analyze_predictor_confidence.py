#!/usr/bin/env python3
"""Calibrate top-1 adjacent-layer predictions prequentially.

Reports precision/coverage and bytes per correct prediction for score and margin
thresholds. Uses only route IDs and the same normalized conditional score as the
runtime predictor.
"""

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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-observations", type=int, default=8)
    args = parser.parse_args()

    layers, tokens = load(args.trace)
    counts = {
        target: [Counter() for _ in range(256)]
        for target in layers[1:]
    }
    source_n = {target: Counter() for target in layers[1:]}
    records = []

    for token_index, token in enumerate(tokens, start=1):
        for source_layer, target_layer in zip(layers, layers[1:]):
            source = token[source_layer]
            actual = set(token[target_layer])
            scores = [0.0] * 256
            eligible = 0
            for source_expert in source:
                observations = source_n[target_layer][source_expert]
                if observations < args.min_observations:
                    continue
                eligible += 1
                for target_expert, count in counts[target_layer][source_expert].items():
                    scores[target_expert] += count / observations
            if eligible:
                ranked = sorted(range(256), key=lambda expert: (-scores[expert], expert))
                top = ranked[0]
                second = ranked[1]
                records.append(
                    {
                        "token": token_index,
                        "source_layer": source_layer,
                        "target_layer": target_layer,
                        "expert": top,
                        "score": scores[top],
                        "margin": scores[top] - scores[second],
                        "ratio": scores[top] / max(scores[second], 1e-12),
                        "correct": top in actual,
                        "eligible_sources": eligible,
                        "bytes": 5505024 if target_layer == 46 else 4079616,
                    }
                )

        for source_layer, target_layer in zip(layers, layers[1:]):
            source = token[source_layer]
            target = token[target_layer]
            for source_expert in source:
                source_n[target_layer][source_expert] += 1
                counts[target_layer][source_expert].update(target)

    thresholds = {}
    for field, values in {
        "score": [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0],
        "margin": [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0],
        "ratio": [1.0, 1.05, 1.1, 1.2, 1.5, 2.0],
    }.items():
        thresholds[field] = []
        for threshold in values:
            selected = [record for record in records if record[field] >= threshold]
            correct = sum(record["correct"] for record in selected)
            raw_bytes = sum(record["bytes"] for record in selected)
            thresholds[field].append(
                {
                    "threshold": threshold,
                    "predictions": len(selected),
                    "coverage": len(selected) / len(records),
                    "precision": correct / len(selected) if selected else None,
                    "correct": correct,
                    "bytes_gib": raw_bytes / 2**30,
                    "mib_per_correct": raw_bytes / 2**20 / correct if correct else None,
                }
            )

    # Equal-population score deciles are useful when raw score scale drifts.
    ordered = sorted(records, key=lambda record: record["score"], reverse=True)
    deciles = []
    for fraction in (0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0):
        selected = ordered[: max(1, round(len(ordered) * fraction))]
        correct = sum(record["correct"] for record in selected)
        deciles.append(
            {
                "top_fraction": fraction,
                "minimum_score": selected[-1]["score"],
                "predictions": len(selected),
                "precision": correct / len(selected),
                "correct": correct,
                "mib_per_correct": sum(record["bytes"] for record in selected) / 2**20 / correct,
            }
        )

    result = {
        "tokens": len(tokens),
        "layers": layers,
        "records": len(records),
        "overall_precision": sum(record["correct"] for record in records) / len(records),
        "thresholds": thresholds,
        "score_top_fractions": deciles,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"records={len(records)} overall precision={result['overall_precision']*100:.2f}%")
    print("top fraction  min score precision  MiB/correct")
    for row in deciles:
        print(
            f"{row['top_fraction']*100:10.0f}% {row['minimum_score']:10.4f} "
            f"{row['precision']*100:8.2f}% {row['mib_per_correct']:12.3f}"
        )
    print("score threshold precision coverage MiB/correct")
    for row in thresholds["score"]:
        print(
            f"{row['threshold']:14.2f} "
            f"{(row['precision'] or 0)*100:8.2f}% {row['coverage']*100:8.2f}% "
            f"{(row['mib_per_correct'] or 0):11.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
