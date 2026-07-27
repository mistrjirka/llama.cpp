#!/usr/bin/env python3
"""Measure confidence selectivity of the online cross-layer Markov predictor."""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_route_predictability import LayerPredictor, load_routes  # noqa: E402


def train(trace: Path, expert_count: int, limit: int = 0):
    steps, layers, routes = load_routes(trace)
    if limit:
        steps = steps[:limit]
    models = {layer: LayerPredictor(expert_count, history=3) for layer in layers[3:]}
    static = {layer: collections.Counter() for layer in layers[3:]}
    for step in steps:
        for index in range(3, len(layers)):
            layer = layers[index]
            model = models[layer]
            features = [routes[step][layers[index - lag]] for lag in range(1, 4)]
            actual = routes[step][layer]
            pred = model.top_indices(model.perceptron_scores_lags(features, (0, 1, 2)), 8)
            model.update(features, actual, pred)
            static[layer].update(actual)
    return layers, models, static


def evaluate(trace: Path, layers, models, static, skip: int, limit: int, online: bool):
    steps, test_layers, routes = load_routes(trace)
    if test_layers != layers:
        raise ValueError("layer mismatch")
    steps = steps[skip: skip + limit if limit else None]
    records = {"lag3": [], "lags23": [], "lags123": []}
    lag_sets = {"lag3": (2,), "lags23": (1, 2), "lags123": (0, 1, 2)}
    total_routes = 0
    base_hits = 0

    for step in steps:
        for index in range(3, len(layers)):
            layer = layers[index]
            model = models[layer]
            features = [routes[step][layers[index - lag]] for lag in range(1, 4)]
            actual_tuple = routes[step][layer]
            actual = set(actual_tuple)
            static16 = set(expert for expert, _ in static[layer].most_common(16))
            total_routes += len(actual)
            base_hits += len(actual & static16)

            for name, lags in lag_sets.items():
                scores = model.markov_scores_lags(features, lags)
                order = np.argsort(scores)[::-1]
                candidates = [int(value) for value in order if int(value) not in static16]
                first = candidates[0]
                second = candidates[1]
                total_score = float(np.maximum(scores, 0).sum())
                probability = float(scores[first] / total_score) if total_score > 0 else 0.0
                margin = float((scores[first] - scores[second]) / total_score) if total_score > 0 else 0.0
                records[name].append({
                    "hit": first in actual,
                    "probability": probability,
                    "margin": margin,
                    "step": step,
                    "layer": layer,
                })

            if online:
                pred = model.top_indices(model.perceptron_scores_lags(features, (0, 1, 2)), 8)
                model.update(features, actual_tuple, pred)

    result = {
        "steps": len(steps),
        "target_layers": len(layers) - 3,
        "total_routes": total_routes,
        "static_top16_coverage": base_hits / total_routes,
        "online_updates": online,
        "stages": {},
    }
    for name, values in records.items():
        stage = {"opportunities": len(values), "issue_rates": {}}
        for rate in (0.01, 0.025, 0.05, 0.10, 0.20, 0.50, 1.0):
            count = max(1, round(len(values) * rate))
            chosen = sorted(values, key=lambda value: (value["probability"], value["margin"]), reverse=True)[:count]
            hits = sum(value["hit"] for value in chosen)
            prefetches_per_token = count / max(1, len(steps))
            stage["issue_rates"][str(rate)] = {
                "issued": count,
                "hits": hits,
                "precision": hits / count,
                "prefetches_per_token": prefetches_per_token,
                "route_coverage_gain": hits / total_routes,
                "resulting_route_coverage": (base_hits + hits) / total_routes,
            }
        result["stages"][name] = stage
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-trace", required=True, type=Path)
    parser.add_argument("--test-trace", required=True, type=Path)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--test-skip", type=int, default=0)
    parser.add_argument("--test-limit", type=int, default=0)
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    layers, models, static = train(args.train_trace, 256, args.train_limit)
    result = evaluate(
        args.test_trace, layers, models, static,
        skip=args.test_skip, limit=args.test_limit, online=args.online)
    result.update({
        "train_trace": str(args.train_trace),
        "test_trace": str(args.test_trace),
        "train_limit": args.train_limit,
        "test_skip": args.test_skip,
        "test_limit": args.test_limit,
    })
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
