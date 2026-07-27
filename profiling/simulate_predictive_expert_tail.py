#!/usr/bin/env python3
"""Simulate a pinned hot core plus a predictor-controlled expert tail.

The simulation preserves a fixed total of 16 resident experts per layer.  A
small tail is filled by the online cross-layer Markov predictor, while the
remaining slots are pinned to training-time LFU experts.  Transfers are counted
when a predicted expert is inserted into a tail; no demand admission is used.
"""

from __future__ import annotations

import argparse
import collections
import copy
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_route_predictability import LayerPredictor, load_routes  # noqa: E402


def train(trace: Path, limit: int):
    steps, layers, routes = load_routes(trace)
    if limit:
        steps = steps[:limit]
    models = {layer: LayerPredictor(256, history=3) for layer in layers[3:]}
    counts = {layer: collections.Counter() for layer in layers[3:]}
    for step in steps:
        for index in range(3, len(layers)):
            layer = layers[index]
            model = models[layer]
            features = [routes[step][layers[index - lag]] for lag in range(1, 4)]
            actual = routes[step][layer]
            predicted = model.top_indices(model.perceptron_scores_lags(features, (0, 1, 2)), 8)
            model.update(features, actual, predicted)
            counts[layer].update(actual)
    return layers, models, counts


def simulate(
    trace: Path,
    layers: list[int],
    trained_models,
    counts,
    skip: int,
    limit: int,
    tail_slots: int,
    lags: tuple[int, ...],
    threshold: float,
):
    steps, test_layers, routes = load_routes(trace)
    if test_layers != layers:
        raise ValueError("layer mismatch")
    steps = steps[skip: skip + limit if limit else None]
    models = copy.deepcopy(trained_models)
    pinned_width = 16 - tail_slots
    pinned = {
        layer: set(expert for expert, _ in counts[layer].most_common(pinned_width))
        for layer in layers[3:]
    }
    tails: dict[int, collections.OrderedDict[int, int]] = {
        layer: collections.OrderedDict()
        for layer in layers[3:]
    }

    total_routes = 0
    pinned_hits = 0
    tail_hits = 0
    transfers = 0
    issued = 0
    immediate_useful = 0
    evicted_entries = 0
    evicted_entry_hits = 0
    next_generation = 1
    entry_hits: dict[tuple[int, int, int], int] = {}

    for step in steps:
        for index in range(3, len(layers)):
            layer = layers[index]
            model = models[layer]
            features = [routes[step][layers[index - lag]] for lag in range(1, 4)]
            actual_tuple = routes[step][layer]
            actual = set(actual_tuple)
            scores = model.markov_scores_lags(features, lags)
            positive_sum = float(np.maximum(scores, 0).sum())
            order = np.argsort(scores)[::-1]
            resident = pinned[layer] | set(tails[layer])
            candidate = next((int(value) for value in order if int(value) not in pinned[layer]), None)
            confidence = float(scores[candidate] / positive_sum) if candidate is not None and positive_sum > 0 else 0.0

            if candidate is not None and confidence >= threshold:
                issued += 1
                if candidate in tails[layer]:
                    tails[layer].move_to_end(candidate)
                else:
                    transfers += 1
                    if len(tails[layer]) >= tail_slots:
                        old_expert, old_generation = tails[layer].popitem(last=False)
                        evicted_entries += 1
                        evicted_entry_hits += entry_hits.get((layer, old_expert, old_generation), 0)
                    generation = next_generation
                    next_generation += 1
                    tails[layer][candidate] = generation
                    entry_hits[(layer, candidate, generation)] = 0
                if candidate in actual:
                    immediate_useful += 1

            total_routes += len(actual)
            pinned_hits += len(actual & pinned[layer])
            selected_tail = actual & set(tails[layer])
            tail_hits += len(selected_tail)
            for expert in selected_tail:
                generation = tails[layer][expert]
                entry_hits[(layer, expert, generation)] += 1
                tails[layer].move_to_end(expert)

            predicted = model.top_indices(model.perceptron_scores_lags(features, (0, 1, 2)), 8)
            model.update(features, actual_tuple, predicted)

    live_entries = 0
    live_entry_hits = 0
    for layer, tail in tails.items():
        for expert, generation in tail.items():
            live_entries += 1
            live_entry_hits += entry_hits.get((layer, expert, generation), 0)

    all_entry_hits = evicted_entry_hits + live_entry_hits
    return {
        "steps": len(steps),
        "target_layers": len(layers) - 3,
        "tail_slots": tail_slots,
        "pinned_slots": pinned_width,
        "lags": [lag + 1 for lag in lags],
        "threshold": threshold,
        "total_routes": total_routes,
        "pinned_hits": pinned_hits,
        "tail_hits": tail_hits,
        "route_coverage": (pinned_hits + tail_hits) / total_routes,
        "pinned_coverage": pinned_hits / total_routes,
        "tail_coverage_gain": tail_hits / total_routes,
        "issued_predictions": issued,
        "issued_per_token": issued / max(1, len(steps)),
        "transfers": transfers,
        "transfers_per_token": transfers / max(1, len(steps)),
        "immediate_prediction_precision": immediate_useful / max(1, issued),
        "tail_hits_per_transfer": all_entry_hits / max(1, transfers),
        "estimated_transfer_mib_per_token_at_11_5_mib": transfers * 11.5 / max(1, len(steps)),
        "evicted_entries": evicted_entries,
        "live_entries": live_entries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-trace", required=True, type=Path)
    parser.add_argument("--test-trace", required=True, type=Path)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--test-skip", type=int, default=0)
    parser.add_argument("--test-limit", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    layers, models, counts = train(args.train_trace, args.train_limit)
    scenarios = []
    for tail_slots in (1, 2, 4):
        for lags in ((2,), (1, 2), (0, 1, 2)):
            for threshold in (0.0, 0.04, 0.05, 0.06, 0.07):
                scenarios.append(simulate(
                    args.test_trace, layers, models, counts,
                    skip=args.test_skip,
                    limit=args.test_limit,
                    tail_slots=tail_slots,
                    lags=lags,
                    threshold=threshold,
                ))
    result = {
        "train_trace": str(args.train_trace),
        "test_trace": str(args.test_trace),
        "train_limit": args.train_limit,
        "test_skip": args.test_skip,
        "test_limit": args.test_limit,
        "scenarios": scenarios,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
