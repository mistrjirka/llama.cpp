#!/usr/bin/env python3
"""Train route predictors on one trace and evaluate frozen/online on another."""

from __future__ import annotations

import argparse
import collections
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_route_predictability import LayerPredictor, PredictionTotals, load_routes  # noqa: E402


def train_models(trace: Path, expert_count: int):
    steps, layers, routes = load_routes(trace)
    models = {layer: LayerPredictor(expert_count, history=3) for layer in layers[3:]}
    static_counts = {layer: collections.Counter() for layer in layers[3:]}
    for step in steps:
        for target_index in range(3, len(layers)):
            layer = layers[target_index]
            model = models[layer]
            features = [routes[step][layers[target_index - lag]] for lag in range(1, 4)]
            actual = routes[step][layer]
            predicted = model.top_indices(model.perceptron_scores_lags(features, (0, 1, 2)), 8)
            model.update(features, actual, predicted)
            static_counts[layer].update(actual)
    return steps, layers, models, static_counts


def evaluate(trace: Path, layers: list[int], models, static_counts, online: bool):
    steps, test_layers, routes = load_routes(trace)
    if test_layers != layers:
        raise ValueError(f"layer mismatch: train={layers} test={test_layers}")
    state = copy.deepcopy(models)
    totals = {
        "markov_lag3": PredictionTotals(),
        "markov_lags23": PredictionTotals(),
        "markov_lags123": PredictionTotals(),
        "perceptron_lags123": PredictionTotals(),
    }
    bins = []
    current_bin = {name: PredictionTotals() for name in totals}
    for step_index, step in enumerate(steps):
        if step_index and step_index % 16 == 0:
            bins.append({name: value.as_dict() for name, value in current_bin.items()})
            current_bin = {name: PredictionTotals() for name in totals}
        for target_index in range(3, len(layers)):
            layer = layers[target_index]
            model = state[layer]
            features = [routes[step][layers[target_index - lag]] for lag in range(1, 4)]
            actual_tuple = routes[step][layer]
            actual = set(actual_tuple)
            static16 = set(expert for expert, _ in static_counts[layer].most_common(16))
            copied = set(features[0])
            predictions = {
                "markov_lag3": model.top_indices(model.markov_scores_lags(features, (2,)), 16),
                "markov_lags23": model.top_indices(model.markov_scores_lags(features, (1, 2)), 16),
                "markov_lags123": model.top_indices(model.markov_scores_lags(features, (0, 1, 2)), 16),
                "perceptron_lags123": model.top_indices(model.perceptron_scores_lags(features, (0, 1, 2)), 16),
            }
            for name, predicted in predictions.items():
                totals[name].add(actual, predicted, copied, static16)
                current_bin[name].add(actual, predicted, copied, static16)
            if online:
                model.update(features, actual_tuple, predictions["perceptron_lags123"][:8])
    bins.append({name: value.as_dict() for name, value in current_bin.items()})
    return {
        "steps": len(steps),
        "online_updates": online,
        "predictors": {name: value.as_dict() for name, value in totals.items()},
        "sixteen_step_bins": bins,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-trace", required=True, type=Path)
    parser.add_argument("--test-trace", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expert-count", type=int, default=256)
    args = parser.parse_args()

    train_steps, layers, models, static_counts = train_models(args.train_trace, args.expert_count)
    result = {
        "train_trace": str(args.train_trace),
        "test_trace": str(args.test_trace),
        "train_steps": len(train_steps),
        "layers": len(layers),
        "frozen": evaluate(args.test_trace, layers, models, static_counts, online=False),
        "online": evaluate(args.test_trace, layers, models, static_counts, online=True),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
