#!/usr/bin/env python3
"""Prequential evaluation of MoE expert predictors on a route trace.

The predictor is evaluated before observing each target layer route, then updated.
This avoids training on the event being predicted.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Iterable


Selected = tuple[int, ...]


def load_tokens(path: Path) -> tuple[list[int], list[dict[int, Selected]]]:
    by_step: dict[int, dict[int, Selected]] = defaultdict(dict)
    with path.open() as stream:
        for line in stream:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("event") != "route" or record.get("phase") != "decode":
                continue
            step = int(record["step"])
            layer = int(record["layer"])
            selected = tuple(dict.fromkeys(int(value) for value in record["selected"]))
            by_step[step][layer] = selected
    steps = sorted(by_step)
    if not steps:
        raise ValueError(f"no decode route events in {path}")
    layers = sorted(set.intersection(*(set(by_step[step]) for step in steps)))
    tokens = [{layer: by_step[step][layer] for layer in layers} for step in steps]
    return layers, tokens


def rank_counter(counter: Counter[int], universe: int) -> list[int]:
    return sorted(range(universe), key=lambda expert: (-counter[expert], expert))


def rank_scores(scores: Iterable[float], fallback: Counter[int]) -> list[int]:
    score_list = list(scores)
    return sorted(
        range(len(score_list)),
        key=lambda expert: (-score_list[expert], -fallback[expert], expert),
    )


def evaluate(layers: list[int], tokens: list[dict[int, Selected]], universe: int) -> dict[str, object]:
    ks = (1, 2, 4, 8, 16, 32)
    methods = (
        "marginal",
        "same_layer_temporal_raw",
        "cross_layer_raw",
        "cross_layer_conditional",
        "cross_layer_blend",
    )
    totals = {
        method: {
            k: {"hits": 0, "routes": 0, "complete": 0, "events": 0, "any": 0}
            for k in ks
        }
        for method in methods
    }
    per_window: dict[str, dict[int, list[dict[str, float]]]] = {
        method: {k: [] for k in ks} for method in methods
    }

    marginal: dict[int, Counter[int]] = {layer: Counter() for layer in layers}
    same_transition: dict[int, list[Counter[int]]] = {
        layer: [Counter() for _ in range(universe)] for layer in layers
    }
    cross_transition: dict[tuple[int, int], list[Counter[int]]] = {
        (source, target): [Counter() for _ in range(universe)]
        for source, target in zip(layers, layers[1:])
    }
    cross_source_count: dict[tuple[int, int], Counter[int]] = {
        pair: Counter() for pair in cross_transition
    }
    previous: dict[int, Selected] = {}

    window_size = 32
    window_acc = {
        method: {
            k: {"hits": 0, "routes": 0, "complete": 0, "events": 0, "any": 0}
            for k in ks
        }
        for method in methods
    }

    def record(method: str, ranking: list[int], actual: Selected) -> None:
        actual_set = set(actual)
        for k in ks:
            predicted = set(ranking[:k])
            hits = len(actual_set & predicted)
            for destination in (totals[method][k], window_acc[method][k]):
                destination["hits"] += hits
                destination["routes"] += len(actual_set)
                destination["complete"] += hits == len(actual_set)
                destination["events"] += 1
                destination["any"] += hits > 0

    for token_index, token in enumerate(tokens):
        for source_layer, target_layer in zip(layers, layers[1:]):
            source = token[source_layer]
            actual = token[target_layer]
            fallback = marginal[target_layer]

            record("marginal", rank_counter(fallback, universe), actual)

            temporal_scores = [0.0] * universe
            prior_target = previous.get(target_layer, ())
            for source_expert in prior_target:
                for target_expert, count in same_transition[target_layer][source_expert].items():
                    temporal_scores[target_expert] += count
            record(
                "same_layer_temporal_raw",
                rank_scores(temporal_scores, fallback),
                actual,
            )

            pair = (source_layer, target_layer)
            raw_scores = [0.0] * universe
            conditional_scores = [0.0] * universe
            for source_expert in source:
                denominator = max(1, cross_source_count[pair][source_expert])
                for target_expert, count in cross_transition[pair][source_expert].items():
                    raw_scores[target_expert] += count
                    conditional_scores[target_expert] += count / denominator
            record("cross_layer_raw", rank_scores(raw_scores, fallback), actual)
            record(
                "cross_layer_conditional",
                rank_scores(conditional_scores, fallback),
                actual,
            )

            # Conditional evidence plus a weak marginal prior. The prior prevents
            # unstable early rankings while allowing strong cross-layer evidence
            # to dominate after a few observations.
            total_target_observations = max(1, sum(fallback.values()))
            blend_scores = [
                conditional_scores[expert] + 0.25 * fallback[expert] / total_target_observations
                for expert in range(universe)
            ]
            record("cross_layer_blend", rank_scores(blend_scores, fallback), actual)

        # Update all models only after predicting the whole token.
        for layer in layers:
            current = token[layer]
            prior = previous.get(layer)
            if prior:
                for source_expert in prior:
                    for target_expert in current:
                        same_transition[layer][source_expert][target_expert] += 1
            marginal[layer].update(current)
            previous[layer] = current

        for source_layer, target_layer in zip(layers, layers[1:]):
            pair = (source_layer, target_layer)
            source = token[source_layer]
            target = token[target_layer]
            for source_expert in source:
                cross_source_count[pair][source_expert] += 1
                for target_expert in target:
                    cross_transition[pair][source_expert][target_expert] += 1

        if (token_index + 1) % window_size == 0 or token_index + 1 == len(tokens):
            start = token_index + 2 - min(window_size, token_index + 1)
            end = token_index + 1
            for method in methods:
                for k in ks:
                    values = window_acc[method][k]
                    per_window[method][k].append(
                        {
                            "token_start": start,
                            "token_end": end,
                            "route_recall": values["hits"] / max(1, values["routes"]),
                            "complete_rate": values["complete"] / max(1, values["events"]),
                            "any_rate": values["any"] / max(1, values["events"]),
                        }
                    )
                    values.update(hits=0, routes=0, complete=0, events=0, any=0)

    summary: dict[str, object] = {
        "tokens": len(tokens),
        "layers": layers,
        "adjacent_layer_pairs": len(layers) - 1,
        "events": len(tokens) * (len(layers) - 1),
        "methods": {},
        "windows": per_window,
    }
    for method in methods:
        method_summary = {}
        for k in ks:
            values = totals[method][k]
            method_summary[str(k)] = {
                "route_recall": values["hits"] / values["routes"],
                "complete_rate": values["complete"] / values["events"],
                "any_rate": values["any"] / values["events"],
                "mean_hits_per_layer": values["hits"] / values["events"],
            }
        summary["methods"][method] = method_summary
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--experts", type=int, default=256)
    args = parser.parse_args()

    layers, tokens = load_tokens(args.trace)
    result = evaluate(layers, tokens, args.experts)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    print(f"tokens={result['tokens']} layers={layers[0]}..{layers[-1]} pairs={result['adjacent_layer_pairs']}")
    print("method                           k  route_recall complete  any      mean_hits")
    for method, values in result["methods"].items():
        for k in (1, 2, 4, 8, 16):
            row = values[str(k)]
            print(
                f"{method:32s} {k:2d} {row['route_recall']*100:11.2f}% "
                f"{row['complete_rate']*100:8.2f}% {row['any_rate']*100:8.2f}% "
                f"{row['mean_hits_per_layer']:9.3f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
