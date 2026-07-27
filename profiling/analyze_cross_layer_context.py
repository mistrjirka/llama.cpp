#!/usr/bin/env python3
"""Evaluate whether layer L-2 adds signal when predicting layer L.

All metrics are prequential: predict first, then update counts from that token.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def load(path: Path) -> tuple[list[int], list[dict[int, tuple[int, ...]]]]:
    by_step: dict[int, dict[int, tuple[int, ...]]] = defaultdict(dict)
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
            by_step[step][layer] = tuple(dict.fromkeys(map(int, record["selected"])))
    steps = sorted(by_step)
    layers = sorted(set.intersection(*(set(by_step[step]) for step in steps)))
    return layers, [{layer: by_step[step][layer] for layer in layers} for step in steps]


def ranking(scores: list[float], marginal: Counter[int]) -> list[int]:
    return sorted(range(len(scores)), key=lambda expert: (-scores[expert], -marginal[expert], expert))


def add_conditional_scores(
    destination: list[float],
    selected: tuple[int, ...],
    counts: list[Counter[int]],
    source_totals: Counter[int],
    weight: float = 1.0,
) -> None:
    for source in selected:
        denominator = max(1, source_totals[source])
        for target, count in counts[source].items():
            destination[target] += weight * count / denominator


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--experts", type=int, default=256)
    args = parser.parse_args()

    layers, tokens = load(args.trace)
    n_expert = args.experts
    ks = (1, 2, 4, 8, 16, 32)
    methods = (
        "adjacent_Lminus1",
        "skip_Lminus2",
        "adjacent_plus_skip_equal",
        "adjacent_plus_skip_half",
        "adjacent_plus_interaction",
    )
    totals = {
        method: {k: Counter(events=0, routes=0, hits=0, complete=0, any=0) for k in ks}
        for method in methods
    }
    windows = {method: {k: [] for k in ks} for method in methods}
    window_totals = {
        method: {k: Counter(events=0, routes=0, hits=0, complete=0, any=0) for k in ks}
        for method in methods
    }

    marginal = {layer: Counter() for layer in layers}
    adjacent_counts = {
        (layers[index - 1], layers[index]): [Counter() for _ in range(n_expert)]
        for index in range(1, len(layers))
    }
    adjacent_totals = {pair: Counter() for pair in adjacent_counts}
    skip_counts = {
        (layers[index - 2], layers[index]): [Counter() for _ in range(n_expert)]
        for index in range(2, len(layers))
    }
    skip_totals = {pair: Counter() for pair in skip_counts}
    # Sparse interaction: (expert at L-2, expert at L-1) -> target counts.
    interaction_counts: dict[tuple[int, int, int], Counter[int]] = defaultdict(Counter)
    interaction_totals: Counter[tuple[int, int, int]] = Counter()

    def record(method: str, predicted: list[int], actual: tuple[int, ...]) -> None:
        actual_set = set(actual)
        for k in ks:
            hits = len(actual_set.intersection(predicted[:k]))
            for counter in (totals[method][k], window_totals[method][k]):
                counter["events"] += 1
                counter["routes"] += len(actual_set)
                counter["hits"] += hits
                counter["complete"] += hits == len(actual_set)
                counter["any"] += hits > 0

    window_size = 32
    for token_index, token in enumerate(tokens):
        for index in range(2, len(layers)):
            previous2 = layers[index - 2]
            previous1 = layers[index - 1]
            target_layer = layers[index]
            source2 = token[previous2]
            source1 = token[previous1]
            actual = token[target_layer]
            fallback = marginal[target_layer]

            adjacent = [0.0] * n_expert
            add_conditional_scores(
                adjacent,
                source1,
                adjacent_counts[(previous1, target_layer)],
                adjacent_totals[(previous1, target_layer)],
            )
            skip = [0.0] * n_expert
            add_conditional_scores(
                skip,
                source2,
                skip_counts[(previous2, target_layer)],
                skip_totals[(previous2, target_layer)],
            )
            interaction = [0.0] * n_expert
            for expert2 in source2:
                for expert1 in source1:
                    key = (target_layer, expert2, expert1)
                    denominator = max(1, interaction_totals[key])
                    for target, count in interaction_counts[key].items():
                        interaction[target] += count / denominator

            record("adjacent_Lminus1", ranking(adjacent, fallback), actual)
            record("skip_Lminus2", ranking(skip, fallback), actual)
            record(
                "adjacent_plus_skip_equal",
                ranking([a + b for a, b in zip(adjacent, skip)], fallback),
                actual,
            )
            record(
                "adjacent_plus_skip_half",
                ranking([a + 0.5 * b for a, b in zip(adjacent, skip)], fallback),
                actual,
            )
            record(
                "adjacent_plus_interaction",
                ranking([a + 0.25 * b for a, b in zip(adjacent, interaction)], fallback),
                actual,
            )

        # Update after all predictions for the token.
        for layer in layers:
            marginal[layer].update(token[layer])
        for index in range(1, len(layers)):
            source_layer = layers[index - 1]
            target_layer = layers[index]
            pair = (source_layer, target_layer)
            for source in token[source_layer]:
                adjacent_totals[pair][source] += 1
                adjacent_counts[pair][source].update(token[target_layer])
        for index in range(2, len(layers)):
            source_layer = layers[index - 2]
            previous_layer = layers[index - 1]
            target_layer = layers[index]
            pair = (source_layer, target_layer)
            for source in token[source_layer]:
                skip_totals[pair][source] += 1
                skip_counts[pair][source].update(token[target_layer])
            for expert2 in token[source_layer]:
                for expert1 in token[previous_layer]:
                    key = (target_layer, expert2, expert1)
                    interaction_totals[key] += 1
                    interaction_counts[key].update(token[target_layer])

        if (token_index + 1) % window_size == 0 or token_index + 1 == len(tokens):
            window_start = token_index + 2 - min(window_size, token_index + 1)
            window_end = token_index + 1
            for method in methods:
                for k in ks:
                    counter = window_totals[method][k]
                    windows[method][k].append(
                        {
                            "token_start": window_start,
                            "token_end": window_end,
                            "route_recall": counter["hits"] / max(1, counter["routes"]),
                            "complete_rate": counter["complete"] / max(1, counter["events"]),
                            "any_rate": counter["any"] / max(1, counter["events"]),
                        }
                    )
                    counter.clear()

    result: dict[str, object] = {
        "tokens": len(tokens),
        "layers": layers,
        "target_layers": len(layers) - 2,
        "events": len(tokens) * (len(layers) - 2),
        "methods": {},
        "windows": windows,
    }
    for method in methods:
        result["methods"][method] = {}
        for k in ks:
            counter = totals[method][k]
            result["methods"][method][str(k)] = {
                "route_recall": counter["hits"] / counter["routes"],
                "complete_rate": counter["complete"] / counter["events"],
                "any_rate": counter["any"] / counter["events"],
                "mean_hits_per_layer": counter["hits"] / counter["events"],
            }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    print(f"tokens={len(tokens)} target_layers={len(layers)-2} events={result['events']}")
    print("method                           k  route_recall complete  any      mean_hits")
    for method in methods:
        for k in (1, 2, 4, 8, 16):
            row = result["methods"][method][str(k)]
            print(
                f"{method:32s} {k:2d} {row['route_recall']*100:11.2f}% "
                f"{row['complete_rate']*100:8.2f}% {row['any_rate']*100:8.2f}% "
                f"{row['mean_hits_per_layer']:9.3f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
