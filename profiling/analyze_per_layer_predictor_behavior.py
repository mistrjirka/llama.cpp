#!/usr/bin/env python3
"""Measure how route predictors behave independently at every MoE layer.

The tested neural predictor is intentionally *not* one global head.  Each target
layer has its own bounded perceptron weights because expert IDs and correlations
are layer-local.  This script quantifies that variation and records whether each
layer prefers a decayed conditional base, a standalone perceptron, or a hybrid.

Evaluation is prequential: predict, score, then update.  The authoritative model
router is never replaced; these scores are cache/prefetch hints only.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics
from typing import Sequence

import numpy as np


Route = tuple[int, ...]


def parse_csv_floats(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected comma-separated floats")
    return values


def load_trace(path: Path, max_steps: int | None = None):
    by_step: dict[int, dict[int, Route]] = defaultdict(dict)
    with path.open(errors="replace") as stream:
        for line in stream:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") != "route" or event.get("phase") != "decode":
                continue
            by_step[int(event["step"])][int(event["layer"])] = tuple(
                dict.fromkeys(int(value) for value in event["selected"])
            )
    if not by_step:
        raise ValueError(f"no decode routes found in {path}")

    max_layers = max(len(layer_map) for layer_map in by_step.values())
    layer_sets = Counter(
        frozenset(layer_map)
        for layer_map in by_step.values()
        if len(layer_map) == max_layers
    )
    expected = set(layer_sets.most_common(1)[0][0])
    layers = sorted(expected)
    steps = sorted(step for step, layer_map in by_step.items() if set(layer_map) == expected)
    if max_steps is not None:
        steps = steps[:max_steps]
    tokens = [{layer: by_step[step][layer] for layer in layers} for step in steps]
    return steps, layers, tokens


def stable_order(score: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    expert_ids = np.arange(score.size, dtype=np.int32)
    return np.lexsort((expert_ids, -fallback, -score))


def normalize(score: np.ndarray) -> np.ndarray:
    centered = score.astype(np.float32, copy=False) - float(np.mean(score))
    scale = float(np.std(centered))
    return centered if scale < 1e-6 else centered / scale


def method_for_alpha(alpha: float) -> str:
    return f"hybrid_a{alpha:g}".replace(".", "p")


def saturating_update(
    weights: np.ndarray,
    rows: Sequence[int],
    features: np.ndarray,
    delta: int,
    limit: int,
) -> None:
    if not rows:
        return
    row_index = np.asarray(rows, dtype=np.int64)
    block = weights[np.ix_(row_index, features)].astype(np.int32)
    block += delta
    np.clip(block, -limit, limit, out=block)
    weights[np.ix_(row_index, features)] = block.astype(np.int16)


def rankdata(values: Sequence[float]) -> np.ndarray:
    values_array = np.asarray(values, dtype=np.float64)
    order = np.argsort(values_array, kind="stable")
    ranks = np.empty(values_array.size, dtype=np.float64)
    position = 0
    while position < order.size:
        end = position + 1
        while end < order.size and values_array[order[end]] == values_array[order[position]]:
            end += 1
        average_rank = 0.5 * (position + end - 1)
        ranks[order[position:end]] = average_rank
        position = end
    return ranks


def spearman(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) < 2 or len(right) != len(left):
        return 0.0
    left_rank = rankdata(left)
    right_rank = rankdata(right)
    left_centered = left_rank - np.mean(left_rank)
    right_centered = right_rank - np.mean(right_rank)
    denominator = float(
        np.sqrt(np.sum(left_centered * left_centered) * np.sum(right_centered * right_centered))
    )
    if denominator == 0.0:
        return 0.0
    return float(np.sum(left_centered * right_centered) / denominator)


def entropy_bits(counts: np.ndarray) -> float:
    total = float(np.sum(counts))
    if total <= 0:
        return 0.0
    probabilities = counts[counts > 0] / total
    return float(-np.sum(probabilities * np.log2(probabilities)))


def summarize_values(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"min": 0.0, "median": 0.0, "mean": 0.0, "max": 0.0, "stdev": 0.0}
    return {
        "min": ordered[0],
        "median": statistics.median(ordered),
        "mean": statistics.mean(ordered),
        "max": ordered[-1],
        "stdev": statistics.pstdev(ordered),
    }


class PerLayerStudy:
    def __init__(
        self,
        *,
        layers: Sequence[int],
        experts: int,
        history: int,
        alphas: Sequence[float],
        decay_interval: int,
        decay_factor: float,
        weight_limit: int,
        bin_size: int,
    ) -> None:
        self.layers = list(layers)
        self.experts = experts
        self.history = history
        self.alphas = tuple(alphas)
        self.decay_interval = decay_interval
        self.decay_factor = decay_factor
        self.weight_limit = weight_limit
        self.bin_size = bin_size
        self.target_layers = self.layers[history:]
        self.target_count = len(self.target_layers)

        self.methods = (
            "decayed_conditional",
            "standalone_perceptron",
            *(method_for_alpha(alpha) for alpha in self.alphas),
        )
        self.method_index = {name: index for index, name in enumerate(self.methods)}
        self.budgets = (8, 16, 32)
        self.budget_index = {budget: index for index, budget in enumerate(self.budgets)}

        shape = (self.target_count, history, experts, experts)
        self.counts = np.zeros(shape, dtype=np.float32)
        self.source_counts = np.zeros((self.target_count, history, experts), dtype=np.float32)
        self.marginal = np.zeros((self.target_count, experts), dtype=np.float32)
        feature_count = 1 + history * experts
        self.perceptron = np.zeros(
            (self.target_count, experts, feature_count), dtype=np.int16
        )

        self.hits = np.zeros(
            (len(self.methods), self.target_count, len(self.budgets)), dtype=np.int64
        )
        self.routes = np.zeros(self.target_count, dtype=np.int64)
        self.actual_frequency = np.zeros((self.target_count, experts), dtype=np.int64)
        self.bin_hits = np.zeros_like(self.hits)
        self.bin_routes = np.zeros_like(self.routes)
        self.bins: list[dict[str, object]] = []
        self.update_pairs = np.zeros(self.target_count, dtype=np.int64)

    def features(self, token: dict[int, Route], target_index: int) -> tuple[list[Route], np.ndarray]:
        routed_index = target_index + self.history
        routes = [
            token[self.layers[routed_index - distance]]
            for distance in range(1, self.history + 1)
        ]
        feature_ids = [0]
        for lag, route in enumerate(routes):
            feature_ids.extend(1 + lag * self.experts + expert for expert in route)
        return routes, np.asarray(feature_ids, dtype=np.int64)

    def conditional_score(self, target: int, routes: Sequence[Route]) -> np.ndarray:
        prior = self.marginal[target]
        prior_probability = (prior + 0.25) / (
            float(np.sum(prior)) + 0.25 * self.experts
        )
        score = 0.35 * prior_probability
        lag_weights = (1.0, 0.72, 0.5184)
        for lag, source_route in enumerate(routes):
            source = np.asarray(source_route, dtype=np.int64)
            denominators = self.source_counts[target, lag, source]
            valid = denominators > 0
            if not np.any(valid):
                continue
            rows = self.counts[target, lag, source[valid]] / denominators[valid, None]
            score = score + lag_weights[lag] * np.mean(rows, axis=0)
        return score

    def record(
        self,
        method: str,
        target: int,
        actual: set[int],
        order: Sequence[int],
    ) -> None:
        method_index = self.method_index[method]
        for budget in self.budgets:
            budget_index = self.budget_index[budget]
            predicted = set(int(value) for value in order[:budget])
            hit_count = len(actual & predicted)
            self.hits[method_index, target, budget_index] += hit_count
            self.bin_hits[method_index, target, budget_index] += hit_count

    def update_perceptron(
        self,
        target: int,
        feature_ids: np.ndarray,
        actual: set[int],
        order: Sequence[int],
    ) -> None:
        predicted = [int(value) for value in order[:8]]
        missed = [expert for expert in actual if expert not in predicted]
        false = [expert for expert in predicted if expert not in actual]
        pair_count = min(len(missed), len(false))
        saturating_update(
            self.perceptron[target],
            missed[:pair_count],
            feature_ids,
            +1,
            self.weight_limit,
        )
        saturating_update(
            self.perceptron[target],
            false[:pair_count],
            feature_ids,
            -1,
            self.weight_limit,
        )
        self.update_pairs[target] += pair_count

    def update_conditional(
        self,
        target: int,
        routes: Sequence[Route],
        actual_tuple: Route,
    ) -> None:
        actual = np.asarray(actual_tuple, dtype=np.int64)
        self.marginal[target, actual] += 1.0
        self.actual_frequency[target, actual] += 1
        for lag, source_route in enumerate(routes):
            source = np.asarray(source_route, dtype=np.int64)
            self.counts[target, lag][np.ix_(source, actual)] += 1.0
            self.source_counts[target, lag, source] += 1.0

    def flush_bin(self, token_end: int) -> None:
        start = 1 if not self.bins else int(self.bins[-1]["token_end"]) + 1
        methods: dict[str, object] = {}
        total_routes = int(np.sum(self.bin_routes))
        for method, method_index in self.method_index.items():
            methods[method] = {
                str(budget): float(
                    np.sum(self.bin_hits[method_index, :, self.budget_index[budget]])
                    / max(1, total_routes)
                )
                for budget in self.budgets
            }
        self.bins.append(
            {
                "token_start": start,
                "token_end": token_end,
                "route_recall": methods,
            }
        )
        self.bin_hits.fill(0)
        self.bin_routes.fill(0)

    def run(self, tokens: Sequence[dict[int, Route]]) -> dict[str, object]:
        for token_number, token in enumerate(tokens, start=1):
            if (
                self.decay_interval > 0
                and token_number > 1
                and (token_number - 1) % self.decay_interval == 0
            ):
                self.counts *= self.decay_factor
                self.source_counts *= self.decay_factor
                self.marginal *= self.decay_factor

            pending = []
            for target, target_layer in enumerate(self.target_layers):
                route_features, feature_ids = self.features(token, target)
                actual_tuple = token[target_layer]
                actual = set(actual_tuple)
                fallback = self.marginal[target]
                conditional = self.conditional_score(target, route_features)
                perceptron_score = np.sum(
                    self.perceptron[target, :, feature_ids], axis=0
                ).astype(np.float32)
                conditional_norm = normalize(conditional)
                perceptron_norm = normalize(perceptron_score)

                conditional_order = stable_order(conditional, fallback)
                perceptron_order = stable_order(perceptron_score, fallback)
                self.record(
                    "decayed_conditional", target, actual, conditional_order
                )
                self.record(
                    "standalone_perceptron", target, actual, perceptron_order
                )
                for alpha in self.alphas:
                    hybrid_order = stable_order(
                        conditional_norm + alpha * perceptron_norm,
                        fallback,
                    )
                    self.record(method_for_alpha(alpha), target, actual, hybrid_order)

                self.routes[target] += len(actual)
                self.bin_routes[target] += len(actual)
                pending.append(
                    (
                        target,
                        route_features,
                        feature_ids,
                        actual_tuple,
                        actual,
                        perceptron_order,
                    )
                )

            for (
                target,
                route_features,
                feature_ids,
                actual_tuple,
                actual,
                perceptron_order,
            ) in pending:
                self.update_perceptron(
                    target, feature_ids, actual, perceptron_order
                )
                self.update_conditional(target, route_features, actual_tuple)

            if token_number % self.bin_size == 0 or token_number == len(tokens):
                self.flush_bin(token_number)

        layer_rows: list[dict[str, object]] = []
        for target, layer in enumerate(self.target_layers):
            route_count = max(1, int(self.routes[target]))
            method_recall: dict[str, dict[str, float]] = {}
            for method, method_index in self.method_index.items():
                method_recall[method] = {
                    str(budget): float(
                        self.hits[method_index, target, self.budget_index[budget]]
                        / route_count
                    )
                    for budget in self.budgets
                }

            best_alpha_8 = max(
                self.alphas,
                key=lambda alpha: method_recall[method_for_alpha(alpha)]["8"],
            )
            best_alpha_16 = max(
                self.alphas,
                key=lambda alpha: method_recall[method_for_alpha(alpha)]["16"],
            )
            best_alpha_32 = max(
                self.alphas,
                key=lambda alpha: method_recall[method_for_alpha(alpha)]["32"],
            )
            frequency = self.actual_frequency[target]
            entropy = entropy_bits(frequency)
            order = np.argsort(-frequency, kind="stable")
            static16_mass = float(np.sum(frequency[order[:16]]) / max(1, np.sum(frequency)))
            conditional16 = method_recall["decayed_conditional"]["16"]
            perceptron16 = method_recall["standalone_perceptron"]["16"]
            best_hybrid16 = method_recall[method_for_alpha(best_alpha_16)]["16"]
            if perceptron16 > conditional16 + 0.01:
                standalone_preference = "perceptron"
            elif conditional16 > perceptron16 + 0.01:
                standalone_preference = "conditional"
            else:
                standalone_preference = "approximately_tied"
            layer_rows.append(
                {
                    "layer": layer,
                    "entropy_bits": entropy,
                    "effective_experts": 2.0**entropy,
                    "static_top16_mass": static16_mass,
                    "method_recall": method_recall,
                    "best_alpha": {
                        "8": best_alpha_8,
                        "16": best_alpha_16,
                        "32": best_alpha_32,
                    },
                    "best_hybrid_gain_over_conditional": {
                        "8": method_recall[method_for_alpha(best_alpha_8)]["8"]
                        - method_recall["decayed_conditional"]["8"],
                        "16": best_hybrid16 - conditional16,
                        "32": method_recall[method_for_alpha(best_alpha_32)]["32"]
                        - method_recall["decayed_conditional"]["32"],
                    },
                    "standalone_preference_at_16": standalone_preference,
                    "perceptron_pair_updates": int(self.update_pairs[target]),
                    "perceptron_nonzero_weights": int(
                        np.count_nonzero(self.perceptron[target])
                    ),
                }
            )

        gains16 = [
            float(row["best_hybrid_gain_over_conditional"]["16"])
            for row in layer_rows
        ]
        entropies = [float(row["entropy_bits"]) for row in layer_rows]
        static_mass = [float(row["static_top16_mass"]) for row in layer_rows]
        conditional16_values = [
            float(row["method_recall"]["decayed_conditional"]["16"])
            for row in layer_rows
        ]
        perceptron16_values = [
            float(row["method_recall"]["standalone_perceptron"]["16"])
            for row in layer_rows
        ]
        best_hybrid16_values = [
            float(
                row["method_recall"][method_for_alpha(row["best_alpha"]["16"])]["16"]
            )
            for row in layer_rows
        ]

        preference_counts = Counter(
            str(row["standalone_preference_at_16"]) for row in layer_rows
        )
        alpha_counts = Counter(float(row["best_alpha"]["16"]) for row in layer_rows)
        highest_gain = sorted(
            layer_rows,
            key=lambda row: float(row["best_hybrid_gain_over_conditional"]["16"]),
            reverse=True,
        )[:10]
        lowest_gain = sorted(
            layer_rows,
            key=lambda row: float(row["best_hybrid_gain_over_conditional"]["16"]),
        )[:10]
        hardest = sorted(
            layer_rows,
            key=lambda row: float(
                row["method_recall"][method_for_alpha(row["best_alpha"]["16"])]["16"]
            ),
        )[:10]
        easiest = sorted(
            layer_rows,
            key=lambda row: float(
                row["method_recall"][method_for_alpha(row["best_alpha"]["16"])]["16"]
            ),
            reverse=True,
        )[:10]

        return {
            "tokens": len(tokens),
            "routed_layers": len(self.layers),
            "target_layers": len(self.target_layers),
            "separate_per_layer_heads": True,
            "shared_across_layers": [
                "feature layout",
                "update algorithm",
                "weight bounds",
            ],
            "not_shared_across_layers": [
                "conditional counts",
                "marginal frequencies",
                "perceptron weights",
                "best correction strength",
            ],
            "configuration": {
                "history": self.history,
                "alphas": list(self.alphas),
                "decay_interval": self.decay_interval,
                "decay_factor": self.decay_factor,
                "weight_limit": self.weight_limit,
                "bin_size": self.bin_size,
            },
            "per_layer": layer_rows,
            "learning_curve": self.bins,
            "population_summary": {
                "conditional_recall_at_16": summarize_values(conditional16_values),
                "standalone_perceptron_recall_at_16": summarize_values(
                    perceptron16_values
                ),
                "best_hybrid_recall_at_16": summarize_values(best_hybrid16_values),
                "best_hybrid_gain_at_16": summarize_values(gains16),
                "standalone_preference_counts": dict(preference_counts),
                "best_alpha_at_16_counts": {
                    str(key): value for key, value in sorted(alpha_counts.items())
                },
                "spearman": {
                    "entropy_vs_conditional_recall16": spearman(
                        entropies, conditional16_values
                    ),
                    "entropy_vs_best_hybrid_gain16": spearman(entropies, gains16),
                    "static_top16_mass_vs_conditional_recall16": spearman(
                        static_mass, conditional16_values
                    ),
                    "static_top16_mass_vs_best_hybrid_gain16": spearman(
                        static_mass, gains16
                    ),
                },
                "layers_with_hybrid_gain_above_1pp": sum(
                    gain > 0.01 for gain in gains16
                ),
                "layers_with_hybrid_gain_above_3pp": sum(
                    gain > 0.03 for gain in gains16
                ),
                "layers_with_hybrid_loss": sum(gain < 0.0 for gain in gains16),
            },
            "selected_layers": {
                "highest_hybrid_gain": highest_gain,
                "lowest_hybrid_gain": lowest_gain,
                "hardest_best_hybrid": hardest,
                "easiest_best_hybrid": easiest,
            },
            "state": {
                "perceptron_dense_int16_mib": self.perceptron.nbytes
                / (1024 * 1024),
                "perceptron_int8_equivalent_mib": self.perceptron.size
                / (1024 * 1024),
                "conditional_float32_analysis_mib": (
                    self.counts.nbytes
                    + self.source_counts.nbytes
                    + self.marginal.nbytes
                )
                / (1024 * 1024),
                "note": "Production counters can use saturating integer state; float32 is used here for decay experiments.",
            },
        }


def print_summary(result: dict[str, object]) -> None:
    summary = result["population_summary"]
    print(
        f"tokens={result['tokens']} target_layers={result['target_layers']} "
        f"separate_heads={result['separate_per_layer_heads']}"
    )
    for key in (
        "conditional_recall_at_16",
        "standalone_perceptron_recall_at_16",
        "best_hybrid_recall_at_16",
        "best_hybrid_gain_at_16",
    ):
        values = summary[key]
        multiplier = 100.0
        print(
            f"{key:38s} min={multiplier*values['min']:6.2f}% "
            f"median={multiplier*values['median']:6.2f}% "
            f"mean={multiplier*values['mean']:6.2f}% "
            f"max={multiplier*values['max']:6.2f}%"
        )
    print("standalone preference:", summary["standalone_preference_counts"])
    print("best alpha@16:", summary["best_alpha_at_16_counts"])
    print(
        "hybrid gain layers:",
        f">1pp={summary['layers_with_hybrid_gain_above_1pp']}",
        f">3pp={summary['layers_with_hybrid_gain_above_3pp']}",
        f"loss={summary['layers_with_hybrid_loss']}",
    )
    print("spearman:", summary["spearman"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--history", type=int, default=3)
    parser.add_argument(
        "--alphas",
        type=parse_csv_floats,
        default=(0.10, 0.20, 0.35, 0.50, 0.75, 1.00, 1.50),
    )
    parser.add_argument("--decay-interval", type=int, default=128)
    parser.add_argument("--decay-factor", type=float, default=0.5)
    parser.add_argument("--weight-limit", type=int, default=31)
    parser.add_argument("--bin-size", type=int, default=128)
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()

    steps, layers, tokens = load_trace(args.trace, args.max_steps)
    result = {
        "trace": str(args.trace),
        "step_range": [steps[0], steps[-1]],
        "methodology": {
            "evaluation": "prequential: predict and score before online update",
            "conditional": "one independent decayed three-lag table per target layer",
            "perceptron": "one independent bounded mistake-driven head per target layer",
            "hybrid": "normalized conditional score plus alpha times normalized perceptron residual",
            "authority": "predictors are cache/prefetch hints; the model router remains authoritative",
        },
        "analysis": PerLayerStudy(
            layers=layers,
            experts=args.experts,
            history=args.history,
            alphas=args.alphas,
            decay_interval=args.decay_interval,
            decay_factor=args.decay_factor,
            weight_limit=args.weight_limit,
            bin_size=args.bin_size,
        ).run(tokens),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print_summary(result["analysis"])
    if args.output:
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
