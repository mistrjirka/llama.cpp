#!/usr/bin/env python3
"""Analyze MoE route compressibility and lightweight cross-layer predictors.

The script deliberately uses train/test time ordering. It reports intrinsic
static top-k coverage and prequential online prediction from the previous
three routed layers. It is a trace analysis, not an end-to-end latency claim.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

Route = Tuple[int, ...]
StepRoutes = Dict[int, Dict[int, Route]]


def load_routes(path: Path) -> StepRoutes:
    by_step: StepRoutes = collections.defaultdict(dict)
    with path.open(errors="replace") as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") != "route":
                continue
            by_step[int(event["step"])][int(event["layer"])] = tuple(
                int(expert) for expert in event["selected"]
            )
    return dict(by_step)


def complete_steps(routes: StepRoutes) -> Tuple[List[int], List[int]]:
    layer_counts = collections.Counter()
    for layer_map in routes.values():
        layer_counts.update(layer_map.keys())
    layers = sorted(layer for layer, _ in layer_counts.most_common())
    expected = set(layers)
    complete = sorted(step for step, layer_map in routes.items() if set(layer_map) == expected)
    return complete, layers


def entropy_bits(counts: np.ndarray) -> float:
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    probs = counts[counts > 0] / total
    return float(-(probs * np.log2(probs)).sum())


def js_divergence_bits(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    a /= a.sum()
    b /= b.sum()
    m = 0.5 * (a + b)

    def kl(p: np.ndarray, q: np.ndarray) -> float:
        mask = p > 0
        return float((p[mask] * np.log2(p[mask] / q[mask])).sum())

    return 0.5 * kl(a, m) + 0.5 * kl(b, m)


def frequencies(
    routes: StepRoutes,
    steps: Sequence[int],
    layers: Sequence[int],
    n_experts: int,
) -> Dict[int, np.ndarray]:
    result = {layer: np.zeros(n_experts, dtype=np.int64) for layer in layers}
    for step in steps:
        for layer in layers:
            result[layer][list(routes[step][layer])] += 1
    return result


def static_coverage(
    routes: StepRoutes,
    train_steps: Sequence[int],
    test_steps: Sequence[int],
    layers: Sequence[int],
    n_experts: int,
    budgets: Sequence[int],
) -> Dict[str, float]:
    train_freq = frequencies(routes, train_steps, layers, n_experts)
    hits = {budget: 0 for budget in budgets}
    total = 0
    for step in test_steps:
        for layer in layers:
            actual = routes[step][layer]
            total += len(actual)
            order = np.argsort(-train_freq[layer], kind="stable")
            for budget in budgets:
                predicted = set(int(x) for x in order[:budget])
                hits[budget] += sum(expert in predicted for expert in actual)
    return {str(budget): hits[budget] / total for budget in budgets}


def affinity_tables(
    routes: StepRoutes,
    train_steps: Sequence[int],
    layers: Sequence[int],
    n_experts: int,
    max_history: int,
) -> Tuple[Dict[int, np.ndarray], Dict[Tuple[int, int], np.ndarray]]:
    prior = frequencies(routes, train_steps, layers, n_experts)
    affinity: Dict[Tuple[int, int], np.ndarray] = {}
    layer_index = {layer: index for index, layer in enumerate(layers)}
    for layer in layers:
        target_index = layer_index[layer]
        for distance in range(1, max_history + 1):
            if target_index - distance < 0:
                continue
            affinity[(layer, distance)] = np.zeros(
                (n_experts, n_experts), dtype=np.float32
            )

    for step in train_steps:
        layer_map = routes[step]
        for layer in layers:
            target = layer_map[layer]
            target_index = layer_index[layer]
            for distance in range(1, max_history + 1):
                if target_index - distance < 0:
                    continue
                source_layer = layers[target_index - distance]
                table = affinity[(layer, distance)]
                for source in layer_map[source_layer]:
                    table[source, list(target)] += 1.0

    # Convert rows to smoothed conditional probabilities. The small prior term
    # makes unseen source experts back off instead of producing arbitrary ties.
    for (layer, _distance), table in affinity.items():
        row_sums = table.sum(axis=1, keepdims=True)
        prior_prob = prior[layer].astype(np.float32)
        prior_prob = (prior_prob + 0.25) / (prior_prob.sum() + 0.25 * n_experts)
        table += 0.25 * prior_prob[None, :]
        row_sums = table.sum(axis=1, keepdims=True)
        table /= row_sums
    return prior, affinity


def evaluate_affinity(
    routes: StepRoutes,
    train_steps: Sequence[int],
    test_steps: Sequence[int],
    layers: Sequence[int],
    n_experts: int,
    budgets: Sequence[int],
    history_lengths: Sequence[int],
) -> Dict[str, Dict[str, float]]:
    prior, affinity = affinity_tables(
        routes, train_steps, layers, n_experts, max(history_lengths)
    )
    layer_index = {layer: index for index, layer in enumerate(layers)}
    totals = {history: 0 for history in history_lengths}
    hits = {
        history: {budget: 0 for budget in budgets} for history in history_lengths
    }
    for step in test_steps:
        layer_map = routes[step]
        for layer in layers:
            target_index = layer_index[layer]
            actual = layer_map[layer]
            prior_prob = prior[layer].astype(np.float32)
            prior_prob = (prior_prob + 0.25) / (prior_prob.sum() + 0.25 * n_experts)
            for history in history_lengths:
                score = 0.35 * prior_prob.copy()
                used = 0
                for distance in range(1, history + 1):
                    if target_index - distance < 0:
                        continue
                    source_layer = layers[target_index - distance]
                    source = layer_map[source_layer]
                    # Geometric recency weighting resembles a short tagged-history
                    # ensemble while keeping the computation sparse.
                    weight = 0.72 ** (distance - 1)
                    score += weight * affinity[(layer, distance)][list(source)].mean(axis=0)
                    used += 1
                if used == 0:
                    score = prior_prob
                order = np.argsort(-score, kind="stable")
                totals[history] += len(actual)
                for budget in budgets:
                    predicted = set(int(x) for x in order[:budget])
                    hits[history][budget] += sum(expert in predicted for expert in actual)
    return {
        str(history): {
            str(budget): hits[history][budget] / totals[history]
            for budget in budgets
        }
        for history in history_lengths
    }


def online_perceptron(
    routes: StepRoutes,
    warmup_steps: Sequence[int],
    eval_steps: Sequence[int],
    layers: Sequence[int],
    n_experts: int,
    budget: int,
    max_history: int = 3,
    weight_limit: int = 31,
) -> Dict[str, float]:
    """Prequential structured perceptron with sparse previous-layer features.

    One predictor exists per target layer. Each target expert has an int8-like
    weight for the bias and every (history distance, expert ID) feature. Only
    missed positives and the same number of highest-ranked false positives are
    updated, and weights saturate. This mirrors confidence-gated, bounded CPU
    predictors more closely than a large general-purpose neural network.
    """

    feature_count = 1 + max_history * n_experts
    weights = {
        layer: np.zeros((n_experts, feature_count), dtype=np.int16)
        for layer in layers
    }
    layer_index = {layer: index for index, layer in enumerate(layers)}
    update_count = 0

    def features(step: int, layer: int) -> np.ndarray:
        index = layer_index[layer]
        values = [0]
        for distance in range(1, max_history + 1):
            if index - distance < 0:
                continue
            source_layer = layers[index - distance]
            values.extend(
                1 + (distance - 1) * n_experts + expert
                for expert in routes[step][source_layer]
            )
        return np.array(values, dtype=np.int64)

    def train_one(step: int, layer: int) -> Tuple[int, int, float]:
        nonlocal update_count
        actual = set(routes[step][layer])
        feat = features(step, layer)
        score = weights[layer][:, feat].sum(axis=1)
        order = np.argsort(-score, kind="stable")
        predicted = set(int(x) for x in order[:budget])
        hit = len(actual & predicted)
        missed = [expert for expert in actual if expert not in predicted]
        false = [int(expert) for expert in order[:budget] if int(expert) not in actual]
        # Pairwise ranking update; no update for already-correct experts.
        for positive, negative in zip(missed, false):
            weights[layer][positive, feat] = np.minimum(
                weight_limit, weights[layer][positive, feat] + 1
            )
            weights[layer][negative, feat] = np.maximum(
                -weight_limit, weights[layer][negative, feat] - 1
            )
            update_count += 1
        # Margin is useful as a confidence signal for deciding whether to prefetch.
        kth = float(score[order[min(budget - 1, len(order) - 1)]])
        next_score = float(score[order[min(budget, len(order) - 1)]])
        return hit, len(actual), kth - next_score

    for step in warmup_steps:
        for layer in layers:
            train_one(step, layer)

    hits = 0
    total = 0
    margins: List[float] = []
    eval_updates_before = update_count
    for step in eval_steps:
        for layer in layers:
            hit, count, margin = train_one(step, layer)
            hits += hit
            total += count
            margins.append(margin)

    nonzero = sum(int(np.count_nonzero(table)) for table in weights.values())
    total_weights = len(layers) * n_experts * feature_count
    return {
        "coverage": hits / total,
        "updates_during_eval": update_count - eval_updates_before,
        "mean_pair_updates_per_layer_token": (update_count - eval_updates_before)
        / (len(eval_steps) * len(layers)),
        "mean_topk_margin": float(np.mean(margins)) if margins else 0.0,
        "nonzero_weights": nonzero,
        "dense_weight_count": total_weights,
        "dense_int8_mib": total_weights / (1024 * 1024),
        "sparse_nonzero_fraction": nonzero / total_weights,
    }


def analyze(
    name: str,
    path: Path,
    n_experts: int,
    train_count: int,
    budgets: Sequence[int],
) -> Dict[str, object]:
    routes = load_routes(path)
    steps, layers = complete_steps(routes)
    if len(steps) <= train_count:
        raise ValueError(f"{name}: only {len(steps)} complete steps for train_count={train_count}")
    train_steps = steps[:train_count]
    test_steps = steps[train_count:]
    all_freq = frequencies(routes, steps, layers, n_experts)
    first_half = frequencies(routes, steps[: len(steps) // 2], layers, n_experts)
    second_half = frequencies(routes, steps[len(steps) // 2 :], layers, n_experts)

    entropies = [entropy_bits(all_freq[layer]) for layer in layers]
    effective = [2.0 ** value for value in entropies]
    unique = [int(np.count_nonzero(all_freq[layer])) for layer in layers]
    js = [js_divergence_bits(first_half[layer], second_half[layer]) for layer in layers]

    result: Dict[str, object] = {
        "name": name,
        "trace": str(path),
        "complete_steps": len(steps),
        "train_steps": len(train_steps),
        "test_steps": len(test_steps),
        "layer_count": len(layers),
        "layer_range": [layers[0], layers[-1]],
        "n_experts": n_experts,
        "routes_per_layer_token": len(routes[steps[-1]][layers[-1]]),
        "compressibility": {
            "mean_entropy_bits": float(np.mean(entropies)),
            "median_entropy_bits": float(np.median(entropies)),
            "mean_normalized_entropy": float(np.mean(entropies)) / math.log2(n_experts),
            "mean_effective_experts": float(np.mean(effective)),
            "median_effective_experts": float(np.median(effective)),
            "mean_unique_experts": float(np.mean(unique)),
            "mean_half_to_half_js_bits": float(np.mean(js)),
            "median_half_to_half_js_bits": float(np.median(js)),
            "static_test_coverage": static_coverage(
                routes, train_steps, test_steps, layers, n_experts, budgets
            ),
        },
        "cross_layer_affinity_test_coverage": evaluate_affinity(
            routes,
            train_steps,
            test_steps,
            layers,
            n_experts,
            budgets,
            history_lengths=(1, 2, 3),
        ),
        "online_perceptron": {
            str(budget): online_perceptron(
                routes,
                train_steps,
                test_steps,
                layers,
                n_experts,
                budget=budget,
                max_history=3,
            )
            for budget in budgets
            if budget in (8, 16, 32)
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--glm-trace",
        type=Path,
        default=Path("profiling/glm52-scale-2026-07-27/routes96.jsonl"),
    )
    parser.add_argument(
        "--qwen-trace",
        type=Path,
        default=Path("profiling/long-coding-cache-2026-07-27/dynamic-trace-r2.jsonl"),
    )
    args = parser.parse_args()
    budgets = (4, 8, 16, 32, 64)
    result = {
        "methodology": {
            "coverage_definition": "fraction of the eight actual routes contained in the predicted top-k candidate set",
            "glm_split": "first 64 complete decode steps train, remaining complete steps test",
            "qwen_split": "first 512 complete decode steps train, remaining complete steps test",
            "predictor_inputs": "selected expert IDs from the previous one, two, or three routed layers of the same token",
            "online_update": "bounded structured perceptron; predict then update, with first train segment used as warmup",
            "limitations": [
                "GLM trace is one prompt and has only 95 complete layer steps.",
                "Trace coverage is not equivalent to end-to-end speedup; transfer lead time and graph topology remain decisive.",
                "The online perceptron uses expert IDs only, not hidden-state or router-logit features.",
            ],
        },
        "models": [
            analyze("GLM-5.2", args.glm_trace, 256, 64, budgets),
            analyze("Qwen3.5-122B-A10B", args.qwen_trace, 256, 512, budgets),
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
