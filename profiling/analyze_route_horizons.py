#!/usr/bin/env python3
"""Measure how route-only prediction accuracy changes with layer lead time.

For horizon H, the predictor observes the selected experts at layer L-H and
predicts the authoritative top-k route at layer L.  All predictions are
prequential: score first, then update the conditional table.  The output pairs
accuracy with measured route-decision lead time from the same trace.

This isolates the central prefetch tradeoff:

- larger H gives more CPU-to-GPU transfer time;
- smaller H usually gives more route correlation.

The model router remains authoritative.  Predictions are cache/prefetch hints.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import statistics
from typing import Sequence

import numpy as np


Route = tuple[int, ...]


def parse_csv_ints(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected comma-separated integers")
    return values


def load_trace(path: Path, max_steps: int | None = None):
    by_step: dict[int, dict[int, Route]] = collections.defaultdict(dict)
    times: dict[int, dict[int, int]] = collections.defaultdict(dict)
    with path.open(errors="replace") as stream:
        for line in stream:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") != "route" or event.get("phase") != "decode":
                continue
            step = int(event["step"])
            layer = int(event["layer"])
            by_step[step][layer] = tuple(
                dict.fromkeys(int(value) for value in event["selected"])
            )
            timestamp = event.get("route_decision_mono_ns")
            if timestamp is not None:
                times[step][layer] = int(timestamp)

    if not by_step:
        raise ValueError(f"no decode routes in {path}")
    max_layers = max(len(layer_map) for layer_map in by_step.values())
    layer_sets = collections.Counter(
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
    timestamps = [{layer: times[step][layer] for layer in layers if layer in times[step]} for step in steps]
    return steps, layers, tokens, timestamps


def stable_order(score: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    ids = np.arange(score.size, dtype=np.int32)
    return np.lexsort((ids, -fallback, -score))


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[int(fraction * (len(ordered) - 1))]


def evaluate_horizon(
    *,
    layers: Sequence[int],
    tokens: Sequence[dict[int, Route]],
    timestamps: Sequence[dict[int, int]],
    horizon: int,
    experts: int,
    budgets: Sequence[int],
    decay_interval: int,
    decay_factor: float,
    warmup: int,
) -> dict[str, object]:
    target_count = len(layers) - horizon
    counts = np.zeros((target_count, experts, experts), dtype=np.float32)
    source_counts = np.zeros((target_count, experts), dtype=np.float32)
    marginal = np.zeros((target_count, experts), dtype=np.float32)

    all_hits = {budget: 0 for budget in budgets}
    all_routes = 0
    eval_hits = {budget: 0 for budget in budgets}
    eval_routes = 0
    bins: list[dict[str, object]] = []
    bin_hits = {budget: 0 for budget in budgets}
    bin_routes = 0
    bin_size = 64

    for token_index, token in enumerate(tokens, start=1):
        if decay_interval > 0 and token_index > 1 and (token_index - 1) % decay_interval == 0:
            counts *= decay_factor
            source_counts *= decay_factor
            marginal *= decay_factor

        for target_index in range(target_count):
            source_layer = layers[target_index]
            target_layer = layers[target_index + horizon]
            source = np.asarray(token[source_layer], dtype=np.int64)
            actual = set(token[target_layer])
            prior = marginal[target_index]
            prior_probability = (prior + 0.25) / (float(np.sum(prior)) + 0.25 * experts)
            denominators = source_counts[target_index, source]
            valid = denominators > 0
            score = 0.35 * prior_probability
            if np.any(valid):
                rows = counts[target_index, source[valid]] / denominators[valid, None]
                score = score + np.mean(rows, axis=0)
            order = stable_order(score, prior)

            all_routes += len(actual)
            bin_routes += len(actual)
            if token_index > warmup:
                eval_routes += len(actual)
            for budget in budgets:
                predicted = set(int(value) for value in order[:budget])
                hits = len(actual & predicted)
                all_hits[budget] += hits
                bin_hits[budget] += hits
                if token_index > warmup:
                    eval_hits[budget] += hits

        # Update only after scoring the complete token.
        for target_index in range(target_count):
            source_layer = layers[target_index]
            target_layer = layers[target_index + horizon]
            source = np.asarray(token[source_layer], dtype=np.int64)
            actual = np.asarray(token[target_layer], dtype=np.int64)
            counts[target_index][np.ix_(source, actual)] += 1.0
            source_counts[target_index, source] += 1.0
            marginal[target_index, actual] += 1.0

        if token_index % bin_size == 0 or token_index == len(tokens):
            bins.append(
                {
                    "token_end": token_index,
                    "route_recall": {
                        str(budget): bin_hits[budget] / max(1, bin_routes)
                        for budget in budgets
                    },
                }
            )
            bin_hits = {budget: 0 for budget in budgets}
            bin_routes = 0

    lead_times: list[float] = []
    for token_times in timestamps:
        for target_index in range(target_count):
            source_layer = layers[target_index]
            target_layer = layers[target_index + horizon]
            if source_layer not in token_times or target_layer not in token_times:
                continue
            delta = (token_times[target_layer] - token_times[source_layer]) / 1e6
            if delta >= 0:
                lead_times.append(delta)

    return {
        "horizon_layers": horizon,
        "target_layers": target_count,
        "all_prequential_recall": {
            str(budget): all_hits[budget] / max(1, all_routes) for budget in budgets
        },
        "post_warmup_tokens": max(0, len(tokens) - warmup),
        "post_warmup_recall": {
            str(budget): eval_hits[budget] / max(1, eval_routes) for budget in budgets
        },
        "lead_time_ms": {
            "samples": len(lead_times),
            "median": statistics.median(lead_times) if lead_times else None,
            "mean": statistics.mean(lead_times) if lead_times else None,
            "p10": percentile(lead_times, 0.10) if lead_times else None,
            "p90": percentile(lead_times, 0.90) if lead_times else None,
        },
        "learning_curve_64_tokens": bins,
        "state_mib": (counts.nbytes + source_counts.nbytes + marginal.nbytes) / (1024 * 1024),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--horizons", type=parse_csv_ints, default=(1, 2, 3, 4, 6, 8, 12, 16))
    parser.add_argument("--budgets", type=parse_csv_ints, default=(8, 16, 32))
    parser.add_argument("--warmup", type=int, default=256)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--decay-interval", type=int, default=128)
    parser.add_argument("--decay-factor", type=float, default=0.5)
    args = parser.parse_args()

    steps, layers, tokens, timestamps = load_trace(args.trace, args.max_steps)
    horizons = [value for value in args.horizons if 0 < value < len(layers)]
    result = {
        "trace": str(args.trace),
        "steps": len(tokens),
        "step_range": [steps[0], steps[-1]],
        "layers": [layers[0], layers[-1]],
        "methodology": {
            "predictor": "decayed per-layer conditional co-occurrence from exactly one earlier routed layer",
            "evaluation": "prequential; score before online update",
            "interpretation": "accuracy/lead-time horizon baseline, not a final production predictor",
        },
        "horizons": [
            evaluate_horizon(
                layers=layers,
                tokens=tokens,
                timestamps=timestamps,
                horizon=horizon,
                experts=args.experts,
                budgets=args.budgets,
                decay_interval=args.decay_interval,
                decay_factor=args.decay_factor,
                warmup=args.warmup,
            )
            for horizon in horizons
        ],
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("horizon lead-ms post-warmup R@8 R@16 R@32")
    for item in result["horizons"]:
        lead = item["lead_time_ms"]["median"]
        recalls = item["post_warmup_recall"]
        print(
            f"{item['horizon_layers']:7d} {lead if lead is not None else 0:7.2f} "
            f"{100*recalls.get('8', 0):6.2f}% {100*recalls.get('16', 0):6.2f}% "
            f"{100*recalls.get('32', 0):6.2f}%"
        )
    if args.output:
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
