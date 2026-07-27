#!/usr/bin/env python3
"""Evaluate a CPU-style indexed history library for MoE route prediction.

A literal key containing the exact top-8 expert sets from the previous three
layers almost never repeats.  This experiment folds each source route into a
coarse group-membership signature, then indexes a bounded per-target-layer LRU
library.  The library acts as a tagged exception provider over a decayed
conditional base predictor.

Evaluation is prequential: predict, score, then update.  The model router is
always authoritative; this predicts cache/prefetch candidates only.
"""

from __future__ import annotations

import argparse
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

import numpy as np


Route = tuple[int, ...]


def parse_csv_ints(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected comma-separated integers")
    return values


def load(path: Path, max_steps: int | None = None):
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
        raise ValueError(f"no decode routes in {path}")
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


def route_signature(route: Route, group_size: int) -> int:
    signature = 0
    for expert in route:
        signature |= 1 << (expert // group_size)
    return signature


def rank(score: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    ids = np.arange(score.size, dtype=np.int32)
    return np.lexsort((ids, -fallback, -score))


@dataclass
class Entry:
    observations: int
    usefulness: int
    counts: np.ndarray


@dataclass
class Totals:
    routes: int = 0
    hits8: int = 0
    hits16: int = 0
    hits32: int = 0
    provider_matches: int = 0
    providers: int = 0
    provider_routes: int = 0
    provider_hits16: int = 0
    allocations: int = 0
    evictions: int = 0
    useful_provider_wins: int = 0
    harmful_provider_losses: int = 0

    def add_hits(self, actual: set[int], order: Sequence[int]) -> None:
        self.routes += len(actual)
        self.hits8 += len(actual & set(int(value) for value in order[:8]))
        self.hits16 += len(actual & set(int(value) for value in order[:16]))
        self.hits32 += len(actual & set(int(value) for value in order[:32]))

    def as_dict(self) -> dict[str, object]:
        return {
            "route_recall": {
                "8": self.hits8 / max(1, self.routes),
                "16": self.hits16 / max(1, self.routes),
                "32": self.hits32 / max(1, self.routes),
            },
            "provider_match_rate": self.provider_matches / max(1, self.provider_routes),
            "provider_use_rate": self.providers / max(1, self.provider_routes),
            "provider_only_recall_at_16": self.provider_hits16 / max(1, self.provider_matches * 8),
            "allocations": self.allocations,
            "evictions": self.evictions,
            "useful_provider_wins": self.useful_provider_wins,
            "harmful_provider_losses": self.harmful_provider_losses,
        }


class Experiment:
    def __init__(
        self,
        *,
        layers: Sequence[int],
        experts: int,
        history: int,
        group_size: int,
        capacity: int,
        min_observations: int,
        provider_weight: float,
        provider_policy: str,
        decay_interval: int,
        decay_factor: float,
    ) -> None:
        self.layers = list(layers)
        self.experts = experts
        self.history = history
        self.group_size = group_size
        self.capacity = capacity
        self.min_observations = min_observations
        self.provider_weight = provider_weight
        self.provider_policy = provider_policy
        self.decay_interval = decay_interval
        self.decay_factor = decay_factor
        self.target_count = len(layers) - history
        self.tables: list[OrderedDict[tuple[int, ...], Entry]] = [
            OrderedDict() for _ in range(self.target_count)
        ]
        self.base_counts = np.zeros(
            (self.target_count, history, experts, experts), dtype=np.float32
        )
        self.base_sources = np.zeros(
            (self.target_count, history, experts), dtype=np.float32
        )
        self.marginal = np.zeros((self.target_count, experts), dtype=np.float32)
        self.totals = Totals()

    def key(self, features: Sequence[Route]) -> tuple[int, ...]:
        return tuple(route_signature(route, self.group_size) for route in features)

    def base_score(self, target: int, features: Sequence[Route]) -> np.ndarray:
        prior = self.marginal[target]
        prior_probability = (prior + 0.25) / (
            float(np.sum(prior)) + 0.25 * self.experts
        )
        score = 0.35 * prior_probability
        weights = (1.0, 0.72, 0.5184)
        for lag, route in enumerate(features):
            source = np.asarray(route, dtype=np.int64)
            denominators = self.base_sources[target, lag, source]
            valid = denominators > 0
            if not np.any(valid):
                continue
            rows = self.base_counts[target, lag, source[valid]] / denominators[valid, None]
            score = score + weights[lag] * np.mean(rows, axis=0)
        return score

    def update_base(self, target: int, features: Sequence[Route], actual: Route) -> None:
        actual_index = np.asarray(actual, dtype=np.int64)
        self.marginal[target, actual_index] += 1.0
        for lag, route in enumerate(features):
            source = np.asarray(route, dtype=np.int64)
            self.base_counts[target, lag][np.ix_(source, actual_index)] += 1.0
            self.base_sources[target, lag, source] += 1.0

    def run(self, tokens: Sequence[dict[int, Route]]) -> dict[str, object]:
        base_total = Totals()
        for token_index, token in enumerate(tokens, start=1):
            if (
                self.decay_interval > 0
                and token_index > 1
                and (token_index - 1) % self.decay_interval == 0
            ):
                self.base_counts *= self.decay_factor
                self.base_sources *= self.decay_factor
                self.marginal *= self.decay_factor

            pending = []
            for target in range(self.target_count):
                layer_index = target + self.history
                features = [
                    token[self.layers[layer_index - distance]]
                    for distance in range(1, self.history + 1)
                ]
                actual_tuple = token[self.layers[layer_index]]
                actual = set(actual_tuple)
                base = self.base_score(target, features)
                base_order = rank(base, self.marginal[target])
                base_total.add_hits(actual, base_order)

                key = self.key(features)
                table = self.tables[target]
                entry = table.get(key)
                provider_order = None
                combined_order = base_order
                self.totals.provider_routes += 1
                if entry is not None:
                    table.move_to_end(key)
                if entry is not None and entry.observations >= self.min_observations:
                    self.totals.provider_matches += 1
                    provider_probability = entry.counts.astype(np.float32) / max(
                        1, entry.observations
                    )
                    provider_order = rank(provider_probability, self.marginal[target])
                    self.totals.provider_hits16 += len(
                        actual & set(int(value) for value in provider_order[:16])
                    )
                    combined = base + self.provider_weight * provider_probability
                    shadow_combined_order = rank(combined, self.marginal[target])
                    use_provider = (
                        self.provider_policy == "always"
                        or entry.usefulness > 0
                    )
                    if use_provider:
                        self.totals.providers += 1
                        combined_order = shadow_combined_order
                    base_hits = len(actual & set(int(value) for value in base_order[:16]))
                    combined_hits = len(
                        actual & set(int(value) for value in shadow_combined_order[:16])
                    )
                    if combined_hits > base_hits:
                        entry.usefulness = min(3, entry.usefulness + 1)
                        self.totals.useful_provider_wins += 1
                    elif combined_hits < base_hits:
                        entry.usefulness = max(0, entry.usefulness - 1)
                        self.totals.harmful_provider_losses += 1

                self.totals.add_hits(actual, combined_order)
                pending.append((target, key, features, actual_tuple))

            # Train only after scoring every target in the token.
            for target, key, features, actual_tuple in pending:
                table = self.tables[target]
                entry = table.get(key)
                if entry is None:
                    if len(table) >= self.capacity:
                        # Prefer an old non-useful entry.  If every entry has
                        # usefulness, age the oldest one and defer allocation.
                        victim = next(
                            (candidate for candidate, value in table.items() if value.usefulness == 0),
                            None,
                        )
                        if victim is None:
                            oldest = next(iter(table))
                            table[oldest].usefulness = max(
                                0, table[oldest].usefulness - 1
                            )
                        else:
                            del table[victim]
                            self.totals.evictions += 1
                    if len(table) < self.capacity:
                        entry = Entry(
                            observations=0,
                            usefulness=0,
                            counts=np.zeros(self.experts, dtype=np.uint16),
                        )
                        table[key] = entry
                        self.totals.allocations += 1
                if entry is not None:
                    entry.observations = min(65535, entry.observations + 1)
                    indices = np.asarray(actual_tuple, dtype=np.int64)
                    values = entry.counts[indices].astype(np.uint32) + 1
                    entry.counts[indices] = np.minimum(values, 65535).astype(np.uint16)
                    table.move_to_end(key)
                self.update_base(target, features, actual_tuple)

        live_entries = sum(len(table) for table in self.tables)
        result = self.totals.as_dict()
        result.update(
            {
                "base_route_recall": base_total.as_dict()["route_recall"],
                "history": self.history,
                "group_size": self.group_size,
                "groups": (self.experts + self.group_size - 1) // self.group_size,
                "capacity_per_layer": self.capacity,
                "min_observations": self.min_observations,
                "provider_weight": self.provider_weight,
                "provider_policy": self.provider_policy,
                "live_entries": live_entries,
                "estimated_library_mib": live_entries
                * (self.experts * 2 + 16)
                / (1024 * 1024),
            }
        )
        return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--histories", type=parse_csv_ints, default=(1, 2, 3))
    parser.add_argument("--group-sizes", type=parse_csv_ints, default=(16, 32, 64))
    parser.add_argument("--capacities", type=parse_csv_ints, default=(32, 128))
    parser.add_argument("--min-observations", type=int, default=2)
    parser.add_argument("--provider-weight", type=float, default=0.75)
    parser.add_argument(
        "--provider-policy",
        choices=("always", "useful"),
        default="useful",
        help="always override on a mature match, or require positive usefulness learned in shadow mode",
    )
    parser.add_argument("--decay-interval", type=int, default=128)
    parser.add_argument("--decay-factor", type=float, default=0.5)
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()

    steps, layers, tokens = load(args.trace, args.max_steps)
    experiments = []
    for history in args.histories:
        if history >= len(layers):
            continue
        for group_size in args.group_sizes:
            for capacity in args.capacities:
                result = Experiment(
                    layers=layers,
                    experts=args.experts,
                    history=history,
                    group_size=group_size,
                    capacity=capacity,
                    min_observations=args.min_observations,
                    provider_weight=args.provider_weight,
                    provider_policy=args.provider_policy,
                    decay_interval=args.decay_interval,
                    decay_factor=args.decay_factor,
                ).run(tokens)
                experiments.append(result)
                print(
                    f"h={history} group={group_size:2d} cap={capacity:3d} "
                    f"match={100*result['provider_match_rate']:5.1f}% "
                    f"use={100*result['provider_use_rate']:5.1f}% "
                    f"base16={100*result['base_route_recall']['16']:5.1f}% "
                    f"library16={100*result['route_recall']['16']:5.1f}% "
                    f"mem={result['estimated_library_mib']:5.1f} MiB"
                )

    output = {
        "trace": str(args.trace),
        "steps": len(tokens),
        "step_range": [steps[0], steps[-1]],
        "methodology": {
            "history_key": "unordered expert sets folded into coarse expert-group bitmasks",
            "provider": "bounded per-layer tagged/LRU exception library over a decayed conditional base",
            "evaluation": "prequential: predict, score, then update",
        },
        "experiments": experiments,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2) + "\n")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
