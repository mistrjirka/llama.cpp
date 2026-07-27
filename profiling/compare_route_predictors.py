#!/usr/bin/env python3
"""Compare online MoE route predictors on chronological route traces.

The evaluator is prequential: each route is predicted before its authoritative
router result is used for an online update.  It is intended to answer three
questions that a single train/test number cannot:

1. Does a predictor continue improving as more decode tokens arrive?
2. Which predictor family wins at an early (+3), middle (+2), or near (+1)
   layer horizon?
3. Are errors complementary enough for a CPU-inspired hybrid predictor to be
   worthwhile?

The predictors are route hints only.  None of them replace the model router.

Requires NumPy.  The repository's analysis environment can be created with:

    python3 -m venv /workspace/.venv-predictor
    /workspace/.venv-predictor/bin/pip install numpy

Example:

    /workspace/.venv-predictor/bin/python profiling/compare_route_predictors.py \
        profiling/glm52-scale-2026-07-27/routes-long512.jsonl \
        --output profiling/glm52-scale-2026-07-27/predictor-comparison-long.json
"""

from __future__ import annotations

import argparse
import collections
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import statistics
import time
from typing import Iterable, Sequence

import numpy as np


Route = tuple[int, ...]


def load_trace(
    path: Path,
    *,
    max_steps: int | None = None,
) -> tuple[list[int], list[int], list[dict[int, Route]], list[dict[int, int]]]:
    """Load complete decode steps and route-decision timestamps."""

    routes_by_step: dict[int, dict[int, Route]] = collections.defaultdict(dict)
    times_by_step: dict[int, dict[int, int]] = collections.defaultdict(dict)
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
            selected = tuple(dict.fromkeys(int(value) for value in event["selected"]))
            routes_by_step[step][layer] = selected
            timestamp = event.get("route_decision_mono_ns")
            if timestamp is None and event.get("wall_us") is not None:
                timestamp = int(event["wall_us"]) * 1000
            if timestamp is not None:
                times_by_step[step][layer] = int(timestamp)

    if not routes_by_step:
        raise ValueError(f"no decode route records found in {path}")

    max_layer_count = max(len(layer_map) for layer_map in routes_by_step.values())
    layer_sets = collections.Counter(
        frozenset(layer_map)
        for layer_map in routes_by_step.values()
        if len(layer_map) == max_layer_count
    )
    expected = set(layer_sets.most_common(1)[0][0])
    layers = sorted(expected)
    steps = sorted(step for step, layer_map in routes_by_step.items() if set(layer_map) == expected)
    if max_steps is not None:
        steps = steps[:max_steps]
    if not steps:
        raise ValueError(f"no complete route steps found in {path}")

    tokens = [{layer: routes_by_step[step][layer] for layer in layers} for step in steps]
    timestamps = [{layer: times_by_step[step][layer] for layer in layers if layer in times_by_step[step]} for step in steps]
    return steps, layers, tokens, timestamps


def parse_csv_ints(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected one or more comma-separated integers")
    return values


def parse_csv_floats(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected one or more comma-separated floats")
    return values


def alpha_method_name(alpha: float) -> str:
    return f"hybrid_decayed_mistake_a{alpha:g}".replace(".", "p")


def stable_ranking(score: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    experts = np.arange(score.size, dtype=np.int32)
    # lexsort uses the final key as primary.  Expert ID gives deterministic ties.
    return np.lexsort((experts, -fallback, -score))


def normalized(score: np.ndarray) -> np.ndarray:
    centered = score.astype(np.float32, copy=False) - float(np.mean(score))
    scale = float(np.std(centered))
    if scale < 1e-6:
        return centered
    return centered / scale


def saturating_add_at(
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
    weights[np.ix_(row_index, features)] = block.astype(weights.dtype)


@dataclass
class MetricCounter:
    events: int = 0
    routes: int = 0
    hits: dict[int, int] = field(default_factory=lambda: collections.defaultdict(int))
    complete: dict[int, int] = field(default_factory=lambda: collections.defaultdict(int))

    def add(self, actual: set[int], ranking: Sequence[int], budgets: Sequence[int]) -> None:
        self.events += 1
        self.routes += len(actual)
        for budget in budgets:
            predicted = set(int(value) for value in ranking[:budget])
            hits = len(actual & predicted)
            self.hits[budget] += hits
            self.complete[budget] += int(hits == len(actual))

    def as_dict(self, budgets: Sequence[int]) -> dict[str, object]:
        return {
            "events": self.events,
            "routes": self.routes,
            "budgets": {
                str(budget): {
                    "route_recall": self.hits[budget] / max(1, self.routes),
                    "precision": self.hits[budget] / max(1, self.events * budget),
                    "mean_hits_per_layer": self.hits[budget] / max(1, self.events),
                    "complete_rate": self.complete[budget] / max(1, self.events),
                }
                for budget in budgets
            },
        }


@dataclass
class TaggedEntry:
    tag: int
    observations: int
    usefulness: int
    target_counts: np.ndarray


class TaggedHistoryTable:
    """Small TAGE-like tagged context table with a sparse payload."""

    def __init__(
        self,
        *,
        target_layers: int,
        experts: int,
        table_sizes: Sequence[int],
        group_size: int,
        min_observations: int,
    ) -> None:
        self.target_layers = target_layers
        self.experts = experts
        self.table_sizes = tuple(table_sizes)
        self.group_size = group_size
        self.min_observations = min_observations
        self.tables: list[list[dict[int, TaggedEntry]]] = [
            [dict() for _ in self.table_sizes] for _ in range(target_layers)
        ]
        self.lookups = 0
        self.matches = [0 for _ in self.table_sizes]
        self.providers = [0 for _ in self.table_sizes]
        self.allocations = [0 for _ in self.table_sizes]
        self.collisions = [0 for _ in self.table_sizes]

    @staticmethod
    def _mix64(value: int) -> int:
        value ^= value >> 30
        value *= 0xBF58476D1CE4E5B9
        value &= (1 << 64) - 1
        value ^= value >> 27
        value *= 0x94D049BB133111EB
        value &= (1 << 64) - 1
        value ^= value >> 31
        return value

    def _route_signature(self, route: Route) -> int:
        mask = 0
        for expert in route:
            group = expert // self.group_size
            mask |= 1 << group
        return mask

    def _context_hash(self, features: Sequence[Route], history: int) -> int:
        value = 0x9E3779B97F4A7C15
        # features are [L-1, L-2, L-3].  The table history uses the nearest H.
        for distance, route in enumerate(features[:history], start=1):
            signature = self._route_signature(route)
            value ^= self._mix64(signature + distance * 0xD6E8FEB86659FD93)
            value = ((value << 17) | (value >> 47)) & ((1 << 64) - 1)
        return self._mix64(value)

    def lookup(
        self,
        target_index: int,
        features: Sequence[Route],
    ) -> tuple[np.ndarray | None, tuple[int, int, int] | None]:
        self.lookups += 1
        # Longest matching provider wins, like TAGE.
        for level in range(len(self.table_sizes) - 1, -1, -1):
            history = level + 1
            hashed = self._context_hash(features, history)
            size = self.table_sizes[level]
            index = hashed % size
            tag = (hashed >> int(math.log2(size))) & 0xFFFF
            entry = self.tables[target_index][level].get(index)
            if entry is None or entry.tag != tag:
                continue
            self.matches[level] += 1
            if entry.observations < self.min_observations:
                continue
            self.providers[level] += 1
            score = entry.target_counts.astype(np.float32) / max(1, entry.observations)
            return score, (level, index, tag)
        return None, None

    def update(
        self,
        target_index: int,
        features: Sequence[Route],
        actual: Sequence[int],
        provider: tuple[int, int, int] | None,
        provider_delta: int,
    ) -> None:
        for level, size in enumerate(self.table_sizes):
            history = level + 1
            hashed = self._context_hash(features, history)
            index = hashed % size
            tag = (hashed >> int(math.log2(size))) & 0xFFFF
            table = self.tables[target_index][level]
            entry = table.get(index)
            if entry is None:
                entry = TaggedEntry(
                    tag=tag,
                    observations=0,
                    usefulness=0,
                    target_counts=np.zeros(self.experts, dtype=np.uint16),
                )
                table[index] = entry
                self.allocations[level] += 1
            elif entry.tag != tag:
                self.collisions[level] += 1
                if entry.usefulness > 0:
                    entry.usefulness -= 1
                    continue
                entry = TaggedEntry(
                    tag=tag,
                    observations=0,
                    usefulness=0,
                    target_counts=np.zeros(self.experts, dtype=np.uint16),
                )
                table[index] = entry
                self.allocations[level] += 1

            entry.observations = min(65535, entry.observations + 1)
            current = entry.target_counts[np.asarray(actual, dtype=np.int64)].astype(np.uint32) + 1
            entry.target_counts[np.asarray(actual, dtype=np.int64)] = np.minimum(current, 65535).astype(np.uint16)

        if provider is not None and provider_delta:
            level, index, tag = provider
            entry = self.tables[target_index][level].get(index)
            if entry is not None and entry.tag == tag:
                entry.usefulness = int(np.clip(entry.usefulness + provider_delta, 0, 3))

    def stats(self) -> dict[str, object]:
        entries = [
            sum(len(self.tables[target][level]) for target in range(self.target_layers))
            for level in range(len(self.table_sizes))
        ]
        payload_bytes = sum(value * self.experts * 2 for value in entries)
        return {
            "table_sizes": list(self.table_sizes),
            "group_size": self.group_size,
            "min_observations": self.min_observations,
            "lookups": self.lookups,
            "matches": self.matches,
            "providers": self.providers,
            "allocations": self.allocations,
            "collisions": self.collisions,
            "live_entries": entries,
            "estimated_counter_payload_mib": payload_bytes / (1024 * 1024),
        }


class PredictorComparison:
    def __init__(
        self,
        *,
        layers: Sequence[int],
        experts: int,
        max_history: int,
        budgets: Sequence[int],
        bin_size: int,
        decay_interval: int,
        decay_factor: float,
        pair_buckets: int,
        pair_weight: float,
        temporal_weight: float,
        hybrid_alpha: float,
        hybrid_alphas: Sequence[float],
        margin: int,
        weight_limit: int,
        tage_sizes: Sequence[int],
        tage_group_size: int,
        tage_min_observations: int,
    ) -> None:
        if len(layers) <= max_history:
            raise ValueError("trace has fewer layers than the requested history")
        if pair_buckets <= 0 or pair_buckets & (pair_buckets - 1):
            raise ValueError("pair_buckets must be a power of two")
        if any(size <= 0 or size & (size - 1) for size in tage_sizes):
            raise ValueError("all TAGE table sizes must be powers of two")

        self.layers = list(layers)
        self.experts = experts
        self.max_history = max_history
        self.target_layers = len(layers) - max_history
        self.budgets = tuple(sorted(set(budgets)))
        self.bin_size = bin_size
        self.decay_interval = decay_interval
        self.decay_factor = decay_factor
        self.pair_buckets = pair_buckets
        self.pair_weight = pair_weight
        self.temporal_weight = temporal_weight
        self.hybrid_alpha = hybrid_alpha
        self.hybrid_alphas = tuple(hybrid_alphas)
        self.margin = margin
        self.weight_limit = weight_limit
        self.expert_ids = np.arange(experts, dtype=np.int32)

        shape = (self.target_layers, max_history, experts, experts)
        self.cross_counts = np.zeros(shape, dtype=np.uint16)
        self.cross_source_counts = np.zeros((self.target_layers, max_history, experts), dtype=np.uint16)
        self.marginal = np.zeros((self.target_layers, experts), dtype=np.uint32)

        self.decayed_cross = np.zeros(shape, dtype=np.float32)
        self.decayed_source = np.zeros((self.target_layers, max_history, experts), dtype=np.float32)
        self.decayed_marginal = np.zeros((self.target_layers, experts), dtype=np.float32)

        self.temporal_counts = np.zeros((self.target_layers, experts, experts), dtype=np.uint16)
        self.temporal_source = np.zeros((self.target_layers, experts), dtype=np.uint16)
        self.previous_same: list[Route | None] = [None] * self.target_layers

        self.within_pair_counts = np.zeros((self.target_layers, pair_buckets, experts), dtype=np.uint16)
        self.within_pair_source = np.zeros((self.target_layers, pair_buckets), dtype=np.uint16)
        self.cross_pair_counts = np.zeros((self.target_layers, pair_buckets, experts), dtype=np.uint16)
        self.cross_pair_source = np.zeros((self.target_layers, pair_buckets), dtype=np.uint16)

        cross_feature_count = 1 + max_history * experts
        st_feature_count = 1 + (max_history + 1) * experts
        self.perceptron_mistake = np.zeros(
            (self.target_layers, experts, cross_feature_count), dtype=np.int16
        )
        self.perceptron_margin = np.zeros(
            (self.target_layers, experts, cross_feature_count), dtype=np.int16
        )
        self.perceptron_st = np.zeros(
            (self.target_layers, experts, st_feature_count), dtype=np.int16
        )

        self.tournament_chooser = np.zeros(self.target_layers, dtype=np.int8)
        self.tage = TaggedHistoryTable(
            target_layers=self.target_layers,
            experts=experts,
            table_sizes=tage_sizes,
            group_size=tage_group_size,
            min_observations=tage_min_observations,
        )

        base_methods = [
            "frequency",
            "markov_lag3_early",
            "markov_lags23_mid",
            "markov_lags123_near",
            "markov_decayed",
            "markov_pair_interactions",
            "same_layer_temporal",
            "spatiotemporal_markov",
            "perceptron_mistake",
            "perceptron_margin",
            "perceptron_spatiotemporal",
            "hybrid_statistical_corrector",
            "hybrid_pair_statistical_corrector",
            "tournament_markov_perceptron",
            "tage_like",
            "tage_sc",
        ]
        base_methods.extend(alpha_method_name(alpha) for alpha in self.hybrid_alphas)
        base_methods.extend([
            "oracle_union_markov_perceptron",
            "oracle_best_markov_perceptron",
        ])
        self.methods = tuple(base_methods)
        self.total_metrics = {name: MetricCounter() for name in self.methods}
        self.bin_metrics = {name: MetricCounter() for name in self.methods}
        self.bins: list[dict[str, object]] = []
        self.update_pairs = collections.Counter()
        self.start_time = time.perf_counter()

    def _pair_bucket(self, first: int, second: int, salt: int) -> int:
        value = (first * 0x9E3779B1) ^ (second * 0x85EBCA6B) ^ salt
        value ^= value >> 16
        return value & (self.pair_buckets - 1)

    def _markov_score(
        self,
        target: int,
        features: Sequence[Route],
        lags: Sequence[int],
        *,
        decayed: bool = False,
    ) -> np.ndarray:
        if decayed:
            marginal = self.decayed_marginal[target]
            counts = self.decayed_cross[target]
            source_counts = self.decayed_source[target]
        else:
            marginal = self.marginal[target].astype(np.float32)
            counts = self.cross_counts[target]
            source_counts = self.cross_source_counts[target]

        prior = (marginal + 0.25) / (float(np.sum(marginal)) + 0.25 * self.experts)
        score = 0.35 * prior.astype(np.float32)
        weights = (1.0, 0.72, 0.72 * 0.72)
        for lag in lags:
            source = np.asarray(features[lag], dtype=np.int64)
            denominators = source_counts[lag, source].astype(np.float32)
            valid = denominators > 0
            if not np.any(valid):
                continue
            rows = counts[lag, source[valid]].astype(np.float32)
            rows /= denominators[valid, None]
            score += weights[lag] * np.mean(rows, axis=0)
        return score

    def _temporal_score(self, target: int) -> np.ndarray:
        previous = self.previous_same[target]
        marginal = self.marginal[target].astype(np.float32)
        prior = (marginal + 0.25) / (float(np.sum(marginal)) + 0.25 * self.experts)
        if not previous:
            return prior
        source = np.asarray(previous, dtype=np.int64)
        denominators = self.temporal_source[target, source].astype(np.float32)
        valid = denominators > 0
        if not np.any(valid):
            return prior
        rows = self.temporal_counts[target, source[valid]].astype(np.float32)
        rows /= denominators[valid, None]
        return 0.25 * prior + np.mean(rows, axis=0)

    def _pair_score(self, target: int, features: Sequence[Route], base: np.ndarray) -> np.ndarray:
        score = base.copy()
        nearest = sorted(set(features[0]))
        within_buckets = {
            self._pair_bucket(nearest[first], nearest[second], 0xA5A5)
            for first in range(len(nearest))
            for second in range(first + 1, len(nearest))
        }
        if within_buckets:
            indices = np.fromiter(within_buckets, dtype=np.int64)
            denominators = self.within_pair_source[target, indices].astype(np.float32)
            valid = denominators > 0
            if np.any(valid):
                rows = self.within_pair_counts[target, indices[valid]].astype(np.float32)
                rows /= denominators[valid, None]
                score += self.pair_weight * np.mean(rows, axis=0)

        cross_buckets = {
            self._pair_bucket(older, newer, 0x5A5A)
            for older in features[1]
            for newer in features[0]
        }
        if cross_buckets:
            indices = np.fromiter(cross_buckets, dtype=np.int64)
            denominators = self.cross_pair_source[target, indices].astype(np.float32)
            valid = denominators > 0
            if np.any(valid):
                rows = self.cross_pair_counts[target, indices[valid]].astype(np.float32)
                rows /= denominators[valid, None]
                score += self.pair_weight * np.mean(rows, axis=0)
        return score

    def _feature_indices(self, features: Sequence[Route], previous: Route | None = None) -> np.ndarray:
        values = [0]
        for lag, route in enumerate(features):
            values.extend(1 + lag * self.experts + expert for expert in route)
        if previous:
            base = 1 + self.max_history * self.experts
            values.extend(base + expert for expert in previous)
        return np.asarray(values, dtype=np.int64)

    def _perceptron_score(self, weights: np.ndarray, target: int, feature_ids: np.ndarray) -> np.ndarray:
        return np.sum(weights[target, :, feature_ids], axis=0).astype(np.float32)

    def _record(
        self,
        rankings: dict[str, np.ndarray],
        actual: set[int],
    ) -> None:
        for name, ranking in rankings.items():
            self.total_metrics[name].add(actual, ranking, self.budgets)
            self.bin_metrics[name].add(actual, ranking, self.budgets)

    def _update_perceptron_mistake(
        self,
        weights: np.ndarray,
        target: int,
        features: np.ndarray,
        actual: set[int],
        ranking: np.ndarray,
        key: str,
    ) -> None:
        update_budget = min(8, len(actual))
        predicted = [int(value) for value in ranking[:update_budget]]
        missed = [expert for expert in actual if expert not in predicted]
        false = [expert for expert in predicted if expert not in actual]
        pairs = min(len(missed), len(false))
        saturating_add_at(weights[target], missed[:pairs], features, +1, self.weight_limit)
        saturating_add_at(weights[target], false[:pairs], features, -1, self.weight_limit)
        self.update_pairs[key] += pairs

    def _update_perceptron_margin(
        self,
        weights: np.ndarray,
        target: int,
        features: np.ndarray,
        actual: set[int],
        score: np.ndarray,
        key: str,
    ) -> None:
        negatives = [int(value) for value in stable_ranking(score, self.marginal[target]) if int(value) not in actual]
        if not negatives:
            return
        negative = negatives[0]
        positives = [expert for expert in actual if score[expert] <= score[negative] + self.margin]
        if not positives:
            return
        # Update each weak positive against the highest-scoring false candidate.
        saturating_add_at(weights[target], positives, features, +1, self.weight_limit)
        saturating_add_at(weights[target], [negative], features, -len(positives), self.weight_limit)
        self.update_pairs[key] += len(positives)

    def _update_counts(
        self,
        target: int,
        features: Sequence[Route],
        actual: Route,
    ) -> None:
        actual_index = np.asarray(actual, dtype=np.int64)
        self.marginal[target, actual_index] += 1
        self.decayed_marginal[target, actual_index] += 1.0
        for lag, route in enumerate(features):
            source = np.asarray(route, dtype=np.int64)
            # Counts are small for the tested traces, so uint16 saturation is not
            # expected.  Explicit clipping keeps longer repeated studies safe.
            block = self.cross_counts[target, lag][np.ix_(source, actual_index)].astype(np.uint32) + 1
            self.cross_counts[target, lag][np.ix_(source, actual_index)] = np.minimum(block, 65535).astype(np.uint16)
            src = self.cross_source_counts[target, lag, source].astype(np.uint32) + 1
            self.cross_source_counts[target, lag, source] = np.minimum(src, 65535).astype(np.uint16)
            self.decayed_cross[target, lag][np.ix_(source, actual_index)] += 1.0
            self.decayed_source[target, lag, source] += 1.0

        previous = self.previous_same[target]
        if previous:
            source = np.asarray(previous, dtype=np.int64)
            block = self.temporal_counts[target][np.ix_(source, actual_index)].astype(np.uint32) + 1
            self.temporal_counts[target][np.ix_(source, actual_index)] = np.minimum(block, 65535).astype(np.uint16)
            src = self.temporal_source[target, source].astype(np.uint32) + 1
            self.temporal_source[target, source] = np.minimum(src, 65535).astype(np.uint16)

        nearest = sorted(set(features[0]))
        within_buckets = {
            self._pair_bucket(nearest[first], nearest[second], 0xA5A5)
            for first in range(len(nearest))
            for second in range(first + 1, len(nearest))
        }
        for bucket in within_buckets:
            self.within_pair_source[target, bucket] = min(
                65535, int(self.within_pair_source[target, bucket]) + 1
            )
            values = self.within_pair_counts[target, bucket, actual_index].astype(np.uint32) + 1
            self.within_pair_counts[target, bucket, actual_index] = np.minimum(values, 65535).astype(np.uint16)

        cross_buckets = {
            self._pair_bucket(older, newer, 0x5A5A)
            for older in features[1]
            for newer in features[0]
        }
        for bucket in cross_buckets:
            self.cross_pair_source[target, bucket] = min(
                65535, int(self.cross_pair_source[target, bucket]) + 1
            )
            values = self.cross_pair_counts[target, bucket, actual_index].astype(np.uint32) + 1
            self.cross_pair_counts[target, bucket, actual_index] = np.minimum(values, 65535).astype(np.uint16)

        self.previous_same[target] = actual

    def _flush_bin(self, token_end: int) -> None:
        token_start = token_end - self.bin_size + 1
        if self.bins:
            token_start = int(self.bins[-1]["token_end"]) + 1
        self.bins.append(
            {
                "token_start": token_start,
                "token_end": token_end,
                "methods": {
                    name: metric.as_dict(self.budgets)
                    for name, metric in self.bin_metrics.items()
                },
            }
        )
        self.bin_metrics = {name: MetricCounter() for name in self.methods}

    def run(self, tokens: Sequence[dict[int, Route]]) -> dict[str, object]:
        for token_index, token in enumerate(tokens, start=1):
            if self.decay_interval > 0 and token_index > 1 and (token_index - 1) % self.decay_interval == 0:
                self.decayed_cross *= self.decay_factor
                self.decayed_source *= self.decay_factor
                self.decayed_marginal *= self.decay_factor

            for target in range(self.target_layers):
                layer_index = target + self.max_history
                features = [token[self.layers[layer_index - lag]] for lag in range(1, self.max_history + 1)]
                actual_tuple = token[self.layers[layer_index]]
                actual = set(actual_tuple)
                fallback = self.marginal[target].astype(np.float32)

                frequency = fallback
                markov_early = self._markov_score(target, features, (2,))
                markov_mid = self._markov_score(target, features, (1, 2))
                markov_near = self._markov_score(target, features, (0, 1, 2))
                markov_decayed = self._markov_score(target, features, (0, 1, 2), decayed=True)
                temporal = self._temporal_score(target)
                pair = self._pair_score(target, features, markov_near)
                spatiotemporal = markov_near + self.temporal_weight * temporal

                cross_features = self._feature_indices(features)
                st_features = self._feature_indices(features, self.previous_same[target])
                p_mistake = self._perceptron_score(
                    self.perceptron_mistake, target, cross_features
                )
                p_margin = self._perceptron_score(
                    self.perceptron_margin, target, cross_features
                )
                p_st = self._perceptron_score(self.perceptron_st, target, st_features)

                hybrid = normalized(markov_near) + self.hybrid_alpha * normalized(p_margin)
                hybrid_pair = normalized(pair) + self.hybrid_alpha * normalized(p_st)
                decayed_base = normalized(markov_decayed)
                mistake_corrector = normalized(p_mistake)

                markov_ranking = stable_ranking(markov_near, fallback)
                perceptron_ranking = stable_ranking(p_margin, fallback)
                if self.tournament_chooser[target] >= 0:
                    tournament = markov_ranking
                else:
                    tournament = perceptron_ranking

                tage_provider, provider_ref = self.tage.lookup(target, features)
                if tage_provider is None:
                    tage_score = markov_near
                else:
                    tage_score = 0.35 * normalized(markov_near) + normalized(tage_provider)
                tage_sc = normalized(tage_score) + self.hybrid_alpha * normalized(p_st)

                rankings = {
                    "frequency": stable_ranking(frequency, fallback),
                    "markov_lag3_early": stable_ranking(markov_early, fallback),
                    "markov_lags23_mid": stable_ranking(markov_mid, fallback),
                    "markov_lags123_near": markov_ranking,
                    "markov_decayed": stable_ranking(markov_decayed, fallback),
                    "markov_pair_interactions": stable_ranking(pair, fallback),
                    "same_layer_temporal": stable_ranking(temporal, fallback),
                    "spatiotemporal_markov": stable_ranking(spatiotemporal, fallback),
                    "perceptron_mistake": stable_ranking(p_mistake, fallback),
                    "perceptron_margin": perceptron_ranking,
                    "perceptron_spatiotemporal": stable_ranking(p_st, fallback),
                    "hybrid_statistical_corrector": stable_ranking(hybrid, fallback),
                    "hybrid_pair_statistical_corrector": stable_ranking(hybrid_pair, fallback),
                    "tournament_markov_perceptron": tournament,
                    "tage_like": stable_ranking(tage_score, fallback),
                    "tage_sc": stable_ranking(tage_sc, fallback),
                }
                for alpha in self.hybrid_alphas:
                    rankings[alpha_method_name(alpha)] = stable_ranking(
                        decayed_base + alpha * mistake_corrector,
                        fallback,
                    )

                union: list[int] = []
                for rank_index in range(max(self.budgets)):
                    for provider_ranking in (markov_ranking, perceptron_ranking):
                        expert = int(provider_ranking[rank_index])
                        if expert not in union:
                            union.append(expert)
                union.extend(expert for expert in range(self.experts) if expert not in union)
                rankings["oracle_union_markov_perceptron"] = np.asarray(union, dtype=np.int32)
                markov_hits = len(actual & set(int(value) for value in markov_ranking[:8]))
                perceptron_hits = len(actual & set(int(value) for value in perceptron_ranking[:8]))
                rankings["oracle_best_markov_perceptron"] = (
                    markov_ranking if markov_hits >= perceptron_hits else perceptron_ranking
                )

                self._record(rankings, actual)

                # CPU-style tournament chooser: positive favors Markov; negative
                # favors the perceptron.  Update only when providers disagree in
                # realized utility.
                if markov_hits > perceptron_hits:
                    self.tournament_chooser[target] = min(7, int(self.tournament_chooser[target]) + 1)
                elif perceptron_hits > markov_hits:
                    self.tournament_chooser[target] = max(-8, int(self.tournament_chooser[target]) - 1)

                tage_hits = len(actual & set(int(value) for value in rankings["tage_like"][:8]))
                provider_delta = 1 if tage_hits > markov_hits else -1 if tage_hits < markov_hits else 0

                self._update_perceptron_mistake(
                    self.perceptron_mistake,
                    target,
                    cross_features,
                    actual,
                    rankings["perceptron_mistake"],
                    "mistake",
                )
                self._update_perceptron_margin(
                    self.perceptron_margin,
                    target,
                    cross_features,
                    actual,
                    p_margin,
                    "margin",
                )
                self._update_perceptron_margin(
                    self.perceptron_st,
                    target,
                    st_features,
                    actual,
                    p_st,
                    "spatiotemporal_margin",
                )
                self.tage.update(
                    target,
                    features,
                    actual_tuple,
                    provider_ref,
                    provider_delta,
                )
                self._update_counts(target, features, actual_tuple)

            if token_index % self.bin_size == 0 or token_index == len(tokens):
                self._flush_bin(token_index)

        elapsed = time.perf_counter() - self.start_time
        dense_bytes = (
            self.cross_counts.nbytes
            + self.cross_source_counts.nbytes
            + self.marginal.nbytes
            + self.decayed_cross.nbytes
            + self.decayed_source.nbytes
            + self.decayed_marginal.nbytes
            + self.temporal_counts.nbytes
            + self.temporal_source.nbytes
            + self.within_pair_counts.nbytes
            + self.within_pair_source.nbytes
            + self.cross_pair_counts.nbytes
            + self.cross_pair_source.nbytes
            + self.perceptron_mistake.nbytes
            + self.perceptron_margin.nbytes
            + self.perceptron_st.nbytes
        )
        return {
            "tokens": len(tokens),
            "routed_layers": len(self.layers),
            "target_layers": self.target_layers,
            "experts": self.experts,
            "history": self.max_history,
            "budgets": list(self.budgets),
            "bin_size": self.bin_size,
            "configuration": {
                "decay_interval": self.decay_interval,
                "decay_factor": self.decay_factor,
                "pair_buckets": self.pair_buckets,
                "pair_weight": self.pair_weight,
                "temporal_weight": self.temporal_weight,
                "hybrid_alpha": self.hybrid_alpha,
                "hybrid_alphas": list(self.hybrid_alphas),
                "perceptron_margin": self.margin,
                "perceptron_weight_limit": self.weight_limit,
            },
            "methods": {
                name: metric.as_dict(self.budgets)
                for name, metric in self.total_metrics.items()
            },
            "learning_curve": self.bins,
            "perceptron_updates": dict(self.update_pairs),
            "tournament": {
                "layers_favoring_markov": int(np.sum(self.tournament_chooser >= 0)),
                "layers_favoring_perceptron": int(np.sum(self.tournament_chooser < 0)),
                "chooser_values": [int(value) for value in self.tournament_chooser],
            },
            "tage": self.tage.stats(),
            "state": {
                "dense_state_mib_for_all_experimental_models": dense_bytes / (1024 * 1024),
                "note": "A production implementation would instantiate selected models only; this total intentionally includes every ablation.",
            },
            "offline_runtime": {
                "wall_seconds": elapsed,
                "milliseconds_per_trace_token": 1000.0 * elapsed / max(1, len(tokens)),
                "note": "Python/NumPy analysis time is not production predictor overhead.",
            },
        }


def timing_summary(
    layers: Sequence[int],
    timestamps: Sequence[dict[int, int]],
    distances: Sequence[int] = (1, 2, 3, 4, 6, 8),
) -> dict[str, object]:
    result: dict[str, object] = {}
    for distance in distances:
        values: list[float] = []
        for token in timestamps:
            for index in range(distance, len(layers)):
                source = layers[index - distance]
                target = layers[index]
                if source not in token or target not in token:
                    continue
                delta_ms = (token[target] - token[source]) / 1e6
                if delta_ms >= 0:
                    values.append(delta_ms)
        if not values:
            continue
        values.sort()
        result[str(distance)] = {
            "samples": len(values),
            "median_ms": statistics.median(values),
            "mean_ms": statistics.mean(values),
            "p10_ms": values[int(0.10 * (len(values) - 1))],
            "p90_ms": values[int(0.90 * (len(values) - 1))],
        }
    return result


def print_summary(result: dict[str, object]) -> None:
    budgets = result["budgets"]
    methods = result["methods"]
    print(
        f"tokens={result['tokens']} routed_layers={result['routed_layers']} "
        f"target_layers={result['target_layers']}"
    )
    header = "method                                  " + " ".join(
        f"R@{budget:>2}" for budget in budgets
    ) + "  C@8"
    print(header)
    for name, data in methods.items():
        values = data["budgets"]
        recalls = " ".join(
            f"{100.0 * values[str(budget)]['route_recall']:5.1f}%" for budget in budgets
        )
        complete = values.get("8", {}).get("complete_rate", 0.0)
        print(f"{name:39s} {recalls} {100.0 * complete:5.1f}%")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--history", type=int, default=3)
    parser.add_argument("--budgets", type=parse_csv_ints, default=(8, 16, 32))
    parser.add_argument("--bin-size", type=int, default=32)
    parser.add_argument("--decay-interval", type=int, default=128)
    parser.add_argument("--decay-factor", type=float, default=0.5)
    parser.add_argument("--pair-buckets", type=int, default=512)
    parser.add_argument("--pair-weight", type=float, default=0.30)
    parser.add_argument("--temporal-weight", type=float, default=0.45)
    parser.add_argument("--hybrid-alpha", type=float, default=0.35)
    parser.add_argument(
        "--hybrid-alphas",
        type=parse_csv_floats,
        default=(0.10, 0.20, 0.35, 0.50, 0.75, 1.00),
        help="decayed-Markov plus mistake-perceptron correction strengths",
    )
    parser.add_argument("--perceptron-margin", type=int, default=1)
    parser.add_argument("--weight-limit", type=int, default=31)
    parser.add_argument("--tage-sizes", type=parse_csv_ints, default=(128, 512, 2048))
    parser.add_argument("--tage-group-size", type=int, default=16)
    parser.add_argument("--tage-min-observations", type=int, default=2)
    args = parser.parse_args()

    steps, layers, tokens, timestamps = load_trace(args.trace, max_steps=args.max_steps)
    comparison = PredictorComparison(
        layers=layers,
        experts=args.experts,
        max_history=args.history,
        budgets=args.budgets,
        bin_size=args.bin_size,
        decay_interval=args.decay_interval,
        decay_factor=args.decay_factor,
        pair_buckets=args.pair_buckets,
        pair_weight=args.pair_weight,
        temporal_weight=args.temporal_weight,
        hybrid_alpha=args.hybrid_alpha,
        hybrid_alphas=args.hybrid_alphas,
        margin=args.perceptron_margin,
        weight_limit=args.weight_limit,
        tage_sizes=args.tage_sizes,
        tage_group_size=args.tage_group_size,
        tage_min_observations=args.tage_min_observations,
    )
    result = {
        "trace": str(args.trace),
        "step_range": [steps[0], steps[-1]],
        "methodology": {
            "evaluation": "prequential: predict, score, then update",
            "coverage": "fraction of authoritative selected experts contained in the predicted top-k set",
            "authority": "the model router remains authoritative; predictors are cache/prefetch hints only",
            "oracle_methods": "diagnostic upper bounds, not deployable predictors",
        },
        "layer_lead_time": timing_summary(layers, timestamps),
        "comparison": comparison.run(tokens),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print_summary(result["comparison"])
    if args.output:
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
