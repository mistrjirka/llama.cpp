#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


def pct(n: float, d: float) -> float:
    return 100.0 * n / d if d else 0.0


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def top_k(counter: collections.Counter[int], capacity: int) -> set[int]:
    return {expert for expert, _ in counter.most_common(capacity)}


def score_set(selected: tuple[int, ...], resident: set[int]) -> tuple[int, bool]:
    hits = sum(expert in resident for expert in selected)
    return hits, hits == len(selected)


@dataclass
class PolicyStats:
    route_hits: int = 0
    routes: int = 0
    full_layers: int = 0
    layers: int = 0
    full_tokens: int = 0
    tokens: int = 0
    per_layer_hits: collections.Counter[int] = field(default_factory=collections.Counter)
    per_layer_routes: collections.Counter[int] = field(default_factory=collections.Counter)
    per_layer_full: collections.Counter[int] = field(default_factory=collections.Counter)
    per_layer_steps: collections.Counter[int] = field(default_factory=collections.Counter)

    def add_layer(self, layer: int, selected: tuple[int, ...], resident: set[int]) -> bool:
        hits, full = score_set(selected, resident)
        self.route_hits += hits
        self.routes += len(selected)
        self.full_layers += int(full)
        self.layers += 1
        self.per_layer_hits[layer] += hits
        self.per_layer_routes[layer] += len(selected)
        self.per_layer_full[layer] += int(full)
        self.per_layer_steps[layer] += 1
        return full

    def finish_token(self, all_full: bool) -> None:
        self.tokens += 1
        self.full_tokens += int(all_full)

    def summary(self) -> dict[str, Any]:
        return {
            'route_hit_pct': pct(self.route_hits, self.routes),
            'full_layer_pct': pct(self.full_layers, self.layers),
            'full_token_pct': pct(self.full_tokens, self.tokens),
            'route_hits': self.route_hits,
            'routes': self.routes,
            'full_layers': self.full_layers,
            'layers': self.layers,
            'full_tokens': self.full_tokens,
            'tokens': self.tokens,
        }


def parse_trace(path: Path) -> tuple[dict[int, dict[int, dict[str, Any]]], list[dict[str, Any]], collections.Counter[str]]:
    routes: dict[int, dict[int, dict[str, Any]]] = collections.defaultdict(dict)
    lifecycle: list[dict[str, Any]] = []
    counts: collections.Counter[str] = collections.Counter()
    with path.open(errors='replace') as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                counts['parse_error'] += 1
                continue
            kind = str(event.get('event', '?'))
            counts[kind] += 1
            if kind == 'route' and event.get('phase') == 'decode':
                routes[int(event['step'])][int(event['layer'])] = event
            elif kind in {'admit', 'evict', 'ready_after_completion', 'reject'}:
                lifecycle.append(event)
    return routes, lifecycle, counts


def infer_capacities(routes: dict[int, dict[int, dict[str, Any]]], lifecycle: list[dict[str, Any]]) -> dict[int, int]:
    capacities: collections.Counter[int] = collections.Counter()
    for event in lifecycle:
        if event.get('event') == 'admit':
            layer = int(event['layer'])
            capacities[layer] = max(capacities[layer], int(event.get('slot', -1)) + 1)
    for layers in routes.values():
        for layer, event in layers.items():
            slots = [int(slot) for slot in event.get('hot_slots', []) if int(slot) >= 0]
            if slots:
                capacities[layer] = max(capacities[layer], max(slots) + 1)
    return dict(capacities)


def simulate(
    routes: dict[int, dict[int, dict[str, Any]]],
    capacities: dict[int, int],
) -> tuple[dict[str, PolicyStats], dict[str, Any]]:
    ordered_steps = sorted(routes)
    layers = sorted({layer for step in ordered_steps for layer in routes[step]})

    policies = {
        name: PolicyStats()
        for name in [
            'actual_ready',
            'previous_token_exact_set',
            'online_lru',
            'online_frequency',
            'online_markov',
            'offline_frequency_oracle',
        ]
    }

    all_frequency: dict[int, collections.Counter[int]] = collections.defaultdict(collections.Counter)
    for step in ordered_steps:
        for layer, event in routes[step].items():
            all_frequency[layer].update(int(expert) for expert in event['selected'])
    offline_resident = {
        layer: top_k(all_frequency[layer], capacities.get(layer, 0))
        for layer in layers
    }

    previous_selected: dict[int, tuple[int, ...]] = {}
    lru_order: dict[int, collections.OrderedDict[int, None]] = collections.defaultdict(collections.OrderedDict)
    frequency: dict[int, collections.Counter[int]] = collections.defaultdict(collections.Counter)
    transition: dict[int, dict[int, collections.Counter[int]]] = collections.defaultdict(
        lambda: collections.defaultdict(collections.Counter)
    )
    marginal: dict[int, collections.Counter[int]] = collections.defaultdict(collections.Counter)

    jaccards: dict[int, list[float]] = collections.defaultdict(list)
    next_overlap: dict[int, list[int]] = collections.defaultdict(list)
    reuse_distances: dict[int, list[int]] = collections.defaultdict(list)
    last_seen: dict[int, dict[int, int]] = collections.defaultdict(dict)

    for step in ordered_steps:
        token_full = {name: True for name in policies}
        for layer in layers:
            event = routes[step].get(layer)
            if event is None:
                token_full = {name: False for name in policies}
                continue
            selected = tuple(int(expert) for expert in event['selected'])
            capacity = capacities.get(layer, 0)

            actual_ready_count = int(event.get('ready_available', 0))
            actual_full = actual_ready_count == len(selected)
            actual_stats = policies['actual_ready']
            actual_stats.route_hits += actual_ready_count
            actual_stats.routes += len(selected)
            actual_stats.full_layers += int(actual_full)
            actual_stats.layers += 1
            actual_stats.per_layer_hits[layer] += actual_ready_count
            actual_stats.per_layer_routes[layer] += len(selected)
            actual_stats.per_layer_full[layer] += int(actual_full)
            actual_stats.per_layer_steps[layer] += 1
            token_full['actual_ready'] &= actual_full

            previous = set(previous_selected.get(layer, ()))
            token_full['previous_token_exact_set'] &= policies['previous_token_exact_set'].add_layer(
                layer, selected, previous
            )

            lru_resident = set(lru_order[layer])
            token_full['online_lru'] &= policies['online_lru'].add_layer(layer, selected, lru_resident)

            frequency_resident = top_k(frequency[layer], capacity)
            token_full['online_frequency'] &= policies['online_frequency'].add_layer(
                layer, selected, frequency_resident
            )

            markov_scores: collections.Counter[int] = collections.Counter()
            for prior in previous_selected.get(layer, ()):
                markov_scores.update(transition[layer][prior])
            # Marginal frequency breaks ties and supplies cold-start candidates.
            for expert, count in marginal[layer].items():
                markov_scores[expert] += count * 1e-6
            markov_resident = top_k(markov_scores, capacity)
            token_full['online_markov'] &= policies['online_markov'].add_layer(
                layer, selected, markov_resident
            )

            token_full['offline_frequency_oracle'] &= policies['offline_frequency_oracle'].add_layer(
                layer, selected, offline_resident[layer]
            )

            if layer in previous_selected:
                previous_set = set(previous_selected[layer])
                selected_set = set(selected)
                union = previous_set | selected_set
                jaccards[layer].append(len(previous_set & selected_set) / len(union) if union else 1.0)
                next_overlap[layer].append(len(previous_set & selected_set))
                for prior in previous_selected[layer]:
                    transition[layer][prior].update(selected)

            for expert in selected:
                if expert in last_seen[layer]:
                    reuse_distances[layer].append(step - last_seen[layer][expert])
                last_seen[layer][expert] = step

            for expert in selected:
                if expert in lru_order[layer]:
                    del lru_order[layer][expert]
                lru_order[layer][expert] = None
            while len(lru_order[layer]) > capacity:
                lru_order[layer].popitem(last=False)

            frequency[layer].update(selected)
            marginal[layer].update(selected)
            previous_selected[layer] = selected

        for name, stats in policies.items():
            stats.finish_token(token_full[name])

    layer_analysis = []
    actual = policies['actual_ready']
    markov = policies['online_markov']
    offline = policies['offline_frequency_oracle']
    for layer in layers:
        distances = reuse_distances[layer]
        layer_analysis.append({
            'layer': layer,
            'capacity': capacities.get(layer, 0),
            'actual_route_hit_pct': pct(actual.per_layer_hits[layer], actual.per_layer_routes[layer]),
            'actual_full_layer_pct': pct(actual.per_layer_full[layer], actual.per_layer_steps[layer]),
            'markov_route_hit_pct': pct(markov.per_layer_hits[layer], markov.per_layer_routes[layer]),
            'markov_full_layer_pct': pct(markov.per_layer_full[layer], markov.per_layer_steps[layer]),
            'offline_frequency_route_hit_pct': pct(offline.per_layer_hits[layer], offline.per_layer_routes[layer]),
            'offline_frequency_full_layer_pct': pct(offline.per_layer_full[layer], offline.per_layer_steps[layer]),
            'previous_token_mean_overlap_of_8': (
                sum(next_overlap[layer]) / len(next_overlap[layer]) if next_overlap[layer] else 0.0
            ),
            'previous_token_mean_jaccard': (
                sum(jaccards[layer]) / len(jaccards[layer]) if jaccards[layer] else 0.0
            ),
            'reuse_distance_p50_tokens': percentile([float(v) for v in distances], 0.50),
            'reuse_distance_p90_tokens': percentile([float(v) for v in distances], 0.90),
            'reuse_distance_p99_tokens': percentile([float(v) for v in distances], 0.99),
        })

    diagnostics = {
        'steps': len(ordered_steps),
        'layers': len(layers),
        'layer_ids': layers,
        'capacities': capacities,
        'per_layer': layer_analysis,
    }
    return policies, diagnostics


def lifecycle_analysis(lifecycle: list[dict[str, Any]]) -> dict[str, Any]:
    counts = collections.Counter(str(event.get('event', '?')) for event in lifecycle)
    reject_reasons = collections.Counter(
        str(event.get('reason', '?')) for event in lifecycle if event.get('event') == 'reject'
    )
    evictions_by_layer = collections.Counter(
        int(event['layer']) for event in lifecycle if event.get('event') == 'evict'
    )
    admissions_by_layer = collections.Counter(
        int(event['layer']) for event in lifecycle if event.get('event') == 'admit'
    )
    return {
        'counts': dict(counts),
        'reject_reasons': dict(reject_reasons),
        'admissions_by_layer': dict(sorted(admissions_by_layer.items())),
        'evictions_by_layer': dict(sorted(evictions_by_layer.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('trace', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()

    routes, lifecycle, event_counts = parse_trace(args.trace)
    capacities = infer_capacities(routes, lifecycle)
    policies, diagnostics = simulate(routes, capacities)
    result = {
        'trace': str(args.trace),
        'event_counts': dict(event_counts),
        'policy_summary': {name: stats.summary() for name, stats in policies.items()},
        'diagnostics': diagnostics,
        'lifecycle': lifecycle_analysis(lifecycle),
    }
    encoded = json.dumps(result, indent=2)
    print(encoded)
    if args.output:
        args.output.write_text(encoded + '\n')


if __name__ == '__main__':
    main()
