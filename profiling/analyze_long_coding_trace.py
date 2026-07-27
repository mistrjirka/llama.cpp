#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path
from typing import Any


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('trace', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()

    events: list[dict[str, Any]] = []
    parse_errors = 0
    with args.trace.open(errors='replace') as handle:
        for line in handle:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                parse_errors += 1

    counts = collections.Counter(event.get('event', '?') for event in events)

    transfers = collections.defaultdict(lambda: {'groups': 0, 'bytes': 0, 'issue_us': 0.0,
                                                   'source_sync_us': 0.0, 'destination_sync_us': 0.0,
                                                   'fallback': 0})
    submits = collections.defaultdict(lambda: {'calls': 0, 'duration_us': 0.0, 'durations': []})
    waits = collections.defaultdict(list)
    route_hist = collections.Counter()
    layers_by_step: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    layer_routes = collections.defaultdict(collections.Counter)
    route_batches = []

    admissions: dict[tuple[int, int], list[dict[str, Any]]] = collections.defaultdict(list)
    ready_events: dict[tuple[int, int], list[dict[str, Any]]] = collections.defaultdict(list)
    evictions: list[dict[str, Any]] = []
    hot_uses = collections.Counter()

    for event in events:
        kind = event.get('event')
        if kind == 'split_transfer':
            direction = f"{event.get('source_type', '?')}->{event.get('destination_type', '?')}"
            item = transfers[direction]
            item['groups'] += 1
            item['bytes'] += int(event.get('bytes', 0))
            item['issue_us'] += float(event.get('issue_us', 0.0))
            item['source_sync_us'] += float(event.get('source_sync_us', 0.0))
            item['destination_sync_us'] += float(event.get('destination_sync_us', 0.0))
            item['fallback'] += bool(event.get('fallback', False))
        elif kind == 'split_submit_return':
            key = event.get('backend_type', '?')
            if event.get('dynamic_hot'):
                key += ':dynamic-hot'
            elif event.get('dynamic_cold'):
                key += ':dynamic-cold'
            else:
                key += ':other'
            duration = float(event.get('duration_us', 0.0))
            submits[key]['calls'] += 1
            submits[key]['duration_us'] += duration
            submits[key]['durations'].append(duration)
        elif kind in {'hot_decision_sync', 'target_sync', 'split_compute_profile'}:
            waits[kind].append(float(event.get('duration_us', 0.0)))
        elif kind == 'fallback_transfer_completed':
            waits['fallback_transfer_sync'].append(float(event.get('sync_us', 0.0)))
        elif kind == 'route':
            ready = int(event.get('ready_available', 0))
            gpu = int(event.get('gpu_routes', 0))
            route_hist[(ready, gpu)] += 1
            step = int(event.get('step', -1))
            layers_by_step[step].append(event)
            layer_routes[int(event.get('layer', -1))][gpu] += 1
            selected = event.get('selected', [])
            slots = event.get('hot_slots', [])
            for expert, slot in zip(selected, slots):
                if int(slot) >= 0:
                    hot_uses[(int(event['layer']), int(expert))] += 1
        elif kind == 'route_batch':
            route_batches.append(event)
        elif kind == 'admit':
            admissions[(int(event['layer']), int(event['expert']))].append(event)
        elif kind == 'ready_after_completion':
            ready_events[(int(event['layer']), int(event['expert']))].append(event)
        elif kind == 'evict':
            evictions.append(event)

    complete_steps = []
    expected_layers = max((len(layer_events) for layer_events in layers_by_step.values()), default=0)
    for step, layer_events in sorted(layers_by_step.items()):
        if len(layer_events) != expected_layers:
            continue
        full = sum(int(event.get('gpu_routes', 0)) == len(event.get('selected', [])) for event in layer_events)
        cpu_dependent = len(layer_events) - full
        gpu_routes = sum(int(event.get('gpu_routes', 0)) for event in layer_events)
        total_routes = sum(len(event.get('selected', [])) for event in layer_events)
        complete_steps.append({
            'step': step,
            'layers': len(layer_events),
            'full_gpu_layers': full,
            'cpu_dependent_layers': cpu_dependent,
            'gpu_routes': gpu_routes,
            'total_routes': total_routes,
        })

    admitted_instances = sum(len(items) for items in admissions.values())
    admitted_unique = len(admissions)
    ready_instances = sum(len(items) for items in ready_events.values())
    unused_admitted = []
    reuse_counts = []
    for key, admission_list in admissions.items():
        uses = hot_uses[key]
        reuse_counts.extend([uses] * len(admission_list))
        if uses == 0:
            unused_admitted.extend([{'layer': key[0], 'expert': key[1]}] * len(admission_list))

    result = {
        'trace': str(args.trace),
        'parse_errors': parse_errors,
        'event_count': len(events),
        'event_counts': dict(counts),
        'time_span_s': (events[-1].get('us', 0) - events[0].get('us', 0)) / 1e6 if events else 0.0,
        'transfers': {
            key: {
                **value,
                'mib': value['bytes'] / 1024 / 1024,
                'issue_ms': value['issue_us'] / 1000,
                'source_sync_ms': value['source_sync_us'] / 1000,
                'destination_sync_ms': value['destination_sync_us'] / 1000,
            }
            for key, value in sorted(transfers.items())
        },
        'submit_returns': {
            key: {
                'calls': value['calls'],
                'total_ms': value['duration_us'] / 1000,
                'mean_us': value['duration_us'] / value['calls'] if value['calls'] else 0.0,
                'p50_us': percentile(value['durations'], 0.50),
                'p95_us': percentile(value['durations'], 0.95),
                'p99_us': percentile(value['durations'], 0.99),
                'max_us': max(value['durations'], default=0.0),
            }
            for key, value in sorted(submits.items())
        },
        'explicit_waits': {
            key: {
                'calls': len(values),
                'total_ms': sum(values) / 1000,
                'mean_us': sum(values) / len(values) if values else 0.0,
                'p95_us': percentile(values, 0.95),
                'max_us': max(values, default=0.0),
            }
            for key, values in sorted(waits.items())
        },
        'decode': {
            'expected_dynamic_layers': expected_layers,
            'complete_steps': len(complete_steps),
            'full_gpu_path_steps': sum(step['cpu_dependent_layers'] == 0 for step in complete_steps),
            'full_gpu_path_pct': pct(sum(step['cpu_dependent_layers'] == 0 for step in complete_steps), len(complete_steps)),
            'full_gpu_layers': sum(step['full_gpu_layers'] for step in complete_steps),
            'total_layers': sum(step['layers'] for step in complete_steps),
            'full_gpu_layer_pct': pct(sum(step['full_gpu_layers'] for step in complete_steps),
                                      sum(step['layers'] for step in complete_steps)),
            'mean_cpu_dependent_layers': (
                sum(step['cpu_dependent_layers'] for step in complete_steps) / len(complete_steps)
                if complete_steps else 0.0
            ),
            'route_hist_ready_to_executed': {
                f'{ready}->{gpu}': count for (ready, gpu), count in sorted(route_hist.items())
            },
            'first_10_steps': complete_steps[:10],
            'last_10_steps': complete_steps[-10:],
        },
        'prefill': {
            'batches': len(route_batches),
            'routes': sum(int(event.get('routes', 0)) for event in route_batches),
            'ready_routes': sum(int(event.get('ready_available', 0)) for event in route_batches),
            'gpu_routes': sum(int(event.get('gpu_routes', 0)) for event in route_batches),
        },
        'cache_lifecycle': {
            'admission_instances': admitted_instances,
            'admitted_unique_layer_experts': admitted_unique,
            'ready_instances': ready_instances,
            'evictions': len(evictions),
            'admitted_with_zero_later_hot_uses': len(unused_admitted),
            'admitted_zero_use_pct': pct(len(unused_admitted), admitted_instances),
            'mean_later_hot_uses_per_admission': sum(reuse_counts) / len(reuse_counts) if reuse_counts else 0.0,
            'p50_later_hot_uses': percentile([float(v) for v in reuse_counts], 0.50),
            'p95_later_hot_uses': percentile([float(v) for v in reuse_counts], 0.95),
            'max_later_hot_uses': max(reuse_counts, default=0),
        },
    }

    encoded = json.dumps(result, indent=2)
    print(encoded)
    if args.output:
        args.output.write_text(encoded + '\n')


if __name__ == '__main__':
    main()
