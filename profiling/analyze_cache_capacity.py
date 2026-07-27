#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any


def parse_trace(path: Path) -> tuple[dict[int, list[tuple[int, ...]]], dict[int, int]]:
    routes: dict[int, list[tuple[int, ...]]] = collections.defaultdict(list)
    capacities: collections.Counter[int] = collections.Counter()
    with path.open(errors="replace") as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") == "route" and event.get("phase") == "decode":
                routes[int(event["layer"])].append(tuple(map(int, event["selected"])))
            elif event.get("event") == "admit":
                layer = int(event["layer"])
                capacities[layer] = max(capacities[layer], int(event["slot"]) + 1)
    return dict(routes), dict(capacities)


def build_curves(routes: dict[int, list[tuple[int, ...]]]) -> dict[int, list[tuple[int, int]]]:
    curves: dict[int, list[tuple[int, int]]] = {}
    for layer, layer_routes in routes.items():
        frequency = collections.Counter(expert for selected in layer_routes for expert in selected)
        ranked = [expert for expert, _ in frequency.most_common()]
        resident: set[int] = set()
        curve: list[tuple[int, int]] = []
        for capacity in range(257):
            if 0 < capacity <= len(ranked):
                resident.add(ranked[capacity - 1])
            hits = sum(sum(expert in resident for expert in selected) for selected in layer_routes)
            full = sum(all(expert in resident for expert in selected) for selected in layer_routes)
            curve.append((hits, full))
        curves[layer] = curve
    return curves


def allocate_greedy(
    curves: dict[int, list[tuple[int, int]]], budget: int, objective: str
) -> dict[int, int]:
    allocation = {layer: 0 for layer in curves}
    for _ in range(budget):
        best: tuple[tuple[int, int, int, int], int] | None = None
        for layer in sorted(curves):
            capacity = allocation[layer]
            if capacity >= 256:
                continue
            hits0, full0 = curves[layer][capacity]
            hits1, full1 = curves[layer][capacity + 1]
            if objective == "routes":
                candidate = (hits1 - hits0, full1 - full0, -capacity, -layer)
            else:
                candidate = (full1 - full0, hits1 - hits0, -capacity, -layer)
            if best is None or candidate > best[0]:
                best = (candidate, layer)
        if best is None:
            break
        allocation[best[1]] += 1
    return allocation


def metrics(
    routes: dict[int, list[tuple[int, ...]]],
    curves: dict[int, list[tuple[int, int]]],
    allocation: dict[int, int],
) -> dict[str, float | int]:
    hits = full = total_routes = total_layers = 0
    for layer, layer_routes in routes.items():
        layer_hits, layer_full = curves[layer][allocation.get(layer, 0)]
        hits += layer_hits
        full += layer_full
        total_routes += sum(len(selected) for selected in layer_routes)
        total_layers += len(layer_routes)
    return {
        "slots": sum(allocation.values()),
        "route_hit_pct": 100.0 * hits / total_routes if total_routes else 0.0,
        "full_layer_pct": 100.0 * full / total_layers if total_layers else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--budgets",
        default="256,512,768,1024,1280,1536,1775,2048,2560,3072,4096,6144,8192",
    )
    args = parser.parse_args()

    routes, current_capacity = parse_trace(args.trace)
    curves = build_curves(routes)
    layers = sorted(routes)
    result: dict[str, Any] = {
        "trace": str(args.trace),
        "layers": layers,
        "current_capacity": current_capacity,
        "current_static_frequency": metrics(routes, curves, current_capacity),
        "budgets": [],
    }

    for budget in map(int, args.budgets.split(",")):
        equal = {
            layer: budget // len(layers) + int(index < budget % len(layers))
            for index, layer in enumerate(layers)
        }
        route_optimal = allocate_greedy(curves, budget, "routes")
        full_optimal = allocate_greedy(curves, budget, "full")
        result["budgets"].append(
            {
                "budget": budget,
                "equal": metrics(routes, curves, equal),
                "route_optimal": metrics(routes, curves, route_optimal),
                "full_layer_optimal": metrics(routes, curves, full_optimal),
                "full_layer_optimal_allocation": full_optimal,
            }
        )

    encoded = json.dumps(result, indent=2)
    print(encoded)
    if args.output:
        args.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
