#!/usr/bin/env python3
"""Compare profiled MoE CPU-fallback cost with expert promotion latency.

The CPU trace must be produced with GGML_MOE_DYNAMIC_PROFILE_COMPUTE=1.
The stream trace must contain admit and ready_after_completion events.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if isinstance(value, dict):
                yield value


def summarize(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    percentiles = (
        statistics.quantiles(values, n=100, method="inclusive")
        if len(values) > 1
        else [values[0]] * 99
    )
    return {
        "count": len(values),
        "mean_us": round(statistics.mean(values), 3),
        "median_us": round(statistics.median(values), 3),
        "p95_us": round(percentiles[94], 3),
        "max_us": round(max(values), 3),
        "sum_ms": round(sum(values) / 1000.0, 3),
    }


def analyze_cpu(path: Path) -> dict[str, Any]:
    latest_gpu_routes: dict[int, int] = {}
    all_cold: list[float] = []
    by_gpu_routes: dict[int, list[float]] = collections.defaultdict(list)
    hot: list[float] = []

    for event in read_jsonl(path):
        event_name = event.get("event")
        if event_name == "route":
            latest_gpu_routes[int(event["layer"])] = int(event.get("gpu_routes", 0))
            continue
        if event_name != "split_compute_profile":
            continue

        duration_us = float(event["duration_us"])
        kind = event.get("kind")
        if kind == "hot":
            hot.append(duration_us)
        elif kind == "cold":
            layer = int(event["layer"])
            all_cold.append(duration_us)
            by_gpu_routes[latest_gpu_routes.get(layer, -1)].append(duration_us)

    return {
        "cold_all": summarize(all_cold),
        "hot_all": summarize(hot),
        "cold_by_route_coverage": {
            str(gpu_routes): {
                "gpu_routes": gpu_routes,
                "cpu_misses": 8 - gpu_routes if gpu_routes >= 0 else None,
                **summarize(values),
            }
            for gpu_routes, values in sorted(by_gpu_routes.items())
        },
    }


def analyze_stream(path: Path) -> dict[str, Any]:
    admitted_at: dict[tuple[int, int], int] = {}
    ready_at: dict[tuple[int, int], int] = {}
    component_counts: collections.Counter[str] = collections.Counter()

    for event in read_jsonl(path):
        event_name = str(event.get("event", ""))
        key = (int(event.get("layer", -1)), int(event.get("expert", -1)))
        if event_name == "admit":
            admitted_at[key] = int(event["us"])
        elif event_name == "ready_after_completion":
            ready_at[key] = int(event["us"])
        elif event_name in {"component_enqueued", "component_completed"}:
            component_counts[event_name] += 1

    latencies = [
        float(ready_at[key] - admitted_us)
        for key, admitted_us in admitted_at.items()
        if key in ready_at
    ]
    return {
        "expert_promotion_latency": summarize(latencies),
        "component_counts": dict(component_counts),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-trace", type=Path, required=True)
    parser.add_argument("--stream-trace", type=Path, required=True)
    parser.add_argument("--gpu-compute-trace", type=Path)
    args = parser.parse_args()

    result: dict[str, Any] = {
        "cpu_fallback": analyze_cpu(args.cpu_trace),
        "streaming": analyze_stream(args.stream_trace),
    }
    if args.gpu_compute_trace is not None:
        result["gpu_compute"] = analyze_cpu(args.gpu_compute_trace)

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
