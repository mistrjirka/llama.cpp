#!/usr/bin/env python3
"""Summarize dynamic MoE cache behavior in token windows."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    routes_by_step: dict[int, list[dict[str, object]]] = defaultdict(list)
    events_by_step: dict[int, Counter[str]] = defaultdict(Counter)
    first_route_us: dict[int, int] = {}
    last_route_us: dict[int, int] = {}
    event_counts: Counter[str] = Counter()
    admissions_by_policy: Counter[str] = Counter()
    parse_errors = 0

    with args.trace.open(errors="replace") as stream:
        for line in stream:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                parse_errors += 1
                continue
            kind = str(event.get("event", "?"))
            event_counts[kind] += 1
            step_value = event.get("step")
            if kind == "route" and event.get("phase") == "decode":
                step = int(step_value)
                routes_by_step[step].append(event)
                timestamp = int(event.get("us", 0))
                first_route_us[step] = min(first_route_us.get(step, timestamp), timestamp)
                last_route_us[step] = max(last_route_us.get(step, timestamp), timestamp)
            elif step_value is not None:
                step = int(step_value)
                events_by_step[step][kind] += 1
                if kind == "admit":
                    admissions_by_policy[str(event.get("policy", "demand"))] += 1

    complete_steps = sorted(routes_by_step)
    expected_layers = max((len(routes_by_step[step]) for step in complete_steps), default=0)
    complete_steps = [step for step in complete_steps if len(routes_by_step[step]) == expected_layers]
    if not complete_steps:
        raise RuntimeError("no complete decode steps")

    step_rows = []
    for step in complete_steps:
        routes = routes_by_step[step]
        total = sum(len(event.get("selected", [])) for event in routes)
        ready = sum(int(event.get("ready_available", 0)) for event in routes)
        executed = sum(int(event.get("gpu_routes", 0)) for event in routes)
        complete_ready = sum(
            int(event.get("ready_available", 0)) == len(event.get("selected", []))
            for event in routes
        )
        complete_gpu = sum(
            int(event.get("gpu_routes", 0)) == len(event.get("selected", []))
            for event in routes
        )
        row = {
            "step": step,
            "layers": len(routes),
            "routes": total,
            "ready_routes": ready,
            "gpu_routes": executed,
            "complete_ready_layers": complete_ready,
            "complete_gpu_layers": complete_gpu,
            "cpu_dependent_layers": len(routes) - complete_gpu,
            "first_route_us": first_route_us[step],
            "last_route_us": last_route_us[step],
            "admissions": events_by_step[step]["admit"],
            "evictions": events_by_step[step]["evict"],
            "ready_events": events_by_step[step]["ready_after_completion"],
            "predict_events": events_by_step[step]["predict_prefetch"],
        }
        step_rows.append(row)

    # First-route deltas approximate end-to-end token intervals without requiring
    # additional timing instrumentation in the sampler.
    for index in range(len(step_rows) - 1):
        delta = step_rows[index + 1]["first_route_us"] - step_rows[index]["first_route_us"]
        step_rows[index]["next_token_us"] = max(0, delta)
    step_rows[-1]["next_token_us"] = None

    windows = []
    for start in range(0, len(step_rows), args.window):
        rows = step_rows[start : start + args.window]
        intervals = [row["next_token_us"] for row in rows if row["next_token_us"] is not None]
        routes = sum(row["routes"] for row in rows)
        ready = sum(row["ready_routes"] for row in rows)
        gpu = sum(row["gpu_routes"] for row in rows)
        layers = sum(row["layers"] for row in rows)
        complete_ready = sum(row["complete_ready_layers"] for row in rows)
        complete_gpu = sum(row["complete_gpu_layers"] for row in rows)
        duration_us = sum(intervals)
        windows.append(
            {
                "step_start": rows[0]["step"],
                "step_end": rows[-1]["step"],
                "tokens": len(rows),
                "estimated_tps": len(intervals) * 1e6 / duration_us if duration_us else None,
                "mean_token_ms": statistics.mean(intervals) / 1000 if intervals else None,
                "p50_token_ms": statistics.median(intervals) / 1000 if intervals else None,
                "route_ready_rate": ready / routes,
                "route_gpu_rate": gpu / routes,
                "complete_ready_layer_rate": complete_ready / layers,
                "complete_gpu_layer_rate": complete_gpu / layers,
                "mean_cpu_dependent_layers": sum(row["cpu_dependent_layers"] for row in rows) / len(rows),
                "admissions": sum(row["admissions"] for row in rows),
                "evictions": sum(row["evictions"] for row in rows),
                "ready_events": sum(row["ready_events"] for row in rows),
                "predict_events": sum(row["predict_events"] for row in rows),
            }
        )

    all_routes = sum(row["routes"] for row in step_rows)
    all_ready = sum(row["ready_routes"] for row in step_rows)
    all_gpu = sum(row["gpu_routes"] for row in step_rows)
    all_layers = sum(row["layers"] for row in step_rows)
    result = {
        "trace": str(args.trace),
        "parse_errors": parse_errors,
        "event_counts": dict(event_counts),
        "admissions_by_policy": dict(admissions_by_policy),
        "expected_layers": expected_layers,
        "complete_steps": len(step_rows),
        "overall": {
            "route_ready_rate": all_ready / all_routes,
            "route_gpu_rate": all_gpu / all_routes,
            "complete_ready_layer_rate": sum(row["complete_ready_layers"] for row in step_rows) / all_layers,
            "complete_gpu_layer_rate": sum(row["complete_gpu_layers"] for row in step_rows) / all_layers,
            "mean_cpu_dependent_layers": sum(row["cpu_dependent_layers"] for row in step_rows) / len(step_rows),
            "admissions": sum(row["admissions"] for row in step_rows),
            "evictions": sum(row["evictions"] for row in step_rows),
        },
        "windows": windows,
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    print(
        f"steps={len(step_rows)} layers={expected_layers} "
        f"ready={result['overall']['route_ready_rate']*100:.2f}% "
        f"complete={result['overall']['complete_ready_layer_rate']*100:.2f}%"
    )
    print(
        "window       tps    ready  complete  gpu-routes  cpu-layers  admits evicts"
    )
    for window in windows:
        print(
            f"{window['step_start']:4d}-{window['step_end']:4d} "
            f"{(window['estimated_tps'] or 0):6.2f} "
            f"{window['route_ready_rate']*100:7.2f}% "
            f"{window['complete_ready_layer_rate']*100:8.2f}% "
            f"{window['route_gpu_rate']*100:9.2f}% "
            f"{window['mean_cpu_dependent_layers']:10.2f} "
            f"{window['admissions']:6d} {window['evictions']:6d}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
