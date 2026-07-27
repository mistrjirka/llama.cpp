#!/usr/bin/env python3
"""Analyze detailed dynamic-MoE promotion timelines.

The input is the JSONL emitted by GGML_MOE_DYNAMIC_TRACE after the transfer
instrumentation in ggml-backend.cpp.  The analyzer keeps component, bundle,
queue/batch, and route-deadline measurements separate so shared chunk
synchronization is not accidentally reported as a per-copy hardware time.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


PERCENTILES = (0.50, 0.90, 0.95, 0.99)


def percentile(values: Iterable[float], q: float) -> float | None:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def describe(values: Iterable[float]) -> dict[str, float | int | None]:
    data = [float(value) for value in values if math.isfinite(float(value))]
    result: dict[str, float | int | None] = {
        "count": len(data),
        "mean": statistics.fmean(data) if data else None,
        "min": min(data) if data else None,
        "max": max(data) if data else None,
    }
    for q in PERCENTILES:
        result[f"p{int(q * 100)}"] = percentile(data, q)
    return result


def counter_dict(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items(), key=lambda item: str(item[0]))}


@dataclass
class Bundle:
    bundle_id: int
    layer: int = -1
    slot: int = -1
    expert: int = -1
    distance: int = 0
    step: int = 0
    bytes: int = 0
    queued_ns: int = 0
    trace_us: int = 0
    jobs: list[dict[str, Any]] = field(default_factory=list)
    ready_ns: int = 0
    ready_us: float | None = None

    def row(self, route: dict[str, Any] | None) -> dict[str, Any]:
        starts = [int(job["job_start_mono_ns"]) for job in self.jobs]
        h2d_starts = [int(job["h2d_issue_begin_mono_ns"]) for job in self.jobs]
        h2d_ends = [int(job["h2d_issue_end_mono_ns"]) for job in self.jobs]
        sync_ends = [int(job["sync_end_mono_ns"]) for job in self.jobs]
        staging_us = [float(job["staging_copy_us"]) for job in self.jobs]
        api_us = [float(job["h2d_api_us"]) for job in self.jobs]
        batch_ids = {int(job["batch_id"]) for job in self.jobs}
        sync_boundaries = {
            (int(job["sync_begin_mono_ns"]), int(job["sync_end_mono_ns"])) for job in self.jobs
        }
        first_start = min(starts) if starts else 0
        first_h2d = min(h2d_starts) if h2d_starts else 0
        last_h2d_return = max(h2d_ends) if h2d_ends else 0
        complete = self.ready_ns or (max(sync_ends) if sync_ends else 0)
        h2d_window_us = (complete - first_h2d) / 1000.0 if complete and first_h2d else None
        gib_per_s = None
        if h2d_window_us is not None and h2d_window_us > 0:
            gib_per_s = self.bytes / (h2d_window_us * 1e-6) / (1024.0 ** 3)

        route_ns = int(route.get("route_decision_mono_ns", 0)) if route else 0
        selected = bool(route and self.expert in route.get("selected", []))
        ready_before_route = None if not route_ns or not complete else complete <= route_ns
        slack_us = None if not route_ns or not complete else (route_ns - complete) / 1000.0

        return {
            "bundle_id": self.bundle_id,
            "layer": self.layer,
            "slot": self.slot,
            "expert": self.expert,
            "distance": self.distance,
            "step": self.step,
            "bytes": self.bytes,
            "components": len(self.jobs),
            "batch_count": len(batch_ids),
            "sync_count": len(sync_boundaries),
            "queued_mono_ns": self.queued_ns,
            "first_job_start_mono_ns": first_start,
            "first_h2d_issue_mono_ns": first_h2d,
            "last_h2d_api_return_mono_ns": last_h2d_return,
            "ready_mono_ns": complete,
            "queue_to_first_job_us": (first_start - self.queued_ns) / 1000.0 if first_start else None,
            "queue_to_first_h2d_us": (first_h2d - self.queued_ns) / 1000.0 if first_h2d else None,
            "queue_to_ready_us": (complete - self.queued_ns) / 1000.0 if complete else None,
            "h2d_window_us": h2d_window_us,
            "effective_gib_s": gib_per_s,
            "staging_total_us": sum(staging_us),
            "h2d_api_total_us": sum(api_us),
            "route_found": route is not None,
            "selected": selected,
            "ready_before_route": ready_before_route,
            "deadline_slack_us": slack_us,
            "route_ready_available": route.get("ready_available") if route else None,
            "route_gpu_routes": route.get("gpu_routes") if route else None,
        }


def load(path: Path) -> tuple[
    Counter[str], dict[int, Bundle], list[dict[str, Any]], dict[tuple[int, int], dict[str, Any]],
    list[dict[str, Any]], list[int], int
]:
    counts: Counter[str] = Counter()
    bundles: dict[int, Bundle] = {}
    jobs: list[dict[str, Any]] = []
    routes: dict[tuple[int, int], dict[str, Any]] = {}
    batches: list[dict[str, Any]] = []
    trace_start_samples: list[int] = []
    malformed = 0

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            name = str(event.get("event", ""))
            counts[name] += 1

            if name == "urgent_bundle_queued":
                bundle_id = int(event.get("bundle_id", 0))
                if bundle_id <= 0:
                    continue
                bundle = bundles.setdefault(bundle_id, Bundle(bundle_id))
                bundle.layer = int(event["layer"])
                bundle.slot = int(event["slot"])
                bundle.expert = int(event["expert"])
                bundle.distance = int(event["distance"])
                bundle.step = int(event["step"])
                bundle.bytes = int(event["bytes"])
                bundle.queued_ns = int(event["queued_mono_ns"])
                bundle.trace_us = int(event["us"])
                trace_start_samples.append(bundle.queued_ns - bundle.trace_us * 1000)
            elif name == "worker_transfer_timing":
                jobs.append(event)
                bundle_id = int(event.get("bundle_id", 0))
                if bundle_id > 0:
                    bundles.setdefault(bundle_id, Bundle(bundle_id)).jobs.append(event)
                sync_end = int(event.get("sync_end_mono_ns", 0))
                if sync_end:
                    trace_start_samples.append(sync_end - int(event["us"]) * 1000)
            elif name == "ready_after_completion":
                bundle_id = int(event.get("bundle_id", 0))
                if bundle_id > 0:
                    bundle = bundles.setdefault(bundle_id, Bundle(bundle_id))
                    bundle.ready_ns = int(event.get("ready_mono_ns", 0))
                    value = event.get("prediction_to_ready_us")
                    bundle.ready_us = float(value) if value is not None else None
            elif name == "route" and event.get("phase") == "decode":
                routes[(int(event["layer"]), int(event["step"]))] = event
            elif name == "worker_batch_completed":
                batches.append(event)

    # Older timeline files lack an explicit route monotonic timestamp.  Estimate
    # the trace start from events containing both clocks.  The error is only the
    # short delay between the captured timestamp and JSON emission, but deadline
    # statistics are marked approximate by the caller.
    trace_start_estimate = int(statistics.median(trace_start_samples)) if trace_start_samples else 0
    if trace_start_estimate:
        for route in routes.values():
            if not route.get("route_decision_mono_ns"):
                route["route_decision_mono_ns"] = trace_start_estimate + int(route["us"]) * 1000
                route["route_time_estimated"] = True

    return counts, bundles, jobs, routes, batches, trace_start_samples, malformed


def metric_group(jobs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "jobs": len(jobs),
        "bytes": sum(int(job["bytes"]) for job in jobs),
        "queue_delay_us": describe(float(job["queue_delay_us"]) for job in jobs),
        "captured_wait_us": describe(float(job["captured_wait_us"]) for job in jobs),
        "staging_copy_us": describe(float(job["staging_copy_us"]) for job in jobs),
        "h2d_api_us": describe(float(job["h2d_api_us"]) for job in jobs),
        "issue_to_sync_end_us": describe(float(job["issue_to_sync_end_us"]) for job in jobs),
        "queued_to_sync_end_us": describe(float(job["prediction_to_sync_end_us"]) for job in jobs),
    }


def count_inversions(rows: list[dict[str, Any]]) -> int:
    ordered = sorted(rows, key=lambda row: (int(row["queued_mono_ns"]), int(row["bundle_id"])))
    starts = sorted({int(row["first_job_start_mono_ns"]) for row in ordered})
    fenwick = [0] * (len(starts) + 1)

    def update(index: int) -> None:
        index += 1
        while index < len(fenwick):
            fenwick[index] += 1
            index += index & -index

    def query(index: int) -> int:
        total = 0
        index += 1
        while index > 0:
            total += fenwick[index]
            index -= index & -index
        return total

    inversions = 0
    seen = 0
    for row in ordered:
        rank = bisect.bisect_left(starts, int(row["first_job_start_mono_ns"]))
        inversions += seen - query(rank)
        update(rank)
        seen += 1
    return inversions


def analyze(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    counts, bundles, jobs, routes, batches, start_samples, malformed = load(path)
    rows = [bundle.row(routes.get((bundle.layer, bundle.step))) for bundle in bundles.values()]
    rows.sort(key=lambda row: int(row["bundle_id"]))

    urgent_jobs = [job for job in jobs if bool(job.get("urgent"))]
    background_jobs = [job for job in jobs if not bool(job.get("urgent"))]
    by_component: dict[str, Any] = {}
    for component in sorted({int(job["component"]) for job in jobs}):
        by_component[str(component)] = metric_group(
            [job for job in jobs if int(job["component"]) == component]
        )
    by_distance: dict[str, Any] = {}
    for distance in sorted({int(job.get("prediction_distance", 0)) for job in urgent_jobs}):
        by_distance[str(distance)] = metric_group(
            [job for job in urgent_jobs if int(job.get("prediction_distance", 0)) == distance]
        )

    batch_size_counts = Counter(int(batch["jobs"]) for batch in batches)
    mixed_batches = [
        batch for batch in batches
        if 0 < int(batch.get("urgent_jobs", 0)) < int(batch.get("jobs", 0))
    ]
    background_intervals = sorted(
        (int(batch["capture_mono_ns"]), int(batch["complete_mono_ns"]))
        for batch in batches if int(batch.get("urgent_jobs", 0)) == 0
    )
    blocked_by_captured_background = 0
    for row in rows:
        queued = int(row["queued_mono_ns"])
        # Batch intervals are short and the run has only ~10k batches; this
        # direct check is intentionally transparent and still inexpensive.
        if any(begin < queued < end for begin, end in background_intervals):
            blocked_by_captured_background += 1

    complete_rows = [row for row in rows if int(row["components"]) == 3 and row["ready_mono_ns"]]
    route_rows = [row for row in complete_rows if bool(row["route_found"])]
    selected_rows = [row for row in route_rows if bool(row["selected"])]
    ready_before_rows = [row for row in selected_rows if row["ready_before_route"] is True]
    late_rows = [row for row in selected_rows if row["ready_before_route"] is False]
    unselected_rows = [row for row in route_rows if not bool(row["selected"])]

    component_accounting_ok = (
        counts.get("component_completed", 0) == len(jobs)
        and counts.get("ready_after_completion", 0) * 3 == len(jobs)
    )
    urgent_accounting_ok = len(urgent_jobs) == counts.get("urgent_bundle_queued", 0) * 3

    def bundle_metrics(group: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "bundles": len(group),
            "queue_to_first_job_us": describe(row["queue_to_first_job_us"] for row in group),
            "queue_to_first_h2d_us": describe(row["queue_to_first_h2d_us"] for row in group),
            "queue_to_ready_us": describe(row["queue_to_ready_us"] for row in group),
            "h2d_window_us": describe(row["h2d_window_us"] for row in group),
            "effective_gib_s": describe(row["effective_gib_s"] for row in group),
            "staging_total_us": describe(row["staging_total_us"] for row in group),
            "h2d_api_total_us": describe(row["h2d_api_total_us"] for row in group),
            "deadline_slack_us": describe(
                row["deadline_slack_us"] for row in group if row["deadline_slack_us"] is not None
            ),
        }

    by_bundle_distance = {
        str(distance): bundle_metrics([row for row in complete_rows if int(row["distance"]) == distance])
        for distance in sorted({int(row["distance"]) for row in complete_rows})
    }

    trace_start_spread_us = None
    if start_samples:
        trace_start_spread_us = (max(start_samples) - min(start_samples)) / 1000.0

    summary: dict[str, Any] = {
        "input": str(path),
        "malformed_lines": malformed,
        "event_counts": counter_dict(counts),
        "accounting": {
            "admissions": counts.get("admit", 0),
            "ready_bundles": counts.get("ready_after_completion", 0),
            "component_jobs": len(jobs),
            "urgent_bundles": counts.get("urgent_bundle_queued", 0),
            "urgent_jobs": len(urgent_jobs),
            "background_jobs": len(background_jobs),
            "component_accounting_ok": component_accounting_ok,
            "urgent_accounting_ok": urgent_accounting_ok,
            "complete_urgent_bundles": len(complete_rows),
        },
        "clock": {
            "route_deadline_estimated": any(bool(route.get("route_time_estimated")) for route in routes.values()),
            "trace_start_samples": len(start_samples),
            "trace_start_spread_us": trace_start_spread_us,
        },
        "components": {
            "all": metric_group(jobs),
            "urgent": metric_group(urgent_jobs),
            "background": metric_group(background_jobs),
            "by_component": by_component,
            "urgent_by_prediction_distance": by_distance,
            "size_counts": counter_dict(Counter(int(job["bytes"]) for job in jobs)),
        },
        "bundles": {
            "all_complete_urgent": bundle_metrics(complete_rows),
            "by_prediction_distance": by_bundle_distance,
            "three_components_one_batch": sum(int(row["batch_count"]) == 1 for row in complete_rows),
            "three_components_one_sync": sum(int(row["sync_count"]) == 1 for row in complete_rows),
            "fifo_inversions": count_inversions(complete_rows),
            "blocked_by_already_captured_background_batch": blocked_by_captured_background,
            "blocked_fraction": blocked_by_captured_background / len(complete_rows) if complete_rows else None,
        },
        "batches": {
            "count": len(batches),
            "size_counts": counter_dict(batch_size_counts),
            "mean_jobs": statistics.fmean(int(batch["jobs"]) for batch in batches) if batches else None,
            "urgent_only": sum(int(batch.get("urgent_jobs", 0)) == int(batch["jobs"]) for batch in batches),
            "background_only": sum(int(batch.get("urgent_jobs", 0)) == 0 for batch in batches),
            "mixed": len(mixed_batches),
            "mixed_jobs": sum(int(batch["jobs"]) for batch in mixed_batches),
            "duration_us": describe(float(batch["duration_us"]) for batch in batches),
        },
        "deadline": {
            "route_matched_bundles": len(route_rows),
            "selected_correct_predictions": len(selected_rows),
            "unselected_wasted_predictions": len(unselected_rows),
            "precision": len(selected_rows) / len(route_rows) if route_rows else None,
            "selected_ready_before_route": len(ready_before_rows),
            "selected_late": len(late_rows),
            "same_token_ready_rate_selected": len(ready_before_rows) / len(selected_rows) if selected_rows else None,
            "same_token_ready_rate_all_uploads": len(ready_before_rows) / len(route_rows) if route_rows else None,
            "selected_ready_before_route_by_distance": {
                str(distance): {
                    "selected": sum(int(row["distance"]) == distance for row in selected_rows),
                    "ready_before_route": sum(
                        int(row["distance"]) == distance and row["ready_before_route"] is True
                        for row in selected_rows
                    ),
                }
                for distance in sorted({int(row["distance"]) for row in selected_rows})
            },
            "late_by_distance": counter_dict(Counter(int(row["distance"]) for row in late_rows)),
        },
    }
    return summary, rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()

    summary, rows = analyze(args.trace)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.csv:
        write_csv(args.csv, rows)
    print(json.dumps(summary, indent=2))
    return 0 if summary["accounting"]["component_accounting_ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
