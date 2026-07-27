#!/usr/bin/env python3
"""Audit the live cache-aware cross-layer predictor.

Unlike route-only top-1 evaluation, this script reconstructs the runtime's
resident/copying set and evaluates the expert that was actually uploaded: the
highest-scoring *absent* expert.  It also independently replays the conditional
count model and verifies every live admission and score.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics
from typing import Any

N_EXPERT = 256
NORMAL_BYTES = 4_079_616
LARGE_BYTES = 5_505_024


def pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    lo = math.floor(position)
    hi = math.ceil(position)
    if lo == hi:
        return ordered[lo]
    f = position - lo
    return ordered[lo] * (1 - f) + ordered[hi] * f


def describe(values: list[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "p50": pct(values, 0.5),
        "p90": pct(values, 0.9),
        "p95": pct(values, 0.95),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def load(path: Path) -> tuple[list[dict[str, Any]], dict[int, dict[int, list[int]]]]:
    events: list[dict[str, Any]] = []
    routes: dict[int, dict[int, list[int]]] = defaultdict(dict)
    with path.open(errors="replace") as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            events.append(event)
            if event.get("event") == "route" and event.get("phase") == "decode":
                routes[int(event["step"])][int(event["layer"])] = list(
                    dict.fromkeys(map(int, event["selected"]))
                )
    return events, routes


def build_snapshots(
    routes: dict[int, dict[int, list[int]]],
    needed: set[tuple[int, int, int, int]],
    min_observations: int,
) -> dict[tuple[int, int, int, int], tuple[list[float], int]]:
    models: dict[tuple[int, int], tuple[list[int], list[Counter[int]]]] = {}

    def model(target: int, distance: int) -> tuple[list[int], list[Counter[int]]]:
        key = (target, distance)
        if key not in models:
            models[key] = ([0] * N_EXPERT, [Counter() for _ in range(N_EXPERT)])
        return models[key]

    snapshots: dict[tuple[int, int, int, int], tuple[list[float], int]] = {}
    for step in sorted(routes):
        token = routes[step]
        # Runtime predicts from this token before learning its target route.
        for source_layer, selected in token.items():
            for distance in (1, 2):
                target = source_layer + distance
                key = (step, source_layer, target, distance)
                if key not in needed:
                    continue
                source_counts, transitions = model(target, distance)
                scores = [0.0] * N_EXPERT
                eligible = 0
                for source in selected:
                    observations = source_counts[source]
                    if observations < min_observations:
                        continue
                    eligible += 1
                    for expert, count in transitions[source].items():
                        scores[expert] += count / observations
                snapshots[key] = (scores, eligible)

        # Prequential update after all predictions for this step.
        for target_layer, target_selected in token.items():
            for distance in (1, 2):
                source_layer = target_layer - distance
                if source_layer not in token:
                    continue
                source_counts, transitions = model(target_layer, distance)
                for source in token[source_layer]:
                    source_counts[source] += 1
                    transitions[source].update(target_selected)
    return snapshots


def evaluate(path: Path, min_observations: int) -> dict[str, Any]:
    events, routes = load(path)
    cross_admissions = [
        event for event in events
        if event.get("event") == "admit"
        and event.get("policy") == "cross_layer_transition_prefetch"
    ]
    needed = {
        (int(event["step"]), int(event["source_layer"]), int(event["layer"]), int(event["distance"]))
        for event in cross_admissions
    }
    snapshots = build_snapshots(routes, needed, min_observations)

    resident: dict[int, set[int]] = defaultdict(set)
    records: list[dict[str, Any]] = []
    exact_expert_matches = 0
    exact_eligible_matches = 0
    score_errors: list[float] = []

    for event in events:
        kind = event.get("event")
        if kind in ("evict", "spill_evict"):
            resident[int(event["layer"])].discard(int(event["expert"]))
            continue
        if kind != "admit":
            continue

        layer = int(event["layer"])
        expert = int(event["expert"])
        victim = int(event.get("victim_expert", -1))
        if event.get("policy") == "cross_layer_transition_prefetch":
            key = (
                int(event["step"]), int(event["source_layer"]), layer, int(event["distance"])
            )
            scores, eligible = snapshots[key]
            ranked = sorted(
                (index for index, score in enumerate(scores) if score > 0),
                key=lambda index: (-scores[index], index),
            )
            absent = [index for index in ranked if index not in resident[layer]]
            expected = absent[0] if absent else None
            expected_score = scores[expected] if expected is not None else None
            all_top = ranked[0] if ranked else None
            all_rank = ranked.index(expert) + 1 if expert in ranked else None
            truth = set(routes.get(int(event["step"]), {}).get(layer, []))
            bundle_bytes = LARGE_BYTES if layer == 46 else NORMAL_BYTES

            expert_match = expected == expert
            eligible_match = eligible == int(event["eligible_sources"])
            exact_expert_matches += expert_match
            exact_eligible_matches += eligible_match
            if expected_score is not None:
                score_errors.append(abs(float(event["prediction_score"]) - expected_score))

            records.append({
                "step": int(event["step"]),
                "source_layer": int(event["source_layer"]),
                "target_layer": layer,
                "distance": int(event["distance"]),
                "uploaded_expert": expert,
                "uploaded_score": float(event["prediction_score"]),
                "uploaded_correct": expert in truth,
                "bundle_bytes": bundle_bytes,
                "resident_count": len(resident[layer]),
                "eligible_sources": eligible,
                "unconstrained_top": all_top,
                "unconstrained_top_score": scores[all_top] if all_top is not None else None,
                "unconstrained_top_correct": all_top in truth if all_top is not None else False,
                "unconstrained_top_resident": all_top in resident[layer] if all_top is not None else False,
                "uploaded_rank_among_all": all_rank,
                "replay_expert_match": expert_match,
                "replay_eligible_match": eligible_match,
            })

        if victim >= 0:
            resident[layer].discard(victim)
        resident[layer].add(expert)

    def group(rows: list[dict[str, Any]]) -> dict[str, Any]:
        correct = sum(bool(row["uploaded_correct"]) for row in rows)
        total_bytes = sum(int(row["bundle_bytes"]) for row in rows)
        wasted_bytes = sum(
            int(row["bundle_bytes"]) for row in rows if not bool(row["uploaded_correct"])
        )
        return {
            "predictions": len(rows),
            "correct": correct,
            "precision": correct / len(rows) if rows else None,
            "bytes_gib": total_bytes / 2**30,
            "wasted_bytes_gib": wasted_bytes / 2**30,
            "mib_per_correct": total_bytes / 2**20 / correct if correct else None,
        }

    thresholds = []
    for threshold in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
        selected = [row for row in records if float(row["uploaded_score"]) >= threshold]
        row = group(selected)
        row["threshold"] = threshold
        row["coverage"] = len(selected) / len(records) if records else None
        thresholds.append(row)

    by_distance = {
        str(distance): group([row for row in records if int(row["distance"]) == distance])
        for distance in (1, 2)
    }
    ranks = [int(row["uploaded_rank_among_all"]) for row in records if row["uploaded_rank_among_all"]]
    top_resident = sum(bool(row["unconstrained_top_resident"]) for row in records)
    top_correct = sum(bool(row["unconstrained_top_correct"]) for row in records)

    return {
        "trace": str(path),
        "min_observations": min_observations,
        "runtime_replay_audit": {
            "admissions": len(records),
            "expert_matches": exact_expert_matches,
            "eligible_source_matches": exact_eligible_matches,
            "score_error": describe(score_errors),
            "exact": (
                exact_expert_matches == len(records)
                and exact_eligible_matches == len(records)
                and all(error <= 1e-6 for error in score_errors)
            ),
        },
        "actual_uploaded_candidate": group(records),
        "by_distance": by_distance,
        "score_thresholds": thresholds,
        "resident_exclusion": {
            "unconstrained_top_precision": top_correct / len(records) if records else None,
            "unconstrained_top_resident_fraction": top_resident / len(records) if records else None,
            "uploaded_rank_among_all": describe([float(rank) for rank in ranks]),
            "resident_count_at_upload": describe([float(row["resident_count"]) for row in records]),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-observations", type=int, default=8)
    args = parser.parse_args()

    result = evaluate(args.trace, args.min_observations)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["runtime_replay_audit"]["exact"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
