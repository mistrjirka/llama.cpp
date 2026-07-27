#!/usr/bin/env python3
"""Aggregate adaptive spill benchmark summaries and rank configurations."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


def key(run: dict[str, object]) -> tuple[object, ...]:
    return (
        run["spill_slots"],
        run["predict_total"],
        run["predict_per_layer"],
        run["variant"],
        run.get("min_score", 0.0),
        run["reserve_mib"],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    runs = []
    for path in sorted(args.directory.glob("*.summary.json")):
        run = json.loads(path.read_text())
        run["path"] = str(path)
        if run.get("exit") == 0 and run.get("assertions_passed"):
            runs.append(run)
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for run in runs:
        groups[key(run)].append(run)

    configurations = []
    for config_key, group in groups.items():
        spill, total, per_layer, variant, min_score, reserve = config_key
        tps = [float(run["measured_tps"]) for run in group]
        complete = [int(run["coverage"][0]) for run in group]
        complete_total = [int(run["coverage"][1]) for run in group]
        resident_hit = [float(run["cache"][4]) for run in group]
        cpu_layers = [float(run["coverage"][5]) for run in group]
        urgent_jobs = [int(run["worker"][2]) for run in group]
        worker_batches = [int(run["worker"][1]) for run in group]
        configurations.append(
            {
                "spill_slots": spill,
                "predict_total": total,
                "predict_per_layer": per_layer,
                "variant": variant,
                "min_score": min_score,
                "reserve_mib": reserve,
                "n": len(group),
                "tps": tps,
                "mean_tps": statistics.mean(tps),
                "stdev_tps": statistics.stdev(tps) if len(tps) > 1 else 0.0,
                "complete_layers": complete,
                "mean_complete_layers": statistics.mean(complete),
                "complete_layer_rate": statistics.mean(
                    value / total_value for value, total_value in zip(complete, complete_total)
                ),
                "mean_resident_hit_rate": statistics.mean(resident_hit),
                "mean_cpu_dependent_layers": statistics.mean(cpu_layers),
                "mean_urgent_jobs": statistics.mean(urgent_jobs),
                "mean_worker_batches": statistics.mean(worker_batches),
                "mean_gpu_util": statistics.mean(float(run["gpu_util_mean"]) for run in group),
                "mean_min_free_mib": statistics.mean(float(run["min_free_mib"]) for run in group),
                "output_hashes": sorted({str(run["sha256"]) for run in group}),
                "runs": [str(run["path"]) for run in group],
            }
        )
    configurations.sort(key=lambda item: (-item["mean_tps"], -item["complete_layer_rate"]))
    result = {"runs": runs, "configurations": configurations}
    output = args.output or args.directory / "aggregate.json"
    output.write_text(json.dumps(result, indent=2) + "\n")

    print(
        "spill total per variant             score reserve n   tps mean  complete  resident  cpu-layers urgent"
    )
    for item in configurations:
        print(
            f"{item['spill_slots']:5d} {item['predict_total']:5d} {item['predict_per_layer']:3d} "
            f"{item['variant']:19s} {item['min_score']:5.1f} {item['reserve_mib']:7d} {item['n']:1d} "
            f"{item['mean_tps']:9.4f} {item['complete_layer_rate']*100:8.3f}% "
            f"{item['mean_resident_hit_rate']*100:8.3f}% "
            f"{item['mean_cpu_dependent_layers']:10.3f} {item['mean_urgent_jobs']:7.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
