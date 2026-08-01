#!/usr/bin/env python3
"""Generate a same-token exact-route layer-completion schedule.

The input is a route trace from a matched control run. For each token, the
schedule considers only packets containing every expert that was cold in one
routed layer. Partial packets are never emitted. Candidates are ranked by
measured cold-branch time removed per uploaded expert, then admitted under a
single per-token bundle/byte budget.

Runtime format:
    step admit_layer target_layer expert victim_expert protected_experts[8]
         gross_value_us modeled_net_value_us

This is deliberately noncausal and intended only to test the completion-overlay
mechanism. It must never be used as a production predictor.
"""
from __future__ import annotations

import argparse
import collections
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

MIB = 1024 * 1024


@dataclass(frozen=True)
class Packet:
    step: int
    source_layer: int
    target_layer: int
    selected: tuple[int, ...]
    missing: tuple[int, ...]
    value_us: float

    @property
    def score(self) -> float:
        return self.value_us / len(self.missing)


def load_layer_values(path: Path | None, fallback_us: float) -> dict[int, float]:
    if path is None:
        return collections.defaultdict(lambda: fallback_us)
    data = json.loads(path.read_text())
    values: dict[int, float] = collections.defaultdict(lambda: fallback_us)
    for row in data.get("per_layer", []):
        if "cold_median_us" in row:
            values[int(row["layer"])] = float(row["cold_median_us"])
    return values


def load_routes(path: Path) -> dict[int, list[dict[str, object]]]:
    routes: dict[int, list[dict[str, object]]] = collections.defaultdict(list)
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") != "route" or event.get("phase") != "decode":
                continue
            selected = tuple(int(value) for value in event.get("selected", []))
            hot_slots = tuple(int(value) for value in event.get("hot_slots", []))
            if len(selected) != 8 or len(hot_slots) != len(selected):
                raise ValueError(
                    f"{path}:{line_number}: expected eight selected experts/hot slots"
                )
            missing = tuple(
                expert for expert, slot in zip(selected, hot_slots) if slot < 0
            )
            routes[int(event["step"])].append({
                "layer": int(event["layer"]),
                "selected": selected,
                "missing": missing,
            })
    if not routes:
        raise ValueError(f"{path}: no decode route events")
    for step in routes:
        routes[step].sort(key=lambda row: int(row["layer"]))
    return dict(sorted(routes.items()))


def select_packets(
    routes: dict[int, list[dict[str, object]]],
    layer_values: dict[int, float],
    max_packet_experts: int,
    max_uploads_per_token: int,
    lead_layers: int,
) -> list[Packet]:
    schedule: list[Packet] = []
    for step, rows in routes.items():
        candidates: list[Packet] = []
        for row in rows:
            layer = int(row["layer"])
            selected = tuple(int(value) for value in row["selected"])
            missing = tuple(dict.fromkeys(int(value) for value in row["missing"]))
            if not missing or len(missing) > max_packet_experts or layer <= 0:
                continue
            source = max(0, layer - lead_layers)
            if source >= layer:
                continue
            candidates.append(Packet(
                step=step,
                source_layer=source,
                target_layer=layer,
                selected=selected,
                missing=missing,
                value_us=float(layer_values[layer]),
            ))

        # Completion value per uploaded bundle is the primary objective. Prefer
        # earlier deadlines only after value and packet size are tied.
        candidates.sort(key=lambda packet: (
            -packet.score,
            len(packet.missing),
            packet.target_layer,
        ))
        used = 0
        chosen: list[Packet] = []
        for packet in candidates:
            if used + len(packet.missing) > max_uploads_per_token:
                continue
            chosen.append(packet)
            used += len(packet.missing)
        schedule.extend(sorted(chosen, key=lambda packet: (
            packet.source_layer,
            packet.target_layer,
        )))
    return schedule


def write_schedule(path: Path, packets: Iterable[Packet]) -> dict[str, object]:
    packets = list(packets)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        stream.write(
            "# step admit_layer target_layer expert victim_expert "
            "protected_experts[8] gross_value_us modeled_net_value_us\n"
        )
        for packet in packets:
            protected = " ".join(str(expert) for expert in packet.selected)
            per_expert_value = packet.value_us / len(packet.missing)
            for expert in packet.missing:
                stream.write(
                    f"{packet.step} {packet.source_layer} {packet.target_layer} "
                    f"{expert} -1 {protected} "
                    f"{per_expert_value:.3f} {per_expert_value:.3f}\n"
                )

    by_step = collections.Counter(packet.step for packet in packets)
    packet_sizes = collections.Counter(len(packet.missing) for packet in packets)
    return {
        "packets": len(packets),
        "schedule_entries": sum(len(packet.missing) for packet in packets),
        "steps": len(by_step),
        "mean_packets_per_step": (
            len(packets) / len(by_step) if by_step else 0.0
        ),
        "max_packets_per_step": max(by_step.values(), default=0),
        "packet_size_histogram": dict(sorted(packet_sizes.items())),
        "modeled_value_us": sum(packet.value_us for packet in packets),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--layer-values", type=Path)
    parser.add_argument("--fallback-layer-us", type=float, default=327.0)
    parser.add_argument("--bundle-bytes", type=int, default=4_079_616)
    parser.add_argument("--max-packet-experts", type=int, default=3)
    parser.add_argument("--max-upload-mib-per-token", type=float, default=64.0)
    parser.add_argument("--max-uploads-per-token", type=int, default=16)
    parser.add_argument("--lead-layers", type=int, default=12)
    args = parser.parse_args()

    byte_limit = int(args.max_upload_mib_per_token * MIB)
    byte_bundles = byte_limit // args.bundle_bytes
    max_uploads = min(args.max_uploads_per_token, byte_bundles)
    if max_uploads <= 0:
        parser.error("upload budget cannot fit one expert bundle")
    if args.max_packet_experts <= 0 or args.lead_layers <= 0:
        parser.error("packet size and lead layers must be positive")

    routes = load_routes(args.trace)
    values = load_layer_values(args.layer_values, args.fallback_layer_us)
    packets = select_packets(
        routes,
        values,
        args.max_packet_experts,
        max_uploads,
        args.lead_layers,
    )
    summary = write_schedule(args.output, packets)
    summary.update({
        "trace": str(args.trace),
        "output": str(args.output),
        "tokens": len(routes),
        "bundle_bytes": args.bundle_bytes,
        "max_packet_experts": args.max_packet_experts,
        "max_upload_mib_per_token": args.max_upload_mib_per_token,
        "max_uploads_per_token": max_uploads,
        "lead_layers": args.lead_layers,
    })
    summary_path = args.output.with_suffix(args.output.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
