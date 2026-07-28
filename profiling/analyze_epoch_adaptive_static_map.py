#!/usr/bin/env python3
"""Simulate low-frequency adaptation around a frozen GLM expert-map core.

The simulation keeps most slots pinned to the measured static-frequency map and
allows only a small per-layer tail to change. Changes are committed at coarse
epoch boundaries under a global migration budget and hysteresis threshold.

This is a residency-policy simulation. It does not model graph publication,
copy overlap, or exact end-to-end throughput. The authoritative route trace is
scored before each online update.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Sequence


Route = tuple[int, ...]


def parse_ints(text: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in text.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected comma-separated integers")
    return values


def parse_floats(text: str) -> tuple[float, ...]:
    values = tuple(float(item) for item in text.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected comma-separated numbers")
    return values


def load_routes(path: Path) -> tuple[list[int], list[int], list[dict[int, Route]]]:
    by_step: dict[int, dict[int, Route]] = defaultdict(dict)
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
            by_step[step][layer] = tuple(dict.fromkeys(int(value) for value in event["selected"]))
    if not by_step:
        raise ValueError(f"no decode route events in {path}")
    max_layers = max(len(value) for value in by_step.values())
    layer_sets = Counter(
        frozenset(value)
        for value in by_step.values()
        if len(value) == max_layers
    )
    expected = set(layer_sets.most_common(1)[0][0])
    layers = sorted(expected)
    steps = sorted(step for step, value in by_step.items() if set(value) == expected)
    tokens = [{layer: by_step[step][layer] for layer in layers} for step in steps]
    return steps, layers, tokens


def load_rankings(path: Path) -> dict[int, list[int]]:
    rankings: dict[int, list[int]] = {}
    with path.open() as stream:
        for line in stream:
            fields = line.split()
            if not fields or fields[0].startswith("#"):
                continue
            layer = int(fields[0])
            ranking = []
            seen = set()
            for value in fields[1:]:
                expert = int(value)
                if expert not in seen:
                    ranking.append(expert)
                    seen.add(expert)
            rankings[layer] = ranking
    return rankings


def load_capacities(path: Path) -> dict[int, int]:
    data = json.loads(path.read_text())
    return {int(row["layer"]): int(row["slots"]) for row in data["layers"]}


@dataclass
class Proposal:
    benefit: float
    ratio: float
    layer: int
    candidate: int
    victim: int


class EpochPolicy:
    def __init__(
        self,
        *,
        layers: Sequence[int],
        capacities: dict[int, int],
        rankings: dict[int, list[int]],
        train_tokens: Sequence[dict[int, Route]],
        tail_slots: int,
        epoch_tokens: int,
        migrations_per_epoch: int,
        admission_ratio: float,
        min_residence: int,
        decay: float,
    ) -> None:
        self.layers = list(layers)
        self.capacities = capacities
        self.tail_slots_requested = tail_slots
        self.epoch_tokens = epoch_tokens
        self.migrations_per_epoch = migrations_per_epoch
        self.admission_ratio = admission_ratio
        self.min_residence = min_residence
        self.decay = decay
        self.score_scale = 1.0
        self.scores = {layer: [0.0] * 256 for layer in layers}
        for token in train_tokens:
            self._decay_and_observe(token)

        self.core: dict[int, set[int]] = {}
        self.tail: dict[int, set[int]] = {}
        self.admitted_at: dict[int, dict[int, int]] = {}
        for layer in layers:
            capacity = capacities[layer]
            tail_count = min(tail_slots, max(0, capacity - 1))
            ranking = rankings[layer]
            self.core[layer] = set(ranking[: capacity - tail_count])
            self.tail[layer] = set(ranking[capacity - tail_count : capacity])
            self.admitted_at[layer] = {expert: 0 for expert in self.tail[layer]}

        self.routes = 0
        self.hits = 0
        self.core_hits = 0
        self.tail_hits = 0
        self.transfers = 0
        self.epochs = 0
        self.epochs_with_changes = 0
        self.proposals_considered = 0
        self.rejected_hysteresis = 0
        self.rejected_residence = 0
        self.migrations_by_layer = Counter()
        self.tail_hit_events_by_expert = Counter()
        self.replaced_before_hit = 0
        self.hit_since_admit: dict[int, dict[int, bool]] = {
            layer: {expert: False for expert in self.tail[layer]} for layer in layers
        }
        self.coverage_by_window: list[dict[str, float | int]] = []
        self._window_routes = 0
        self._window_hits = 0

    def _decay_and_observe(self, token: dict[int, Route]) -> None:
        self.score_scale *= self.decay
        if self.score_scale < 1e-100:
            for layer in self.layers:
                values = self.scores[layer]
                for expert in range(256):
                    values[expert] *= self.score_scale
            self.score_scale = 1.0
        increment = 1.0 / self.score_scale
        for layer in self.layers:
            values = self.scores[layer]
            for expert in token[layer]:
                values[expert] += increment

    def score_token(self, token: dict[int, Route], token_number: int) -> None:
        for layer in self.layers:
            core = self.core[layer]
            tail = self.tail[layer]
            for expert in token[layer]:
                self.routes += 1
                self._window_routes += 1
                if expert in core:
                    self.hits += 1
                    self.core_hits += 1
                    self._window_hits += 1
                elif expert in tail:
                    self.hits += 1
                    self.tail_hits += 1
                    self._window_hits += 1
                    self.hit_since_admit[layer][expert] = True
                    self.tail_hit_events_by_expert[(layer, expert)] += 1
        self._decay_and_observe(token)
        if token_number % self.epoch_tokens == 0:
            self.commit(token_number)
        if token_number % 64 == 0:
            self.coverage_by_window.append({
                "token_end": token_number,
                "coverage": self._window_hits / max(1, self._window_routes),
            })
            self._window_routes = 0
            self._window_hits = 0

    def commit(self, token_number: int) -> None:
        self.epochs += 1
        proposals: list[Proposal] = []
        for layer in self.layers:
            if not self.tail[layer]:
                continue
            scores = self.scores[layer]
            resident = self.core[layer] | self.tail[layer]
            candidate = max((expert for expert in range(256) if expert not in resident), key=scores.__getitem__)
            victim = min(self.tail[layer], key=scores.__getitem__)
            candidate_score = scores[candidate]
            victim_score = scores[victim]
            self.proposals_considered += 1
            admitted_at = self.admitted_at[layer].get(victim, 0)
            if token_number - admitted_at < self.min_residence:
                self.rejected_residence += 1
                continue
            ratio = candidate_score / max(victim_score, 1e-9)
            if candidate_score <= 0.0 or ratio < self.admission_ratio:
                self.rejected_hysteresis += 1
                continue
            proposals.append(Proposal(
                benefit=candidate_score - victim_score,
                ratio=ratio,
                layer=layer,
                candidate=candidate,
                victim=victim,
            ))
        proposals.sort(key=lambda item: (item.benefit, item.ratio), reverse=True)
        changed = 0
        for proposal in proposals[: self.migrations_per_epoch]:
            layer = proposal.layer
            if not self.hit_since_admit[layer].get(proposal.victim, False):
                self.replaced_before_hit += 1
            self.tail[layer].remove(proposal.victim)
            self.admitted_at[layer].pop(proposal.victim, None)
            self.hit_since_admit[layer].pop(proposal.victim, None)
            self.tail[layer].add(proposal.candidate)
            self.admitted_at[layer][proposal.candidate] = token_number
            self.hit_since_admit[layer][proposal.candidate] = False
            self.transfers += 1
            self.migrations_by_layer[layer] += 1
            changed += 1
        self.epochs_with_changes += changed > 0

    def finish(self, token_count: int) -> dict[str, object]:
        if self._window_routes:
            self.coverage_by_window.append({
                "token_end": token_count,
                "coverage": self._window_hits / max(1, self._window_routes),
            })
        useful_tail_experts = sum(
            1 for value in self.tail_hit_events_by_expert.values() if value > 0
        )
        return {
            "tail_slots": self.tail_slots_requested,
            "epoch_tokens": self.epoch_tokens,
            "migrations_per_epoch": self.migrations_per_epoch,
            "admission_ratio": self.admission_ratio,
            "min_residence": self.min_residence,
            "decay": self.decay,
            "routes": self.routes,
            "hits": self.hits,
            "coverage": self.hits / max(1, self.routes),
            "core_coverage": self.core_hits / max(1, self.routes),
            "tail_coverage": self.tail_hits / max(1, self.routes),
            "transfers": self.transfers,
            "transfers_per_token": self.transfers / max(1, token_count),
            "tail_hits_per_transfer": self.tail_hits / max(1, self.transfers),
            "useful_tail_experts": useful_tail_experts,
            "replaced_before_first_hit": self.replaced_before_hit,
            "epochs": self.epochs,
            "epochs_with_changes": self.epochs_with_changes,
            "proposals_considered": self.proposals_considered,
            "rejected_hysteresis": self.rejected_hysteresis,
            "rejected_min_residence": self.rejected_residence,
            "most_migrated_layers": self.migrations_by_layer.most_common(10),
            "coverage_by_64_token_window": self.coverage_by_window,
        }


def static_coverage(
    tokens: Sequence[dict[int, Route]],
    layers: Sequence[int],
    capacities: dict[int, int],
    rankings: dict[int, list[int]],
) -> dict[str, float | int]:
    resident = {layer: set(rankings[layer][: capacities[layer]]) for layer in layers}
    routes = hits = 0
    for token in tokens:
        for layer in layers:
            for expert in token[layer]:
                routes += 1
                hits += expert in resident[layer]
    return {"routes": routes, "hits": hits, "coverage": hits / max(1, routes)}


def run_sweep(
    *,
    label: str,
    tokens: Sequence[dict[int, Route]],
    train_tokens: Sequence[dict[int, Route]],
    layers: Sequence[int],
    capacities: dict[int, int],
    rankings: dict[int, list[int]],
    tail_slots: Sequence[int],
    epochs: Sequence[int],
    migration_budgets: Sequence[int],
    ratios: Sequence[float],
    min_residences: Sequence[int],
    decay: float,
) -> dict[str, object]:
    baseline = static_coverage(tokens, layers, capacities, rankings)
    results = []
    for tail in tail_slots:
        for epoch in epochs:
            for migrations in migration_budgets:
                for ratio in ratios:
                    for min_residence in min_residences:
                        policy = EpochPolicy(
                            layers=layers,
                            capacities=capacities,
                            rankings=rankings,
                            train_tokens=train_tokens,
                            tail_slots=tail,
                            epoch_tokens=epoch,
                            migrations_per_epoch=migrations,
                            admission_ratio=ratio,
                            min_residence=min_residence,
                            decay=decay,
                        )
                        for token_number, token in enumerate(tokens, start=1):
                            policy.score_token(token, token_number)
                        result = policy.finish(len(tokens))
                        result["coverage_gain_over_static"] = result["coverage"] - baseline["coverage"]
                        result["incremental_hits_per_transfer"] = (
                            (result["hits"] - baseline["hits"]) / max(1, result["transfers"])
                        )
                        results.append(result)

    # Pareto frontier: no other point has both greater/equal coverage and fewer/equal transfers.
    frontier = []
    for candidate in results:
        dominated = any(
            other["coverage"] >= candidate["coverage"]
            and other["transfers_per_token"] <= candidate["transfers_per_token"]
            and (
                other["coverage"] > candidate["coverage"]
                or other["transfers_per_token"] < candidate["transfers_per_token"]
            )
            for other in results
        )
        if not dominated:
            frontier.append(candidate)
    frontier.sort(key=lambda row: (row["transfers_per_token"], -row["coverage"]))

    budgets = [0.05, 0.10, 0.25, 0.50, 1.0, 2.0, 4.0]
    best_under_budget = []
    for budget in budgets:
        eligible = [row for row in results if row["transfers_per_token"] <= budget]
        if not eligible:
            continue
        best = max(eligible, key=lambda row: (row["coverage"], -row["transfers_per_token"]))
        best_under_budget.append({"budget": budget, "result": best})

    return {
        "label": label,
        "tokens": len(tokens),
        "static_baseline": baseline,
        "results": results,
        "pareto_frontier": frontier,
        "best_under_transfer_budget": best_under_budget,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-trace", type=Path, required=True)
    parser.add_argument("--shift-trace", type=Path)
    parser.add_argument("--static-map", type=Path, required=True)
    parser.add_argument("--static-analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-tokens", type=int, default=64)
    parser.add_argument("--tail-slots", type=parse_ints, default=(1, 2, 4))
    parser.add_argument("--epochs", type=parse_ints, default=(4, 8, 16, 32, 64))
    parser.add_argument("--migration-budgets", type=parse_ints, default=(1, 2, 4, 8, 16))
    parser.add_argument("--ratios", type=parse_floats, default=(1.05, 1.15, 1.30))
    parser.add_argument("--min-residences", type=parse_ints, default=(8, 16, 32))
    parser.add_argument("--decay", type=float, default=0.98)
    args = parser.parse_args()

    steps, layers, primary = load_routes(args.primary_trace)
    rankings = load_rankings(args.static_map)
    capacities = load_capacities(args.static_analysis)
    missing = [layer for layer in layers if layer not in rankings or layer not in capacities]
    if missing:
        raise ValueError(f"static map/capacities missing layers: {missing}")
    train = primary[: args.train_tokens]
    evaluation = primary[args.train_tokens :]
    studies = [run_sweep(
        label="primary_trace_post_training",
        tokens=evaluation,
        train_tokens=train,
        layers=layers,
        capacities=capacities,
        rankings=rankings,
        tail_slots=args.tail_slots,
        epochs=args.epochs,
        migration_budgets=args.migration_budgets,
        ratios=args.ratios,
        min_residences=args.min_residences,
        decay=args.decay,
    )]
    if args.shift_trace:
        shift_steps, shift_layers, shifted = load_routes(args.shift_trace)
        if shift_layers != layers:
            raise ValueError("shift trace has a different routed-layer set")
        studies.append(run_sweep(
            label="secondary_trace_shift",
            tokens=shifted,
            train_tokens=train,
            layers=layers,
            capacities=capacities,
            rankings=rankings,
            tail_slots=args.tail_slots,
            epochs=args.epochs,
            migration_budgets=args.migration_budgets,
            ratios=args.ratios,
            min_residences=args.min_residences,
            decay=args.decay,
        ))
    output = {
        "primary_trace": str(args.primary_trace),
        "primary_step_range": [steps[0], steps[-1]],
        "shift_trace": str(args.shift_trace) if args.shift_trace else None,
        "methodology": {
            "evaluation": "route is scored against current residency before online score update",
            "base_map": "measured GLM static-frequency ranking with actual 13/14/17 per-layer capacities",
            "adaptation": "protected static core plus decayed-frequency adaptive tail",
            "publication": "coarse token epochs with global migration budget, hysteresis, and minimum residence",
            "limitation": "residency simulation only; does not model graph/map epoch publication or transfer overlap",
        },
        "configuration": {
            "train_tokens": args.train_tokens,
            "tail_slots": list(args.tail_slots),
            "epochs": list(args.epochs),
            "migration_budgets": list(args.migration_budgets),
            "ratios": list(args.ratios),
            "min_residences": list(args.min_residences),
            "decay": args.decay,
        },
        "studies": studies,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    for study in studies:
        print(f"\n{study['label']}: static={100*study['static_baseline']['coverage']:.3f}%")
        for row in study["best_under_transfer_budget"]:
            result = row["result"]
            print(
                f"budget<={row['budget']:.2f}/tok coverage={100*result['coverage']:.3f}% "
                f"gain={100*result['coverage_gain_over_static']:+.3f}pp "
                f"xfer={result['transfers_per_token']:.4f}/tok "
                f"tail={result['tail_slots']} epoch={result['epoch_tokens']} "
                f"migrations={result['migrations_per_epoch']} ratio={result['admission_ratio']} "
                f"minres={result['min_residence']} hits/xfer={result['incremental_hits_per_transfer']:.2f}"
            )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
