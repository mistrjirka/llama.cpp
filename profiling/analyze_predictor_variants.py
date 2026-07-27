#!/usr/bin/env python3
"""Prequential comparison of stronger route-only adjacent-layer predictors."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import itertools
import json
import math
from pathlib import Path


def load(path: Path):
    by_step = defaultdict(dict)
    with path.open() as stream:
        for line in stream:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("event") == "route" and record.get("phase") == "decode":
                by_step[int(record["step"])][int(record["layer"])] = tuple(
                    dict.fromkeys(map(int, record["selected"]))
                )
    steps = sorted(by_step)
    layers = sorted(set.intersection(*(set(by_step[step]) for step in steps)))
    return layers, [{layer: by_step[step][layer] for layer in layers} for step in steps]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--window", type=int, default=64)
    args = parser.parse_args()

    layers, tokens = load(args.trace)
    ecount = args.experts
    ks = (4, 8, 16)
    methods = (
        "current_conditional",
        "rank_weighted",
        "positive_pmi",
        "naive_bayes_presence",
        "source_pair_interaction",
        "cross_Lminus2_interaction",
        "all_interactions",
        "sliding64_conditional",
    )
    metrics = {method: {k: Counter() for k in ks} for method in methods}

    models = {}
    for index in range(1, len(layers)):
        previous = layers[index - 1]
        target = layers[index]
        models[target] = {
            "co": [Counter() for _ in range(ecount)],
            "source_n": Counter(),
            "target_n": Counter(),
            "events": 0,
            "source_pairs": defaultdict(Counter),
            "cross_pairs": defaultdict(Counter),
            "window": deque(),
            "window_co": [Counter() for _ in range(ecount)],
            "window_source_n": Counter(),
            "window_target_n": Counter(),
        }

    def rank(scores, fallback):
        return sorted(range(ecount), key=lambda expert: (-scores[expert], -fallback[expert], expert))

    def record(method, ranked, actual):
        actual_set = set(actual)
        for k in ks:
            hits = len(actual_set.intersection(ranked[:k]))
            counter = metrics[method][k]
            counter["events"] += 1
            counter["routes"] += len(actual_set)
            counter["hits"] += hits
            counter["complete"] += hits == len(actual_set)

    for token in tokens:
        for index in range(1, len(layers)):
            source_layer = layers[index - 1]
            target_layer = layers[index]
            source = token[source_layer]
            actual = token[target_layer]
            model = models[target_layer]
            fallback = model["target_n"]

            current = [0.0] * ecount
            ranked_source = [0.0] * ecount
            pmi = [0.0] * ecount
            naive = [0.0] * ecount
            sliding = [0.0] * ecount
            total = max(1, model["events"])
            window_total = max(1, len(model["window"]))
            for rank_index, source_expert in enumerate(source):
                source_n = model["source_n"][source_expert]
                window_source_n = model["window_source_n"][source_expert]
                if source_n:
                    for target_expert, count in model["co"][source_expert].items():
                        conditional = count / source_n
                        current[target_expert] += conditional
                        ranked_source[target_expert] += conditional / (rank_index + 1)
                        target_n = model["target_n"][target_expert]
                        association = math.log(
                            ((count + 0.5) * (total + 1.0))
                            / ((source_n + 0.5) * (target_n + 0.5))
                        )
                        if association > 0:
                            pmi[target_expert] += association
                # Bernoulli naive Bayes using only positive source features.
                for target_expert in range(ecount):
                    target_n = model["target_n"][target_expert]
                    count = model["co"][source_expert][target_expert]
                    naive[target_expert] += math.log((count + 0.5) / (target_n + 1.0))
                if window_source_n:
                    for target_expert, count in model["window_co"][source_expert].items():
                        sliding[target_expert] += count / window_source_n
            for target_expert in range(ecount):
                naive[target_expert] += math.log((model["target_n"][target_expert] + 0.5) / (total + ecount * 0.5))

            source_pair_score = current.copy()
            for pair in itertools.combinations(sorted(source), 2):
                pair_counts = model["source_pairs"].get(pair)
                if not pair_counts:
                    continue
                denominator = sum(pair_counts.values()) / max(1, len(actual))
                if denominator <= 0:
                    continue
                for target_expert, count in pair_counts.items():
                    source_pair_score[target_expert] += 0.25 * count / denominator

            cross_score = current.copy()
            if index >= 2:
                source2 = token[layers[index - 2]]
                for expert2 in source2:
                    for expert1 in source:
                        pair_counts = model["cross_pairs"].get((expert2, expert1))
                        if not pair_counts:
                            continue
                        denominator = sum(pair_counts.values()) / max(1, len(actual))
                        if denominator <= 0:
                            continue
                        for target_expert, count in pair_counts.items():
                            cross_score[target_expert] += 0.25 * count / denominator

            all_score = [a + b + c - 2 * d for a, b, c, d in zip(source_pair_score, cross_score, current, current)]
            record("current_conditional", rank(current, fallback), actual)
            record("rank_weighted", rank(ranked_source, fallback), actual)
            record("positive_pmi", rank(pmi, fallback), actual)
            record("naive_bayes_presence", rank(naive, fallback), actual)
            record("source_pair_interaction", rank(source_pair_score, fallback), actual)
            record("cross_Lminus2_interaction", rank(cross_score, fallback), actual)
            record("all_interactions", rank(all_score, fallback), actual)
            record("sliding64_conditional", rank(sliding, model["window_target_n"]), actual)

        # Update after predicting every target in this token.
        for index in range(1, len(layers)):
            source_layer = layers[index - 1]
            target_layer = layers[index]
            source = token[source_layer]
            actual = token[target_layer]
            model = models[target_layer]
            model["events"] += 1
            model["target_n"].update(actual)
            for source_expert in source:
                model["source_n"][source_expert] += 1
                model["co"][source_expert].update(actual)
            for pair in itertools.combinations(sorted(source), 2):
                model["source_pairs"][pair].update(actual)
            if index >= 2:
                source2 = token[layers[index - 2]]
                for expert2 in source2:
                    for expert1 in source:
                        model["cross_pairs"][(expert2, expert1)].update(actual)

            model["window"].append((source, actual))
            model["window_target_n"].update(actual)
            for source_expert in source:
                model["window_source_n"][source_expert] += 1
                model["window_co"][source_expert].update(actual)
            while len(model["window"]) > args.window:
                old_source, old_actual = model["window"].popleft()
                model["window_target_n"].subtract(old_actual)
                for source_expert in old_source:
                    model["window_source_n"][source_expert] -= 1
                    model["window_co"][source_expert].subtract(old_actual)

    result = {"tokens": len(tokens), "layers": layers, "methods": {}}
    for method in methods:
        result["methods"][method] = {}
        for k in ks:
            counter = metrics[method][k]
            result["methods"][method][str(k)] = {
                "route_recall": counter["hits"] / counter["routes"],
                "complete_rate": counter["complete"] / counter["events"],
                "mean_hits": counter["hits"] / counter["events"],
            }
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("method                          k=4 recall  k=8 recall  k=16 recall  k=8 complete")
    for method in methods:
        values = result["methods"][method]
        print(
            f"{method:31s} {values['4']['route_recall']*100:10.2f}% "
            f"{values['8']['route_recall']*100:10.2f}% "
            f"{values['16']['route_recall']*100:11.2f}% "
            f"{values['8']['complete_rate']*100:12.2f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
