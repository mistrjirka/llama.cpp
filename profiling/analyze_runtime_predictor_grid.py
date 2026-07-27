#!/usr/bin/env python3
"""Prequential grid for runtime-friendly adjacent-layer predictor variants."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import itertools
import json
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


def halve(counter: Counter):
    for key in list(counter):
        value = (counter[key] + 1) // 2
        if value:
            counter[key] = value
        else:
            del counter[key]


def evaluate(layers, tokens, decay_interval: int, interaction_weight: float):
    ks = (1, 2, 4, 8, 16)
    metric = {k: Counter() for k in ks}
    models = {}
    for index in range(2, len(layers)):
        target = layers[index]
        models[target] = {
            "co": [Counter() for _ in range(256)],
            "source_n": Counter(),
            "joint": defaultdict(Counter),
            "joint_n": Counter(),
            "marginal": Counter(),
            "last_decay": 0,
        }

    for token_index, token in enumerate(tokens, start=1):
        for index in range(2, len(layers)):
            l2, l1, target = layers[index - 2], layers[index - 1], layers[index]
            source2, source1, actual = token[l2], token[l1], token[target]
            model = models[target]
            if decay_interval and token_index >= model["last_decay"] + decay_interval:
                for counter in model["co"]:
                    halve(counter)
                halve(model["source_n"])
                for key in list(model["joint"]):
                    halve(model["joint"][key])
                    if not model["joint"][key]:
                        del model["joint"][key]
                halve(model["joint_n"])
                halve(model["marginal"])
                model["last_decay"] = token_index

            scores = [0.0] * 256
            for source in source1:
                denominator = model["source_n"][source]
                if denominator:
                    for expert, count in model["co"][source].items():
                        scores[expert] += count / denominator
            if interaction_weight:
                for expert2 in source2:
                    for expert1 in source1:
                        key = (expert2, expert1)
                        denominator = model["joint_n"][key]
                        if denominator:
                            for expert, count in model["joint"][key].items():
                                scores[expert] += interaction_weight * count / denominator
            ranked = sorted(
                range(256),
                key=lambda expert: (-scores[expert], -model["marginal"][expert], expert),
            )
            actual_set = set(actual)
            for k in ks:
                hits = len(actual_set.intersection(ranked[:k]))
                m = metric[k]
                m["hits"] += hits
                m["routes"] += len(actual_set)
                m["events"] += 1
                m["any"] += hits > 0
                m["complete"] += hits == len(actual_set)

        # Update after every target was predicted.
        for index in range(2, len(layers)):
            l2, l1, target = layers[index - 2], layers[index - 1], layers[index]
            source2, source1, actual = token[l2], token[l1], token[target]
            model = models[target]
            model["marginal"].update(actual)
            for source in source1:
                model["source_n"][source] += 1
                model["co"][source].update(actual)
            for expert2 in source2:
                for expert1 in source1:
                    key = (expert2, expert1)
                    model["joint_n"][key] += 1
                    model["joint"][key].update(actual)

    return {
        str(k): {
            "route_recall": metric[k]["hits"] / metric[k]["routes"],
            "any_rate": metric[k]["any"] / metric[k]["events"],
            "complete_rate": metric[k]["complete"] / metric[k]["events"],
            "mean_hits": metric[k]["hits"] / metric[k]["events"],
        }
        for k in ks
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    layers, tokens = load(args.trace)
    configs = []
    for decay in (0, 32, 64, 128):
        configs.append((f"base_decay{decay}", decay, 0.0))
    for weight in (0.1, 0.25, 0.5, 1.0):
        configs.append((f"joint_w{weight:g}_decay0", 0, weight))
    for decay in (32, 64, 128):
        configs.append((f"joint_w0.25_decay{decay}", decay, 0.25))

    result = {"tokens": len(tokens), "layers": layers, "configs": {}}
    for name, decay, weight in configs:
        result["configs"][name] = {
            "decay_interval": decay,
            "interaction_weight": weight,
            "metrics": evaluate(layers, tokens, decay, weight),
        }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("config                         top1-hit  k4-recall k8-recall k8-complete")
    for name, item in result["configs"].items():
        m = item["metrics"]
        print(
            f"{name:30s} {m['1']['any_rate']*100:8.2f}% "
            f"{m['4']['route_recall']*100:9.2f}% "
            f"{m['8']['route_recall']*100:9.2f}% "
            f"{m['8']['complete_rate']*100:11.2f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
