#!/usr/bin/env python3
"""Compare GGML_COMPLETION_LOGITS_DUMP files without third-party packages."""

from __future__ import annotations

import argparse
from array import array
import json
import math
from pathlib import Path
import struct
import sys
from typing import BinaryIO

MAGIC = 0x4D4F454C


def open_dump(path: Path) -> tuple[BinaryIO, int]:
    stream = path.open("rb")
    header = stream.read(12)
    if len(header) != 12:
        raise ValueError(f"{path}: truncated header")
    magic, version, n_vocab = struct.unpack("<IIi", header)
    if magic != MAGIC or version != 1 or n_vocab <= 0:
        raise ValueError(
            f"{path}: unsupported header magic={magic:#x} version={version} n_vocab={n_vocab}"
        )
    return stream, n_vocab


def read_logits(stream: BinaryIO, n_vocab: int) -> array:
    values = array("f")
    try:
        values.fromfile(stream, n_vocab)
    except EOFError as exc:
        raise ValueError("truncated logit record") from exc
    if len(values) != n_vocab:
        raise ValueError("truncated logit record")
    if sys.byteorder != "little":
        values.byteswap()
    return values


def compare(reference: Path, candidate: Path) -> dict[str, object]:
    ref_stream, n_vocab = open_dump(reference)
    cand_stream, candidate_vocab = open_dump(candidate)
    if n_vocab != candidate_vocab:
        raise ValueError(f"vocabulary mismatch: {n_vocab} != {candidate_vocab}")

    per_step: list[dict[str, object]] = []
    total_count = 0
    total_abs = 0.0
    total_squared = 0.0
    global_max = 0.0

    while True:
        ref_header = ref_stream.read(12)
        cand_header = cand_stream.read(12)
        if bool(ref_header) != bool(cand_header):
            raise ValueError("record count mismatch")
        if not ref_header:
            break
        if len(ref_header) != 12 or len(cand_header) != 12:
            raise ValueError("truncated record header")

        ref_step, ref_token = struct.unpack("<Qi", ref_header)
        cand_step, cand_token = struct.unpack("<Qi", cand_header)
        if (ref_step, ref_token) != (cand_step, cand_token):
            raise ValueError(
                f"record mismatch: {(ref_step, ref_token)} != {(cand_step, cand_token)}"
            )

        ref_logits = read_logits(ref_stream, n_vocab)
        cand_logits = read_logits(cand_stream, n_vocab)
        ref_max = max(ref_logits)
        cand_max = max(cand_logits)

        step_abs = 0.0
        step_squared = 0.0
        step_max = 0.0
        ref_exp_sum = 0.0
        cand_exp_sum = 0.0
        for ref_value, cand_value in zip(ref_logits, cand_logits):
            difference = abs(float(ref_value) - float(cand_value))
            step_abs += difference
            step_squared += difference * difference
            step_max = max(step_max, difference)
            ref_exp_sum += math.exp(float(ref_value) - ref_max)
            cand_exp_sum += math.exp(float(cand_value) - cand_max)

        ref_log_z = ref_max + math.log(ref_exp_sum)
        cand_log_z = cand_max + math.log(cand_exp_sum)
        divergence = 0.0
        ref_top_index = -1
        cand_top_index = -1
        ref_top_value = -math.inf
        cand_top_value = -math.inf
        for index, (ref_value, cand_value) in enumerate(zip(ref_logits, cand_logits)):
            if ref_value > ref_top_value:
                ref_top_value = float(ref_value)
                ref_top_index = index
            if cand_value > cand_top_value:
                cand_top_value = float(cand_value)
                cand_top_index = index
            probability = math.exp(float(ref_value) - ref_log_z)
            if probability:
                divergence += probability * (
                    (float(ref_value) - ref_log_z) - (float(cand_value) - cand_log_z)
                )

        total_count += n_vocab
        total_abs += step_abs
        total_squared += step_squared
        global_max = max(global_max, step_max)
        per_step.append(
            {
                "step": ref_step,
                "forced_token": ref_token,
                "max_abs": step_max,
                "mean_abs": step_abs / n_vocab,
                "rms": math.sqrt(step_squared / n_vocab),
                "top1_reference": ref_top_index,
                "top1_candidate": cand_top_index,
                "top1_match": ref_top_index == cand_top_index,
                "forced_logit_abs": abs(
                    float(ref_logits[ref_token]) - float(cand_logits[ref_token])
                ),
                "kl_reference_to_candidate": divergence,
            }
        )

    if not per_step:
        raise ValueError("no records")

    return {
        "reference": str(reference),
        "candidate": str(candidate),
        "n_vocab": n_vocab,
        "steps": len(per_step),
        "max_abs": global_max,
        "mean_abs": total_abs / total_count,
        "rms": math.sqrt(total_squared / total_count),
        "top1_matches": sum(bool(item["top1_match"]) for item in per_step),
        "max_kl": max(float(item["kl_reference_to_candidate"]) for item in per_step),
        "mean_kl": sum(float(item["kl_reference_to_candidate"]) for item in per_step)
        / len(per_step),
        "per_step": per_step,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = compare(args.reference, args.candidate)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    summary = {key: value for key, value in result.items() if key != "per_step"}
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
