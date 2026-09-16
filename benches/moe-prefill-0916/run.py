#!/usr/bin/env python3
"""Run the opt-in Flash-Next prefill benchmark with explicit hardware presets."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import shutil
import statistics
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PRESETS = {
    "v100": (["V100"], [48], [14]),
    "rtx2080ti": (["RTX 2080 Ti"], [48], [6]),
    "v100-rtx2080ti": (["V100", "RTX 2080 Ti"], [36, 12], [16, 10]),
    "rtx2080ti-v100": (["RTX 2080 Ti", "V100"], [12, 36], [10, 16]),
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=PRESETS, required=True)
    parser.add_argument("--model", type=Path, required=True, help="Local GGUF, or first part of a split GGUF")
    parser.add_argument("--build", type=Path, default=ROOT / "build", help="Existing CUDA build directory")
    parser.add_argument("--output", type=Path, required=True, help="New output directory; never overwritten")
    parser.add_argument("--devices", help="Comma-separated GPU UUIDs in preset order; auto-detected when omitted")
    parser.add_argument("--base", help="Override per-device base expert-cache allowances, e.g. 14 or 16,10")
    parser.add_argument("--workspace-mib", type=int, default=3072)
    parser.add_argument("--tokens", type=int, help="New input tokens: default 4096 cold or 1000 with saved prefix")
    parser.add_argument("--prefix-state", type=Path, help="Compatible saved sequence state; never silently regenerated")
    parser.add_argument("--prefix-tokens", type=Path, help="Original int32 token IDs matching the saved prefix")
    parser.add_argument("--prefix-length", type=int, default=100000)
    parser.add_argument("--suffix", type=Path, help="Text fixture; defaults to Python for cold input or C++ for a saved prefix")
    parser.add_argument("--cuda-root", type=Path, help="CUDA toolkit root; default from nvcc")
    parser.add_argument("--compile-only", action="store_true", help="Compile the harness without loading a model")
    args = parser.parse_args()

    build, model, out = args.build.resolve(), args.model.resolve(), args.output.resolve()
    lib = build / "bin"
    if not (lib / "libllama.so").is_file() or not (lib / "libggml-cuda.so").is_file():
        parser.error("Build llama-server with CUDA first; expected libllama.so and libggml-cuda.so in BUILD/bin")
    nvcc = shutil.which("nvcc")
    cuda = args.cuda_root.resolve() if args.cuda_root else Path(nvcc).resolve().parent.parent if nvcc else None
    if cuda is None or not (cuda / "include/cuda_runtime_api.h").is_file():
        parser.error("CUDA headers not found; pass --cuda-root /path/to/cuda")
    names, layers, bases = PRESETS[args.preset]
    if args.base:
        try:
            bases = [int(x) for x in args.base.split(",")]
        except ValueError:
            parser.error("--base requires comma-separated integers")
    if len(bases) != len(layers) or any(b < 0 or b > n for b, n in zip(bases, layers)):
        parser.error("--base must supply one allowance per preset device, within its layer count")
    if not 1 <= args.workspace_mib <= 65536:
        parser.error("--workspace-mib must be between 1 and 65536")
    if bool(args.prefix_state) != bool(args.prefix_tokens):
        parser.error("--prefix-state and --prefix-tokens are required together")
    prefix = args.prefix_length if args.prefix_state else 0
    n = args.tokens if args.tokens is not None else 1000 if prefix else 4096
    args.suffix = args.suffix or HERE / "fixtures" / ("cpp.txt" if prefix else "python.txt")
    if not 32 <= n <= 131072 or prefix < 0 or prefix + n >= 250000:
        parser.error("Unsupported request size for this diagnostic harness")
    for path in [model, args.suffix, args.prefix_state, args.prefix_tokens]:
        if path is not None and not path.is_file():
            parser.error(f"File not found: {path}")
    if args.prefix_state and prefix <= 0:
        parser.error("--prefix-length must be positive with a saved prefix")
    for path in [args.suffix, args.prefix_state, args.prefix_tokens]:
        if path is not None and "," in str(path.resolve()):
            parser.error("The harness's fixture sequence cannot represent commas in paths")
    if out.exists():
        parser.error(f"Output already exists: {out}; choose a new directory")

    compiler = shlex.split(os.environ.get("CXX", "c++"))
    binary = out / "bench-prefill"
    compile_command = compiler + ["-std=c++17", "-O2", str(HERE / "bench-prefill.cpp")]
    compile_command += [f"-I{ROOT / sub}" for sub in ("include", "ggml/include", "src")]
    compile_command += [f"-I{cuda / 'include'}", f"-L{lib}", f"-Wl,-rpath,{lib}"]
    compile_command += ["-lllama", "-lggml", "-lggml-base", "-lggml-cuda", "-lggml-cpu"]
    compile_command += [f"-L{cuda / 'lib64'}", "-lcudart", "-ldl", "-o", str(binary)]
    out.mkdir(parents=True)
    (out / "compile.json").write_text(json.dumps(compile_command, indent=2) + "\n")
    subprocess.run(compile_command, check=True)
    if args.compile_only:
        print(binary)
        return

    inventory = subprocess.check_output([
        "nvidia-smi", "--query-gpu=uuid,name,memory.total", "--format=csv,noheader,nounits",
    ], text=True)
    gpus = [{"uuid": row[0].strip(), "name": row[1].strip(), "memory_mib": int(row[2])}
            for row in csv.reader(io.StringIO(inventory)) if row]
    explicit = args.devices.split(",") if args.devices else None
    if explicit is not None and len(explicit) != len(names):
        parser.error("--devices must match the preset's device count")
    selected = []
    for i, name in enumerate(names):
        candidates = [g for g in gpus if name in g["name"] and g not in selected]
        if explicit is not None:
            candidates = [g for g in candidates if g["uuid"] == explicit[i].strip()]
        if not candidates:
            parser.error(f"No matching device for preset position {i}: {name}")
        gpu = candidates[0]
        minimum = 30720 if name == "V100" else 21504
        if gpu["memory_mib"] < minimum:
            parser.error(f"{gpu['name']} has less memory than this measured preset; use a smaller model or a separate manual test")
        selected.append(gpu)
    print("Devices:", json.dumps(selected), flush=True)
    variants = [0, 0, 1, 1, 0, 1, 0]
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("BENCH_", "LLAMA_MOE_", "GGML_CUDA_", "QWEN4EXP_", "DIAG_", "PROFILE_", "DEEP_"))
           and k not in ("LD_PRELOAD", "LD_LIBRARY_PATH")}
    env.update({
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": ",".join(g["uuid"] for g in selected),
        "LD_LIBRARY_PATH": f"{lib}:{cuda / 'lib64'}", "BENCH_GPU_LAYERS": ",".join(map(str, layers)),
        "GGML_CUDA_VOLTA_FORCE_MMQ": "moe", "GGML_CUDA_MOE_STREAM": "0",
        "GGML_CUDA_PREFETCH_WEIGHTS": "0", "GGML_OP_OFFLOAD_MIN_BATCH": "1", "GGML_CUDA_LF_ACCUM_FUSE": "1",
        "QWEN4EXP_QSA_PP_BLOCK_TOPK": "0", "QWEN4EXP_QSA_PP_GATHER": "0", "LLAMA_MOE_LAYER_FIRST_AUTO": "0",
        "BENCH_MODES": "stagger", "BENCH_SAMPLES": "8", "BENCH_QUALITY": "0", "DIAG_ALL_OUTPUTS": "0",
        "BENCH_CONTINUATION": "8", "BENCH_EXPERT_ROWS": "1024", "BENCH_DEVICE_MIB": str(args.workspace_mib),
        "BENCH_CONTEXT": str(max(131072 if prefix else 32768, prefix + n + 512)),
        "BENCH_SEQUENCE": ",".join(str(n) for _ in variants), "BENCH_SPARSE_SEQUENCE": ",".join(map(str, variants)),
        "BENCH_PREPARATION_SEQUENCE": ",".join("3" for _ in variants),
        "BENCH_ATTENTION_SEQUENCE": ",".join("3" for _ in variants),
        "BENCH_SUFFIX_SEQUENCE": ",".join(str(args.suffix.resolve()) for _ in variants),
        "BENCH_RECORD_NUMERIC_DIFFERENCE": "1", "LLAMA_MOE_LAYER_FIRST_RESIDENT_GROUP": "16",
        "LLAMA_MOE_LAYER_FIRST_PLAN_SHARED_SCRATCH": "0", "LLAMA_MOE_LAYER_FIRST_KV_RING": "0",
        "LLAMA_MOE_LAYER_FIRST_ASYNC_INPUTS": "0", "LLAMA_MOE_LAYER_FIRST_OUTPUT_ALIAS": "0",
        "LLAMA_MOE_LAYER_FIRST_ADAPTIVE_MAX_LAYERS": "2", "LLAMA_MOE_PROFILE": "0",
    })
    for i, base in enumerate(bases):
        env[f"LLAMA_MOE_LAYER_FIRST_BASE_LAYERS_{i}"] = str(base)
    if prefix:
        env.update({"BENCH_PREFIX_STATE": str(args.prefix_state.resolve()), "BENCH_TOKENS": str(args.prefix_tokens.resolve())})
    command = [str(binary), str(model), str(n), "1024", str(prefix), str(len(variants) - 1), str(out / "run")]
    manifest = {
        "command": command, "preset": args.preset, "devices": selected, "layers": layers, "base": bases,
        "environment": {k: env[k] for k in env if k not in os.environ or env[k] != os.environ[k]},
        "runtime_sha256": {f: digest(lib / f) for f in ("libllama.so.0", "libggml-cuda.so.0")},
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT)),
        "harness_sha256": digest(HERE / "bench-prefill.cpp"),
        "suffix_sha256": digest(args.suffix), "prefix_length": prefix, "tokens": n,
        "scope": "Prompt-processing time; eight sampled prompt score rows and eight fixed input continuation rows. Not an upstream comparison or broad quality test.",
    }
    (out / "command.json").write_text(json.dumps(manifest, indent=2) + "\n")
    started = time.monotonic()
    with (out / "run.log").open("w") as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    (out / "exit.json").write_text(json.dumps({"returncode": result.returncode, "wall_seconds": time.monotonic() - started}, indent=2) + "\n")
    if result.returncode:
        print("\n".join((out / "run.log").read_text().splitlines()[-35:]), file=sys.stderr)
        raise RuntimeError(f"Benchmark failed; see {out / 'run.log'}")
    rows = [json.loads(line) for line in (out / "run.jsonl").read_text().splitlines()]
    if len(rows) != len(variants):
        raise RuntimeError("Missing benchmark rounds")
    summary = {"prefix": prefix, "tokens": n, "settings": {}}
    for setting in (0, 1):
        group = [r for r in rows if r["sparse"] == setting]
        signature = None
        for r in group:
            stem = out / f"run-stagger-{r['fixture']}-n{n}-round{r['round']}"
            current = [digest(Path(str(stem) + ext)) for ext in (".f32", "-next.f32")]
            if signature is not None and current != signature:
                raise RuntimeError("Repeated scores differ within one attention setting")
            signature = current
        timings = [r["ms"] for r in group[1:]]
        median = statistics.median(timings)
        summary["settings"][str(setting)] = {"median_ms": median, "input_tokens_per_second": 1000 * n / median,
            "warm_samples_ms": timings, "repeats_bitwise": True}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print("Saved:", out)


if __name__ == "__main__":
    try:
        main()
    except (OSError, subprocess.SubprocessError, RuntimeError) as error:
        print(f"prefill benchmark: {error}", file=sys.stderr)
        sys.exit(1)
