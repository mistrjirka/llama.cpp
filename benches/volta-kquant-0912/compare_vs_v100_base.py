#!/usr/bin/env python3
import json, os, subprocess, statistics
from pathlib import Path

GPU = "GPU-1d64d3a4-4aea-87ec-9048-33a5217efd79"
BASE = Path("/workspace/llama-muse-v100-baseline/build-v100/bin/llama-bench")
NEW  = Path("/workspace/muse-glimmer-v100/build-muse-sm70/bin/llama-bench")
OUT = Path("/workspace/volta-extra-quant-0912/regression-vs-v100-base")
OUT.mkdir(parents=True, exist_ok=True)

models = [
    ("ornith-q2k", "/models/ornith9b-normal-quants-0909/Ornith-1.5-9B-Q2_K.gguf"),
    ("ornith-q3k", "/models/ornith9b-normal-quants-0909/Ornith-1.5-9B-Q3_K_L.gguf"),
    ("muse-q4k", "/models/Muse-Glimmer-30B/Muse-Glimmer-30B-UD-Q4_K_XL.gguf"),
    ("qwen-q5q6", "/models/Qwen3.8-27B/Qwen3.8-27B-UD-Q5_K_XL.gguf"),
    ("pxa-mxfp4", "/models/PXA-Fusion4-35B-GGUF/PXA-Fusion4-35B-PXQ4.gguf"),
    ("ornith-pxq4hq-control", "/models/ornith9b-pxq-all/Ornith-1.5-9B-PXQ4HQ.gguf"),
]

common = ["-p", "4096", "-n", "0", "-b", "4096", "-ub", "1024",
          "-ctk", "q8_0", "-ctv", "q8_0", "-ngl", "999", "-fa", "on",
          "-dev", "CUDA0", "-r", "5", "-o", "json"]

def run(binary, model, tag, name):
    path = OUT / f"{name}-{tag}.json"
    err = OUT / f"{name}-{tag}.err"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = GPU
    # Current defaults are intentionally tested without rollback envs.
    for key in ["GGML_CUDA_VOLTA_MXFP4_PACKED", "GGML_CUDA_VOLTA_Q2K_GROUPED",
                "GGML_CUDA_VOLTA_Q3K_PACKED", "GGML_CUDA_VOLTA_Q4K_PACKED",
                "GGML_CUDA_VOLTA_Q5K_PACKED", "GGML_CUDA_VOLTA_Q6K_GROUPED"]:
        env.pop(key, None)
    with path.open("w") as fo, err.open("w") as fe:
        subprocess.run([str(binary), "-m", model, *common], env=env, stdout=fo, stderr=fe, check=True)
    return json.loads(path.read_text())[0]

summary = {}
for name, model in models:
    # A B B A gives some protection from clock/thermal drift.
    rows = {
        "base": [run(BASE, model, "A0", name)],
        "new":  [run(NEW,  model, "B0", name), run(NEW, model, "B1", name)],
    }
    rows["base"].append(run(BASE, model, "A1", name))
    item = {}
    for side in ["base", "new"]:
        ts = [x["avg_ts"] for x in rows[side]]
        ms = [x["avg_ns"] / 1e6 for x in rows[side]]
        item[side] = {
            "tok_s_runs": ts,
            "mean_tok_s": statistics.mean(ts),
            "mean_ms": statistics.mean(ms),
            "sd_between_runs": statistics.stdev(ts),
        }
    item["change_pct"] = 100.0 * (item["new"]["mean_tok_s"] / item["base"]["mean_tok_s"] - 1.0)
    item["time_delta_ms"] = item["new"]["mean_ms"] - item["base"]["mean_ms"]
    summary[name] = item
    print(name, f"{item['base']['mean_tok_s']:.3f} -> {item['new']['mean_tok_s']:.3f} tok/s  {item['change_pct']:+.3f}%  {item['time_delta_ms']:+.3f} ms", flush=True)

(OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
