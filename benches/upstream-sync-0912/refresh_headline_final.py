#!/usr/bin/env python3
"""Build the canonical user-facing benchmark summary from retained raw summaries.

V100 MoE rows use the fastest valid upstream policy available and the intended
fork policy:
- Ornith V100: upstream global FORCE_MMQ vs current selective MMQ=moe.
- Gemma 26B-A4B: upstream global FORCE_MMQ vs current selective MMQ=moe.
Other rows use the matched headline upstream/current measurements directly.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
server = json.loads((ROOT / "headline-server-summary.json").read_text())
gemma = json.loads((ROOT / "headline-gemma-summary.json").read_text())
ornith_fair = json.loads((ROOT / "mmq-fairness/ornith-v100.json").read_text())["summary"]
gemma26_fair = json.loads((ROOT / "mmq-fairness/gemma26.json").read_text())["summary"]

out = {}
for key, src in server.items():
    out[key] = {
        "upstream": src["upstream"],
        "current": src["current"],
        "pp_gain_pct": src["pp_gain_pct"],
        "tg_gain_pct": src["tg_gain_pct"],
        "ttft_reduction_pct": src["ttft_reduction_pct"],
        "policy": "matched headline server configuration",
    }

u = ornith_fair["upstream-force"]
c = ornith_fair["current-moe"]
out["ornith-v100"] = {
    "upstream": u,
    "current": c,
    "pp_gain_pct": 100.0 * (c["pp"] / u["pp"] - 1.0),
    "tg_gain_pct": 100.0 * (c["tg"] / u["tg"] - 1.0),
    "ttft_reduction_pct": 100.0 * (1.0 - c["ttft_wall_ms"] / u["ttft_wall_ms"]),
    "policy": "upstream global FORCE_MMQ; current selective GGML_CUDA_VOLTA_FORCE_MMQ=moe",
}

for key in ("gemma31", "muse-v100"):
    src = gemma[key]["summary"]
    out[key] = {
        "upstream": src["upstream"],
        "current": src["current"],
        "pp_gain_pct": src["pp_gain_pct"],
        "tg_gain_pct": src["tg_gain_pct"],
        "ttft_reduction_pct": None,
        "policy": "normal dispatch on both arms; llama-bench depth mode",
    }

u = gemma26_fair["upstream-force"]
c = gemma26_fair["current-moe"]
out["gemma26"] = {
    "upstream": u,
    "current": c,
    "pp_gain_pct": 100.0 * (c["pp"] / u["pp"] - 1.0),
    "tg_gain_pct": 100.0 * (c["tg"] / u["tg"] - 1.0),
    "ttft_reduction_pct": None,
    "policy": "upstream global FORCE_MMQ; current selective GGML_CUDA_VOLTA_FORCE_MMQ=moe; llama-bench depth mode",
}

(ROOT / "headline-final-summary.json").write_text(json.dumps(out, indent=2) + "\n")
print(ROOT / "headline-final-summary.json")
