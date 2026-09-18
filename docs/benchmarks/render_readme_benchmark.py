#!/usr/bin/env python3
"""Render dated upstream comparisons and the separate Flash-Next experiment."""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from xml.sax.saxutils import escape

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FINAL = ROOT / "benches/upstream-sync-0912/headline-final-summary.json"
EXPERIMENT = ROOT / "benches/moe-prefill-0916/results/graph.json"
UPSTREAM_RUNTIME = ROOT / "benches/moe-prefill-0916/results/upstream-runtime-0916"
TG_HERO = HERE / "qwen38-v100-tg-0918.json"
ORDER = [
    ("qwen-v100", "Qwen3.8 27B", "V100 32 GB", "100k cached + 1k input"),
    ("qwen-rtx", "Qwen3.8 27B", "RTX 2080 Ti 22 GB", "65k cached + 1k input"),
    ("qwen-dual", "Qwen3.8 27B", "V100 + RTX 2080 Ti", "100k cached + 1k input"),
    ("ornith-v100", "Ornith 1.5 35B-A3B", "V100 32 GB", "100k cached + 1k input"),
    ("ornith-rtx", "Ornith 1.5 35B-A3B", "RTX 2080 Ti 22 GB", "65k cached + 1k input"),
    ("ornith-dual", "Ornith 1.5 35B-A3B", "V100 + RTX 2080 Ti", "100k cached + 1k input"),
    ("gemma31", "Gemma 4 31B", "V100 32 GB", "100k depth + 1k input"),
    ("gemma26", "Gemma 4 26B-A4B", "V100 32 GB", "100k depth + 1k input"),
    ("muse-v100", "Muse Glimmer 30B", "V100 32 GB", "100k depth + 1k input"),
]
GRAY = "#8C959F"
BLUE = "#0969DA"
INK = "#24292F"
MUTED = "#57606A"
GRID = "#D8DEE4"
FONT = "-apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif"


def text(x, y, value, *, size=18, anchor="start", weight=400, fill=INK):
    return (f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-family="{FONT}" '
            f'font-size="{size}" font-weight="{weight}" fill="{fill}">{escape(str(value))}</text>')


def paired_chart(path, *, title, subtitle, legends, rows, step, footer):
    width, left, plot_width, top, row_height = 1200, 370, 630, 146, 78
    bottom = top + len(rows) * row_height
    height = bottom + 62
    ymax = math.ceil(max(max(r["a"], r["b"]) for r in rows) * 1.13 / step) * step
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
             f'<title id="title">{escape(title)}</title>',
             f'<desc id="desc">{escape(subtitle)} {escape(footer)}</desc>',
             f'<rect width="{width}" height="{height}" fill="#FFFFFF"/>',
             text(26, 38, title, size=29, weight=700), text(26, 66, subtitle, size=17, fill=MUTED)]
    for x, label, color in [(left, legends[0], GRAY), (left + 360, legends[1], BLUE)]:
        parts += [f'<rect x="{x}" y="88" width="18" height="18" rx="3" fill="{color}"/>',
                  text(x + 27, 103, label, size=17, fill=MUTED)]
    for tick in range(0, ymax + 1, step):
        x = left + tick / ymax * plot_width
        parts += [f'<line x1="{x:.1f}" y1="{top-9}" x2="{x:.1f}" y2="{bottom-8}" stroke="{GRID}"/>',
                  text(x, 133, tick, size=16, anchor="middle", fill=MUTED)]
    parts.append(text(1120, 132, "Speed gain", anchor="middle", size=16, fill=MUTED))
    for i, row in enumerate(rows):
        y = top + i * row_height
        parts += [text(26, y + 22, row["model"], size=20, weight=600),
                  text(26, y + 44, row["hardware"], size=17),
                  text(26, y + 64, row["workload"], size=15, fill=MUTED)]
        for key, dy, color in [("a", 7, GRAY), ("b", 35, BLUE)]:
            value = row[key]
            bar_width = value / ymax * plot_width
            parts += [f'<rect x="{left}" y="{y+dy}" width="{bar_width:.2f}" height="21" rx="3" fill="{color}"/>',
                      text(left + bar_width + 8, y + dy + 17, f"{value:.1f}", size=17, weight=600, fill=color if key == "b" else MUTED)]
        gain = (row["b"] / row["a"] - 1) * 100
        parts.append(text(1120, y + 38, f"{gain:+.1f}%", anchor="middle", size=21, weight=600))
        if row.get("first_token_reduction") is not None:
            parts.append(text(1120, y + 60, f"First token -{row['first_token_reduction']:.1f}%", anchor="middle", size=14, fill=MUTED))
    parts += [text(26, height - 34, "Tokens per second - higher is faster", size=17, weight=600),
              text(26, height - 10, footer, size=15, fill=MUTED), "</svg>"]
    path.write_text("\n".join(parts) + "\n")


def main():
    data = json.loads(FINAL.read_text())
    for metric, filename, title, subtitle in [
        ("pp", "long-context-prompt-processing.svg", "Reading new input after a long conversation", "Published comparison with upstream llama.cpp, 12 September 2026"),
        ("tg", "token-generation-throughput.svg", "Writing the answer: generation speed", "128 output tokens after the long-context input, 12 September 2026"),
    ]:
        rows = [{"model": model, "hardware": hardware, "workload": workload,
                 "a": data[key]["upstream"][metric], "b": data[key]["current"][metric],
                 "first_token_reduction": data[key].get("ttft_reduction_pct") if metric == "pp" else None}
                for key, model, hardware, workload in ORDER]
        if metric == "pp":
            upstream_rows = [json.loads(line) for line in (UPSTREAM_RUNTIME / "upstream-1000.jsonl").read_text().splitlines()]
            optimized_rows = [json.loads(line) for line in (UPSTREAM_RUNTIME / "request-ordered.jsonl").read_text().splitlines()]
            upstream_ms = statistics.median(row["ms"] for row in upstream_rows[1:])
            optimized_ms = statistics.median(row["ms"] for row in optimized_rows if row["sparse"] == 1 and row["round"] > 6)
            flash_next = {
                "model": "Qwen3.8 Flash-Next",
                "hardware": "V100 + RTX 2080 Ti",
                "workload": "100k cached + 1k input · 16 Sep",
                "a": 1_000_000 / upstream_ms,
                "b": 1_000_000 / optimized_ms,
            }
            rows.insert(3, flash_next)
        paired_chart(HERE / filename, title=title,
                     subtitle="Matched comparisons with upstream llama.cpp, 12–16 September 2026" if metric == "pp" else subtitle,
                     legends=(("Matched upstream", "v100-optimized") if metric == "pp" else
                              ("Upstream 3057bb66", "v100-optimized, 12 Sep")), rows=rows,
                     step=400 if metric == "pp" else 20,
                     footer=("Q8 history | 12 Sep rows: upstream 3057bb66 · Flash-Next 16 Sep: upstream 83078fec0"
                             if metric == "pp" else
                             "Q8 history cache | Upstream 3057bb66 vs v100-optimized, 12 Sep | Exact settings: benches/upstream-sync-0912/REPORT.md"))
    if TG_HERO.exists():
        tg = json.loads(TG_HERO.read_text())
        rows = [
            {
                "model": row["label"],
                "hardware": "Qwen3.8 27B · V100 32 GB",
                "workload": "100k cached · TG512",
                "a": row["upstream_tg"],
                "b": row["optimized_tg"],
            }
            for row in tg["rows"]
        ]
        acceptance = next((row.get("acceptance") for row in tg["rows"] if row.get("acceptance") is not None), None)
        acceptance_text = f" · MTP3 acceptance {acceptance*100:.2f}% on both" if acceptance is not None else ""
        paired_chart(
            HERE / "qwen38-v100-token-generation.svg",
            title="Qwen3.8 27B on V100: token generation",
            subtitle="100k cached context · TG512 · Q8_0 KV cache",
            legends=(f"Upstream {tg['upstream_commit'][:9]}", "v100-optimized"),
            rows=rows,
            step=10,
            footer=f"MTP off vs MTP3{acceptance_text} · measured {tg['date']}",
        )

    if EXPERIMENT.exists():
        experiment = json.loads(EXPERIMENT.read_text())
        rows = [{"model": row["label"], "hardware": row["placement"], "workload": "100k cached + 1k input",
                 "a": 1_000_000 / row["control_ms"], "b": 1_000_000 / row["sparse_ms"]} for row in experiment["rows"]]
        paired_chart(HERE / "flash-next-prefill.svg", title="Qwen3.8-Flash-Next: optional sparse attention",
                     subtitle="Same request-wide MoE executor, sparse off/on. This is not an upstream comparison.",
                     legends=("Previous attention kernel", "Selected-entry attention"), rows=rows, step=100,
                     footer=experiment["caption"])
    print("Rendered README benchmark charts.")


if __name__ == "__main__":
    main()
