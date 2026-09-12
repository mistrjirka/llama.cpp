#!/usr/bin/env python3
"""Render the README's user-facing upstream-vs-v100-optimized benchmark charts.

Both charts are backed by the fresh September 12, 2026 benchmark summaries:
- benches/upstream-sync-0912/headline-final-summary.json

The prompt-processing chart compares the complete current fork against current
upstream; quant-isolation and component microbenchmarks deliberately do not
appear in the README hero chart.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from xml.sax.saxutils import escape

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PP_OUT = HERE / "long-context-prompt-processing.svg"
TG_OUT = HERE / "token-generation-throughput.svg"
FINAL = ROOT / "benches/upstream-sync-0912/headline-final-summary.json"

ORDER = [
    ("qwen-v100",  ("Qwen3.8 27B", "V100 32 GB", "100k cached + 1k")),
    ("qwen-rtx",   ("Qwen3.8 27B", "RTX 2080 Ti 22 GB", "65k cached + 1k")),
    ("qwen-dual",  ("Qwen3.8 27B", "V100 32 GB + RTX 2080 Ti", "100k cached + 1k")),
    ("ornith-v100",("Ornith 1.5 35B-A3B", "V100 32 GB", "100k cached + 1k")),
    ("ornith-rtx", ("Ornith 1.5 35B-A3B", "RTX 2080 Ti 22 GB", "65k cached + 1k")),
    ("ornith-dual",("Ornith 1.5 35B-A3B", "V100 32 GB + RTX 2080 Ti", "100k cached + 1k")),
    ("gemma31",    ("Gemma 4 31B", "V100 32 GB", "100k depth + 1k")),
    ("gemma26",    ("Gemma 4 26B-A4B", "V100 32 GB", "100k depth + 1k")),
    ("muse-v100",  ("Muse Glimmer 30B", "V100 32 GB", "100k depth + 1k")),
]

UPSTREAM_COLOR = "#8C959F"
FORK_COLOR = "#0969DA"
TEXT_COLOR = "#24292F"
MUTED = "#57606A"
GRID = "#D8DEE4"
PILL_FILL = "#F6F8FA"
PILL_STROKE = "#D0D7DE"
FONT = "-apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif"


def text(x, y, value, *, size=18, weight=400, anchor="middle", fill=TEXT_COLOR):
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="{FONT}" font-size="{size}" font-weight="{weight}" fill="{fill}">'
        f'{escape(str(value))}</text>'
    )


def load_rows():
    final = json.loads(FINAL.read_text())
    rows = []
    for key, lines in ORDER:
        src = final[key]
        upstream, current = src["upstream"], src["current"]
        rows.append({
            "key": key,
            "lines": lines,
            "upstream_pp": upstream["pp"],
            "current_pp": current["pp"],
            "pp_gain": src["pp_gain_pct"],
            "upstream_tg": upstream["tg"],
            "current_tg": current["tg"],
            "tg_gain": src["tg_gain_pct"],
            "ttft_reduction": src.get("ttft_reduction_pct"),
        })
    return rows


def nice_ymax(values, step):
    return max(step, math.ceil(max(values) * 1.10 / step) * step)


def render_pp(rows):
    W, H = 2200, 820
    left, right, top, bottom = 90, 45, 128, 170
    plot_w, plot_h = W - left - right, H - top - bottom
    ymax = nice_ymax([r[k] for r in rows for k in ("upstream_pp", "current_pp")], 400)
    baseline_y = top + plot_h
    y = lambda v: baseline_y - (v / ymax) * plot_h
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="title desc">',
        '<title id="title">Long-context prompt processing: current upstream versus v100-optimized</title>',
        '<desc id="desc">Grouped bars comparing prompt-processing throughput for current upstream llama.cpp and v100-optimized across Qwen, Ornith, Gemma, and Muse Glimmer workloads.</desc>',
        f'<rect width="{W}" height="{H}" rx="12" fill="#FFFFFF"/>',
        text(left, 42, "Long-context prompt processing", size=30, weight=700, anchor="start"),
        text(left, 72, "Current upstream vs v100-optimized · prompt processing throughput (tok/s)", size=17, anchor="start", fill=MUTED),
    ]
    lx = W - 360
    parts += [
        f'<rect x="{lx}" y="37" width="20" height="20" rx="4" fill="{UPSTREAM_COLOR}"/>',
        text(lx + 30, 53, "Upstream", size=16, anchor="start", fill=MUTED),
        f'<rect x="{lx+135}" y="37" width="20" height="20" rx="4" fill="{FORK_COLOR}"/>',
        text(lx + 165, 53, "v100-optimized", size=16, anchor="start", fill=MUTED),
    ]
    for tick in range(0, int(ymax) + 1, 400):
        yy = y(tick)
        parts.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{W-right}" y2="{yy:.1f}" stroke="{GRID}" stroke-width="1"/>')
        parts.append(text(left - 14, yy + 6, tick, size=14, anchor="end", fill=MUTED))
    parts.append(text(24, top + plot_h/2, "tok/s", size=14, fill=MUTED).replace('<text ', f'<text transform="rotate(-90 24 {top + plot_h/2:.1f})" ', 1))
    group_w = plot_w / len(rows); bar_w = 52; gap = 12
    for i, r in enumerate(rows):
        cx = left + group_w * (i + .5); xu = cx - gap/2 - bar_w; xf = cx + gap/2
        yu, yf = y(r["upstream_pp"]), y(r["current_pp"])
        parts += [
            f'<rect x="{xu:.1f}" y="{yu:.1f}" width="{bar_w}" height="{baseline_y-yu:.1f}" rx="5" fill="{UPSTREAM_COLOR}"/>',
            f'<rect x="{xf:.1f}" y="{yf:.1f}" width="{bar_w}" height="{baseline_y-yf:.1f}" rx="5" fill="{FORK_COLOR}"/>',
            text(xu+bar_w/2, yu-9, f'{r["upstream_pp"]:.1f}', size=14, weight=600, fill=MUTED),
            text(xf+bar_w/2, yf-9, f'{r["current_pp"]:.1f}', size=14, weight=700, fill=FORK_COLOR),
        ]
        pw, ph = 190, 52; py = max(92, min(yu, yf) - 79); px = cx - pw/2
        parts += [f'<rect x="{px:.1f}" y="{py:.1f}" width="{pw}" height="{ph}" rx="10" fill="{PILL_FILL}" stroke="{PILL_STROKE}"/>',
                  text(cx,py+21,f'{r["pp_gain"]:+.1f}% PP',size=16,weight=700)]
        if r["ttft_reduction"] is None:
            parts.append(text(cx,py+41,'depth-mode llama-bench',size=13,weight=600,fill=MUTED))
        else:
            parts.append(text(cx,py+41,f'{r["ttft_reduction"]:.1f}% lower TTFT',size=13,weight=600,fill=MUTED))
        ly=baseline_y+30
        for j,line in enumerate(r["lines"]):
            parts.append(text(cx,ly+22*j,line,size=15 if j else 16,weight=650 if j==0 else 500,fill=TEXT_COLOR if j<2 else MUTED))
    parts.append(f'<line x1="{left}" y1="{baseline_y}" x2="{W-right}" y2="{baseline_y}" stroke="#8C959F" stroke-width="1.2"/>')
    parts.append(text(W-right,H-20,"Q8 K/V · FlashAttention · current upstream 3057bb66 · 2026-09-12",size=13,anchor="end",fill=MUTED))
    parts.append('</svg>')
    PP_OUT.write_text('\n'.join(parts)+'\n')


def render_tg(rows):
    W, H = 2200, 780
    left, right, top, bottom = 90, 45, 128, 170
    plot_w, plot_h = W-left-right, H-top-bottom
    ymax = nice_ymax([r[k] for r in rows for k in ("upstream_tg", "current_tg")], 10)
    baseline_y = top + plot_h
    y = lambda v: baseline_y - (v / ymax) * plot_h
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="title desc">',
        '<title id="title">Token generation throughput: current upstream versus v100-optimized</title>',
        '<desc id="desc">Grouped bars comparing 128-token generation throughput for current upstream llama.cpp and v100-optimized after the same long-context setup used in the prompt-processing benchmark.</desc>',
        f'<rect width="{W}" height="{H}" rx="12" fill="#FFFFFF"/>',
        text(left,42,"Token generation throughput",size=30,weight=700,anchor="start"),
        text(left,72,"Current upstream vs v100-optimized · 128 generated tokens (tok/s)",size=17,anchor="start",fill=MUTED),
    ]
    lx=W-360
    parts += [f'<rect x="{lx}" y="37" width="20" height="20" rx="4" fill="{UPSTREAM_COLOR}"/>',text(lx+30,53,"Upstream",size=16,anchor="start",fill=MUTED),f'<rect x="{lx+135}" y="37" width="20" height="20" rx="4" fill="{FORK_COLOR}"/>',text(lx+165,53,"v100-optimized",size=16,anchor="start",fill=MUTED)]
    for tick in range(0,int(ymax)+1,10):
        yy=y(tick); parts.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{W-right}" y2="{yy:.1f}" stroke="{GRID}" stroke-width="1"/>'); parts.append(text(left-14,yy+6,tick,size=14,anchor="end",fill=MUTED))
    parts.append(text(24,top+plot_h/2,"tok/s",size=14,fill=MUTED).replace('<text ',f'<text transform="rotate(-90 24 {top+plot_h/2:.1f})" ',1))
    group_w=plot_w/len(rows); bar_w=52; gap=12
    for i,r in enumerate(rows):
        cx=left+group_w*(i+.5); xu=cx-gap/2-bar_w; xf=cx+gap/2; yu=y(r['upstream_tg']); yf=y(r['current_tg'])
        parts += [f'<rect x="{xu:.1f}" y="{yu:.1f}" width="{bar_w}" height="{baseline_y-yu:.1f}" rx="5" fill="{UPSTREAM_COLOR}"/>',f'<rect x="{xf:.1f}" y="{yf:.1f}" width="{bar_w}" height="{baseline_y-yf:.1f}" rx="5" fill="{FORK_COLOR}"/>',text(xu+bar_w/2,yu-9,f'{r["upstream_tg"]:.1f}',size=14,weight=600,fill=MUTED),text(xf+bar_w/2,yf-9,f'{r["current_tg"]:.1f}',size=14,weight=700,fill=FORK_COLOR)]
        pw,ph=170,32; py=max(92,min(yu,yf)-55); px=cx-pw/2
        parts += [f'<rect x="{px:.1f}" y="{py:.1f}" width="{pw}" height="{ph}" rx="10" fill="{PILL_FILL}" stroke="{PILL_STROKE}"/>',text(cx,py+22,f'{r["tg_gain"]:+.1f}% TG',size=15,weight=700)]
        ly=baseline_y+30
        for j,line in enumerate(r['lines']): parts.append(text(cx,ly+22*j,line,size=15 if j else 16,weight=650 if j==0 else 500,fill=TEXT_COLOR if j<2 else MUTED))
    parts.append(f'<line x1="{left}" y1="{baseline_y}" x2="{W-right}" y2="{baseline_y}" stroke="#8C959F" stroke-width="1.2"/>')
    parts.append(text(W-right,H-20,"128 generated tokens · same long-context setup as PP chart · current upstream 3057bb66 · 2026-09-12",size=13,anchor="end",fill=MUTED))
    parts.append('</svg>')
    TG_OUT.write_text('\n'.join(parts)+'\n')


if __name__ == '__main__':
    rows=load_rows(); render_pp(rows); render_tg(rows); print(PP_OUT); print(TG_OUT)
