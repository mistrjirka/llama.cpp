#!/usr/bin/env python3
"""Render the README benchmark overview as a standalone SVG.

The upper panel contains the retained long-context upstream comparisons.
The lower panel is the September 12, 2026 direct regression gate comparing
pre-quant v100-optimized (eae5d0ec) with the current quantized-conversion work.
"""
from pathlib import Path
from xml.sax.saxutils import escape
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "long-context-prompt-processing.svg"
QUANT_SUMMARY = ROOT / "benches/volta-kquant-0912/regression-vs-v100-base-summary.json"

long_rows = [
    {"lines": ("Qwen3.8 27B", "V100 32 GB", "100k cached + 1k"), "upstream": 297.69, "fork": 429.66, "gain": 44.33, "ttft": 30.23},
    {"lines": ("Qwen3.8 27B", "RTX 2080 Ti 22 GB", "65k cached + 1k"), "upstream": 382.30, "fork": 494.92, "gain": 29.46, "ttft": 22.53},
    {"lines": ("Qwen3.8 27B", "V100 32 GB + RTX 2080 Ti", "100k cached + 1k"), "upstream": 408.08, "fork": 690.96, "gain": 69.32, "ttft": 39.93},
    {"lines": ("Ornith 1.5 35B-A3B", "V100 32 GB", "100k cached + 1k"), "upstream": 539.11, "fork": 803.18, "gain": 48.98, "ttft": 32.06},
    {"lines": ("Ornith 1.5 35B-A3B", "RTX 2080 Ti 22 GB", "65k cached + 1k"), "upstream": 1332.91, "fork": 1505.73, "gain": 12.97, "ttft": 11.13},
    {"lines": ("Ornith 1.5 35B-A3B", "V100 32 GB + RTX 2080 Ti", "100k cached + 1k"), "upstream": 1247.39, "fork": 1458.85, "gain": 16.95, "ttft": 14.44},
    {"lines": ("Gemma 4 31B", "V100 32 GB", "100k depth + 1k"), "upstream": 199.84, "fork": 273.19, "gain": 36.70, "ttft": None},
    {"lines": ("Gemma 4 26B-A4B", "V100 32 GB", "100k depth + 1k"), "upstream": 689.52, "fork": 905.84, "gain": 31.37, "ttft": None},
]

quant_meta = [
    ("ornith-q2k", ("Ornith 1.5 9B", "Q2_K", "4k prompt")),
    ("ornith-q3k", ("Ornith 1.5 9B", "Q3_K_L", "4k prompt")),
    ("muse-q4k", ("Muse Glimmer 30B", "Q4_K_XL", "4k prompt")),
    ("qwen-q5q6", ("Qwen3.8 27B", "Q5_K + Q6_K", "4k prompt")),
    ("pxa-mxfp4", ("PXA Fusion4 35B", "MXFP4 dense path", "4k prompt")),
    ("ornith-pxq4hq-control", ("Ornith 1.5 9B", "PXQ4-HQ control", "4k prompt")),
]

summary = json.loads(QUANT_SUMMARY.read_text())
quant_rows = []
for key, lines in quant_meta:
    r = summary[key]
    quant_rows.append({
        "lines": lines,
        "base": r["base"]["mean_tok_s"],
        "fork": r["new"]["mean_tok_s"],
        "gain": r["change_pct"],
        "delta_ms": r["time_delta_ms"],
        "control": key.endswith("control"),
    })

W, H = 1900, 1290
left, right = 90, 45
upstream_color = "#8C959F"
fork_color = "#0969DA"
text_color = "#24292F"
muted = "#57606A"
grid = "#D8DEE4"
pill_fill = "#F6F8FA"
pill_stroke = "#D0D7DE"
panel_rule = "#AFB8C1"


def svg_text(x, y0, text, *, size=18, weight=400, anchor="middle", fill=text_color):
    return (
        f'<text x="{x:.1f}" y="{y0:.1f}" text-anchor="{anchor}" '
        f'font-family="-apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}">{escape(str(text))}</text>'
    )


def legend(parts, x, y0, left_label, right_label):
    parts.extend([
        f'<rect x="{x}" y="{y0-14}" width="20" height="20" rx="4" fill="{upstream_color}"/>',
        svg_text(x + 30, y0 + 2, left_label, size=16, anchor="start", fill=muted),
        f'<rect x="{x+190}" y="{y0-14}" width="20" height="20" rx="4" fill="{fork_color}"/>',
        svg_text(x + 220, y0 + 2, right_label, size=16, anchor="start", fill=muted),
    ])


def axis(parts, top, plot_h, ymax, ticks):
    baseline = top + plot_h
    def y(v):
        return baseline - (v / ymax) * plot_h
    for tick in ticks:
        yy = y(tick)
        parts.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{W-right}" y2="{yy:.1f}" stroke="{grid}" stroke-width="1"/>')
        parts.append(svg_text(left - 14, yy + 6, tick, size=14, anchor="end", fill=muted))
    parts.append(svg_text(24, top + plot_h / 2, "tok/s", size=14, fill=muted).replace(
        '<text ', f'<text transform="rotate(-90 24 {top + plot_h / 2:.1f})" ', 1))
    return y, baseline


parts = [
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="title desc">',
    '<title id="title">V100 and RTX prompt processing benchmarks</title>',
    '<desc id="desc">Two grouped-bar panels: retained long-context upstream versus v100-optimized results, followed by direct V100 pre-quant versus current quant-conversion regression measurements.</desc>',
    f'<rect width="{W}" height="{H}" rx="12" fill="#FFFFFF"/>',
]

# Panel 1: retained long-context headline results.
parts += [
    svg_text(left, 42, "Long-context prompt processing", size=30, weight=700, anchor="start"),
    svg_text(left, 72, "Retained upstream vs v100-optimized measurements · grouped by model family and hardware", size=17, anchor="start", fill=muted),
]
legend(parts, 1470, 51, "Upstream", "v100-optimized")
y1, base1 = axis(parts, 128, 492, 1600.0, range(0, 1601, 400))
plot_w = W - left - right
n = len(long_rows)
group_w = plot_w / n
bar_w, bar_gap = 54, 12
for i, row in enumerate(long_rows):
    cx = left + group_w * (i + 0.5)
    xu, xf = cx - bar_gap / 2 - bar_w, cx + bar_gap / 2
    yu, yf = y1(row["upstream"]), y1(row["fork"])
    parts += [
        f'<rect x="{xu:.1f}" y="{yu:.1f}" width="{bar_w}" height="{base1-yu:.1f}" rx="5" fill="{upstream_color}"/>',
        f'<rect x="{xf:.1f}" y="{yf:.1f}" width="{bar_w}" height="{base1-yf:.1f}" rx="5" fill="{fork_color}"/>',
        svg_text(xu + bar_w/2, yu - 9, f'{row["upstream"]:.1f}', size=15, weight=600, fill=muted),
        svg_text(xf + bar_w/2, yf - 9, f'{row["fork"]:.1f}', size=15, weight=700, fill=fork_color),
    ]
    pill_w, pill_h = 184, 52
    pill_y = max(92, min(yu, yf) - 79)
    pill_x = cx - pill_w / 2
    parts.append(f'<rect x="{pill_x:.1f}" y="{pill_y:.1f}" width="{pill_w}" height="{pill_h}" rx="10" fill="{pill_fill}" stroke="{pill_stroke}"/>')
    parts.append(svg_text(cx, pill_y + 21, f'+{row["gain"]:.1f}% PP', size=16, weight=700))
    sub = f'{row["ttft"]:.1f}% lower TTFT' if row["ttft"] is not None else 'depth-mode llama-bench'
    parts.append(svg_text(cx, pill_y + 41, sub, size=14 if row["ttft"] is not None else 13, weight=600, fill=muted))
    for j, line in enumerate(row["lines"]):
        parts.append(svg_text(cx, base1 + 30 + 22*j, line, size=15 if j else 16, weight=650 if j == 0 else 500, fill=text_color if j < 2 else muted))
parts.append(f'<line x1="{left}" y1="{base1}" x2="{W-right}" y2="{base1}" stroke="{upstream_color}" stroke-width="1.2"/>')
parts.append(svg_text(W-right, 760, "Q8 K/V · FlashAttention · retained 2026-09-11 long-context measurements", size=13, anchor="end", fill=muted))

# Divider.
parts.append(f'<line x1="{left}" y1="790" x2="{W-right}" y2="790" stroke="{panel_rule}" stroke-width="1.2"/>')

# Panel 2: direct regression gate for generic quant conversion work.
parts += [
    svg_text(left, 838, "Volta quantization regression gate", size=27, weight=700, anchor="start"),
    svg_text(left, 868, "Same V100 and 4k prompt · pre-quant eae5d0ec vs current branch · five samples/process, A-B-B-A", size=16, anchor="start", fill=muted),
]
legend(parts, 1390, 847, "Pre-quant V100", "Current")
y2, base2 = axis(parts, 910, 245, 3600.0, (0, 900, 1800, 2700, 3600))
n2 = len(quant_rows)
group_w2 = plot_w / n2
bar_w2, gap2 = 62, 14
for i, row in enumerate(quant_rows):
    cx = left + group_w2 * (i + 0.5)
    xb, xn = cx - gap2 / 2 - bar_w2, cx + gap2 / 2
    yb, yn = y2(row["base"]), y2(row["fork"])
    parts += [
        f'<rect x="{xb:.1f}" y="{yb:.1f}" width="{bar_w2}" height="{base2-yb:.1f}" rx="5" fill="{upstream_color}"/>',
        f'<rect x="{xn:.1f}" y="{yn:.1f}" width="{bar_w2}" height="{base2-yn:.1f}" rx="5" fill="{fork_color}"/>',
        svg_text(xb + bar_w2/2, yb - 8, f'{row["base"]:.0f}', size=14, weight=600, fill=muted),
        svg_text(xn + bar_w2/2, yn - 8, f'{row["fork"]:.0f}', size=14, weight=700, fill=fork_color),
    ]
    pill_w, pill_h = 168, 29
    pill_y = max(882, min(yb, yn) - 47)
    pill_x = cx - pill_w / 2
    parts.append(f'<rect x="{pill_x:.1f}" y="{pill_y:.1f}" width="{pill_w}" height="{pill_h}" rx="9" fill="{pill_fill}" stroke="{pill_stroke}"/>')
    if row["control"]:
        pill = f'{row["gain"]:+.2f}% control'
    else:
        pill = f'{row["gain"]:+.2f}%'
    parts.append(svg_text(cx, pill_y + 20, pill, size=15, weight=700))
    for j, line in enumerate(row["lines"]):
        parts.append(svg_text(cx, base2 + 27 + 21*j, line, size=15 if j == 0 else 14, weight=650 if j == 0 else 500, fill=text_color if j < 2 else muted))
parts.append(f'<line x1="{left}" y1="{base2}" x2="{W-right}" y2="{base2}" stroke="{upstream_color}" stroke-width="1.2"/>')
parts.append(svg_text(W-right, H-18, "Q8 K/V · FlashAttention · quant regression gate measured 2026-09-12", size=13, anchor="end", fill=muted))
parts.append('</svg>')

OUT.write_text("\n".join(parts) + "\n")
print(OUT)
