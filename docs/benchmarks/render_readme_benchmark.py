#!/usr/bin/env python3
"""Render the README long-context benchmark as a standalone SVG.

The values below are the headline measurements retained in
benches/upstream-sync-0911/REPORT.md.
"""
from pathlib import Path
from xml.sax.saxutils import escape

OUT = Path(__file__).with_name("long-context-prompt-processing.svg")

rows = [
    {
        "lines": ("Ornith 1.5 35B-A3B", "V100 32 GB", "100k cached + 1k"),
        "upstream": 539.11,
        "fork": 803.18,
        "gain": 48.98,
        "ttft": 32.06,
    },
    {
        "lines": ("Qwen3.8 27B", "V100 32 GB", "100k cached + 1k"),
        "upstream": 297.69,
        "fork": 429.66,
        "gain": 44.33,
        "ttft": 30.23,
    },
    {
        "lines": ("Gemma 4 31B", "V100 32 GB", "100k depth + 1k"),
        "upstream": 199.84,
        "fork": 273.19,
        "gain": 36.70,
        "ttft": None,
    },
    {
        "lines": ("Gemma 4 26B-A4B", "V100 32 GB", "100k depth + 1k"),
        "upstream": 689.52,
        "fork": 905.84,
        "gain": 31.37,
        "ttft": None,
    },
    {
        "lines": ("Qwen3.8 27B", "RTX 2080 Ti 22 GB", "65k cached + 1k"),
        "upstream": 382.30,
        "fork": 494.92,
        "gain": 29.46,
        "ttft": 22.53,
    },
    {
        "lines": ("Qwen3.8 27B", "V100 + RTX 2080 Ti", "100k cached + 1k"),
        "upstream": 408.08,
        "fork": 690.96,
        "gain": 69.32,
        "ttft": 39.93,
    },
]

W, H = 1500, 720
left, right, top, bottom = 90, 45, 128, 158
plot_w = W - left - right
plot_h = H - top - bottom
ymax = 1000.0
baseline_y = top + plot_h

upstream_color = "#8C959F"
fork_color = "#0969DA"
text_color = "#24292F"
muted = "#57606A"
grid = "#D8DEE4"
pill_fill = "#F6F8FA"
pill_stroke = "#D0D7DE"


def y(v: float) -> float:
    return baseline_y - (v / ymax) * plot_h


def svg_text(x, y0, text, *, size=18, weight=400, anchor="middle", fill=text_color):
    return (
        f'<text x="{x:.1f}" y="{y0:.1f}" text-anchor="{anchor}" '
        f'font-family="-apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}">{escape(str(text))}</text>'
    )

parts = [
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="title desc">',
    '<title id="title">Long-context prompt processing: upstream versus v100-optimized</title>',
    '<desc id="desc">Grouped vertical bars comparing upstream llama.cpp with v100-optimized across single-V100, single-RTX-2080-Ti, Gemma, Ornith, and mixed-GPU long-context workloads.</desc>',
    f'<rect width="{W}" height="{H}" rx="12" fill="#FFFFFF"/>',
    svg_text(left, 42, "Long-context prompt processing", size=30, weight=700, anchor="start"),
    svg_text(left, 72, "Prompt processing throughput (tok/s) · single-GPU/model results first, mixed-GPU last", size=17, anchor="start", fill=muted),
]

# Legend
legend_y = 51
legend_x = 1160
parts += [
    f'<rect x="{legend_x}" y="{legend_y-14}" width="20" height="20" rx="4" fill="{upstream_color}"/>',
    svg_text(legend_x + 30, legend_y + 2, "Upstream", size=16, anchor="start", fill=muted),
    f'<rect x="{legend_x+135}" y="{legend_y-14}" width="20" height="20" rx="4" fill="{fork_color}"/>',
    svg_text(legend_x + 165, legend_y + 2, "v100-optimized", size=16, anchor="start", fill=muted),
]

# Grid / y axis
for tick in range(0, 901, 200):
    yy = y(tick)
    parts.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{W-right}" y2="{yy:.1f}" stroke="{grid}" stroke-width="1"/>')
    parts.append(svg_text(left - 14, yy + 6, tick, size=14, anchor="end", fill=muted))
parts.append(svg_text(24, top + plot_h / 2, "tok/s", size=14, anchor="middle", fill=muted).replace('<text ', '<text transform="rotate(-90 24 %.1f)" ' % (top + plot_h / 2), 1))

# Bars
n = len(rows)
group_w = plot_w / n
bar_w = 54
bar_gap = 12

for i, row in enumerate(rows):
    cx = left + group_w * (i + 0.5)
    xu = cx - bar_gap / 2 - bar_w
    xf = cx + bar_gap / 2
    yu, yf = y(row["upstream"]), y(row["fork"])
    hu, hf = baseline_y - yu, baseline_y - yf

    parts.append(f'<rect x="{xu:.1f}" y="{yu:.1f}" width="{bar_w}" height="{hu:.1f}" rx="5" fill="{upstream_color}"/>')
    parts.append(f'<rect x="{xf:.1f}" y="{yf:.1f}" width="{bar_w}" height="{hf:.1f}" rx="5" fill="{fork_color}"/>')
    parts.append(svg_text(xu + bar_w/2, yu - 9, f'{row["upstream"]:.1f}', size=15, weight=600, fill=muted))
    parts.append(svg_text(xf + bar_w/2, yf - 9, f'{row["fork"]:.1f}', size=15, weight=700, fill=fork_color))

    # Improvement pill positioned above the taller bar.
    pill_w, pill_h = 184, 52
    pill_y = max(92, min(yu, yf) - 79)
    pill_x = cx - pill_w / 2
    parts.append(f'<rect x="{pill_x:.1f}" y="{pill_y:.1f}" width="{pill_w}" height="{pill_h}" rx="10" fill="{pill_fill}" stroke="{pill_stroke}"/>')
    parts.append(svg_text(cx, pill_y + 21, f'+{row["gain"]:.1f}% PP', size=16, weight=700))
    if row["ttft"] is not None:
        parts.append(svg_text(cx, pill_y + 41, f'{row["ttft"]:.1f}% lower TTFT', size=14, weight=600, fill=muted))
    else:
        parts.append(svg_text(cx, pill_y + 41, 'depth-mode llama-bench', size=13, weight=600, fill=muted))

    label_y = baseline_y + 30
    for j, line in enumerate(row["lines"]):
        parts.append(svg_text(cx, label_y + 22*j, line, size=15 if j else 16, weight=650 if j == 0 else 500, fill=text_color if j < 2 else muted))

# Baseline and footer
parts.append(f'<line x1="{left}" y1="{baseline_y}" x2="{W-right}" y2="{baseline_y}" stroke="#8C959F" stroke-width="1.2"/>')
parts.append(svg_text(W-right, H-20, "Q8 K/V · FlashAttention · 2026-09-11 measurements", size=13, anchor="end", fill=muted))
parts.append('</svg>')

OUT.write_text("\n".join(parts) + "\n")
print(OUT)
