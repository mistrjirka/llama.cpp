#!/usr/bin/env python3
"""Reverse the initial ABBA order and retain ten PP samples per engine."""
from pathlib import Path
import json
import statistics
from benchmark_streamk_regression import arm, PROFILES, ROOT

AREA = Path('/models/.bench-ornith-mtp4')
BINS = {key: AREA / f'int8-streamk-{key}-src/build-matched-force/bin/llama-server'
        for key in ('baseline', 'fixed')}
rows = []
for i, key in enumerate(('fixed', 'baseline', 'baseline', 'fixed')):
    result = arm(BINS[key], f'confirm-ornith-rtx-{key}-{i}', 41300 + i, PROFILES['ornith-rtx'], 5)
    result['key'] = key
    rows.append(result)
    (ROOT / 'confirm-ornith-rtx.progress.json').write_text(json.dumps(rows, indent=2) + '\n')
summary = {}
for key in ('baseline', 'fixed'):
    selected = [r for r in rows if r['key'] == key]
    pp = [x['timings']['prompt_per_second'] for r in selected for x in r['rows'][1:]]
    summary[key] = {'pp': statistics.mean(pp), 'pp_sd': statistics.stdev(pp),
                    'tg': statistics.mean(r['tg'] for r in selected), 'pp_samples': len(pp)}
summary['pp_delta_pct'] = 100 * (summary['fixed']['pp'] / summary['baseline']['pp'] - 1)
summary['tg_delta_pct'] = 100 * (summary['fixed']['tg'] / summary['baseline']['tg'] - 1)
report = {'order': 'BAAB', 'profile': 'ornith-rtx', 'build_mode': 'force-mmq', 'summary': summary, 'arms': rows}
(ROOT / 'confirm-ornith-rtx.json').write_text(json.dumps(report, indent=2) + '\n')
print('CONFIRMATION', json.dumps(summary), flush=True)
