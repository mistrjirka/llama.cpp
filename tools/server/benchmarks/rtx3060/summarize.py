#!/usr/bin/env python3
"""Summarize recorded experiments without rerunning model inference."""
import argparse
import json
from pathlib import Path
import re
import statistics


def summarize(path):
    result = json.loads((path / 'results.json').read_text())
    records = result['records']
    generated = sum(max(0, r['timings'].get('predicted_n', 0)-1) for r in records)
    decode_s = sum(r['timings'].get('predicted_ms', 0)/1000 for r in records)
    out = {
        'variant': path.name,
        'requests': len(records),
        'generation_tok_s': generated/decode_s if decode_s else None,
        'wall_s': sum(r['wall_s'] for r in records),
        'prompt_s': sum(r['timings'].get('prompt_ms', 0)/1000 for r in records),
        'peak_VRAM_MiB': result['gpu']['peak_used_MiB'],
        'peak_PSS_MiB': result['peak_MiB']['Pss'],
        'peak_RSS_MiB': result.get('peak_RSS_MiB'),
        'RSS_peak_reset': result.get('RSS_peak_reset'),
        'start_PSS_MiB': result.get('start_MiB', {}).get('Pss'),
        'peak_anon_MiB': result['peak_MiB']['Pss_Anon'],
        'peak_swap_MiB': result['peak_MiB']['Swap'],
        'idle_PSS_MiB': result['idle_MiB']['Pss'],
        'cached_tokens': sum(r['timings'].get('cache_n', 0) for r in records),
        'drafted': sum(r['timings'].get('draft_n', 0) for r in records),
        'accepted': sum(r['timings'].get('draft_n_accepted', 0) for r in records),
        'output_mismatches': [r['case'] for r in records if r.get('same_as_first') is False],
        'answer_errors': [r['case'] for r in records if r.get('expected_match') is False],
        'by_case': {},
    }
    for name in dict.fromkeys(r['case'] for r in records):
        subset = [r for r in records if r['case'] == name]
        out['by_case'][name] = {
            'median_generation_tok_s': statistics.median(r['timings'].get('predicted_per_second', 0) for r in subset),
            'median_wall_s': statistics.median(r['wall_s'] for r in subset),
            'hashes': [r['content_sha256'] for r in subset],
        }
    text = (path / 'server.log').read_text(errors='replace')
    cache = [tuple(map(float, m)) for m in re.findall(r'prompt cache memory: (\d+) entries, ([\d.]+) MiB accounted, ([\d.]+) MiB logical, ([\d.]+) MiB unique payload, (\d+) blocks, ([\d.]+) MiB limit', text)]
    checkpoints = [tuple(map(float, m)) for m in re.findall(r'parked checkpoint memory: (\d+) references, (\d+) unique, ([\d.]+) MiB logical, ([\d.]+) MiB unique payload', text)]
    if cache:
        out['cache'] = {
            'peak_entries': max(c[0] for c in cache),
            'peak_accounted_MiB': max(c[1] for c in cache),
            'peak_logical_MiB': max(c[2] for c in cache),
            'peak_unique_block_MiB': max(c[3] for c in cache),
            'max_snapshot_saving_MiB': max(c[2]-c[1] for c in cache),
            'within_budget': all(c[1] <= c[5]+.001 for c in cache),
        }
    if checkpoints:
        out['checkpoints'] = {
            'peak_references': max(c[0] for c in checkpoints),
            'peak_unique': max(c[1] for c in checkpoints),
            'peak_logical_MiB': max(c[2] for c in checkpoints),
            'peak_unique_MiB': max(c[3] for c in checkpoints),
            'max_sharing_saving_MiB': max(c[2]-c[3] for c in checkpoints),
        }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('directory', type=Path)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    results = [summarize(f.parent) for f in sorted(args.directory.glob('*/results.json'))]
    print('| Variant | Generation tok/s | Total seconds | VRAM MiB | Peak PSS MiB |')
    print('|---|---:|---:|---:|---:|')
    for r in results:
        print(f"| {r['variant']} | {r['generation_tok_s']:.2f} | {r['wall_s']:.2f} | {r['peak_VRAM_MiB']:.0f} | {r['peak_PSS_MiB']:.1f} |")
    if args.output:
        args.output.write_text(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
