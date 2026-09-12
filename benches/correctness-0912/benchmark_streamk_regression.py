#!/usr/bin/env python3
"""Matched normal-build ABBA, with explicit launch settings and cached-token checks."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import statistics
import sys
from validate_streamk_e2e import Server, ROOT, RTX, V100, MODEL, saved_tokens

ORNITH_SEQ = Path('/workspace/oai-qwen38-pp-lab/results/ornith15-seq101k.json')
QWEN_SEQ = Path('/workspace/oai-qwen38-pp-lab/results/cache100k-sharedstate-ab/seq.json')
QWEN_MODEL = '/models/Qwen3.8-27B/Qwen3.8-27B-UD-Q5_K_XL.gguf'
PROFILES = {
    'ornith-dual': dict(model=MODEL, seq=ORNITH_SEQ, state=Path('/models/.bench-ornith-sync'),
        filename='ornith100k.bin', prefix=100000, ctx=131072, batch=2048, ubatch=512,
        cuda=f'{V100},{RTX}', placement=['--split-mode','layer','--device','CUDA1,CUDA0','--tensor-split','14,35']),
    'ornith-rtx': dict(model='/models/Ornith-1.5-35B-A3B/Ornith-1.5-35B-A3B-AD-Q5_K-Q4_K.gguf',
        seq=ORNITH_SEQ, state=Path('/models/.bench-ornith2080'), filename='ornith65536-q5q4-force-mmq.bin',
        prefix=65536, ctx=67584, batch=4096, ubatch=512, cuda=RTX, placement=['--split-mode','none']),
    'qwen-rtx': dict(model=QWEN_MODEL, seq=QWEN_SEQ, state=Path('/models/.bench-qwen2080-sync'),
        filename='cache65536.bin', prefix=65536, ctx=67584, batch=4096, ubatch=2048,
        cuda=RTX, placement=['--split-mode','none']),
    'qwen-v100': dict(model=QWEN_MODEL, seq=QWEN_SEQ,
        state=Path('/workspace/oai-qwen38-pp-lab/results/sync-master-20260828/cache-v3-regenerated'),
        filename='cache100k.bin', prefix=100000, ctx=131072, batch=4096, ubatch=4096,
        cuda=V100, placement=['--split-mode','none']),
}


def arm(binary: Path, label: str, port: int, profile: dict, reps: int) -> dict:
    cfg = profile
    tokens = json.loads(cfg['seq'].read_text())[:cfg['prefix'] + 1000]
    if saved_tokens(cfg['state'] / cfg['filename']) != tokens[:cfg['prefix']]:
        raise ValueError('Prompt and saved prefix differ; refusing an invalid cached benchmark')
    s = Server(binary, label, port, cfg['state'])
    args = [str(s.binary), '--model', cfg['model'], '--host', '127.0.0.1', '--port', str(port),
            '--ctx-size', str(cfg['ctx']), '--parallel', '1', '--fit', 'off', '--gpu-layers', 'all',
            '--flash-attn', 'on', '--batch-size', str(cfg['batch']), '--ubatch-size', str(cfg['ubatch']),
            '--cache-type-k','q8_0','--cache-type-v','q8_0','--cache-ram','0','--ctx-checkpoints','0',
            '--slot-save-path', str(cfg['state']), '--no-warmup', '--perf', '--skip-chat-parsing',
            '--pipeline-copies','1', '--reasoning','off','--reasoning-format','none'] + cfg['placement']
    if cfg['model'] == QWEN_MODEL:
        args += ['--prefill-reuse','1024']
    s.args = args
    s.env['CUDA_VISIBLE_DEVICES'] = cfg['cuda']
    s.manifest['argv'] = args
    s.manifest['env']['CUDA_VISIBLE_DEVICES'] = cfg['cuda']
    rows = []
    with s:
        for rep in range(reps + 1):
            s.restore(0, cfg['filename'], cfg['prefix'])
            row = s.generate(tokens, count=1, expected_cache=cfg['prefix'])
            rows.append({'rep': rep, 'warmup': rep == 0, **row})
            print(label, 'PP', rep, row['timings']['prompt_per_second'], flush=True)
        s.restore(0, cfg['filename'], cfg['prefix'])
        decode = s.generate(tokens, count=64, expected_cache=cfg['prefix'])
        print(label, 'TG64', decode['timings']['predicted_per_second'], 'hash', decode['sha256'][:12], flush=True)
    return {'label': label, 'pp': statistics.mean(x['timings']['prompt_per_second'] for x in rows[1:]),
            'prompt_ms': statistics.mean(x['timings']['prompt_ms'] for x in rows[1:]),
            'tg': decode['timings']['predicted_per_second'], 'decode': decode, 'rows': rows, 'manifest': s.manifest}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('profile', choices=PROFILES)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--fixed', type=Path, required=True)
    p.add_argument('--reps', type=int, default=3)
    p.add_argument('--build-mode', choices=['normal', 'force-mmq'], default='normal')
    p.add_argument('--port', type=int, default=41200)
    a = p.parse_args()
    results = []
    for i, (key, b) in enumerate([('baseline', a.baseline), ('fixed', a.fixed), ('fixed', a.fixed), ('baseline', a.baseline)]):
        label = f'bench-{a.profile}-{key}-{i}'
        row = arm(b, label, a.port + i, PROFILES[a.profile], a.reps)
        row['key'] = key
        results.append(row)
        (ROOT / f'bench-{a.profile}.progress.json').write_text(json.dumps(results, indent=2) + '\n')
    summary = {}
    for key in ('baseline','fixed'):
        chosen = [r for r in results if r['key'] == key]
        pp_samples = [x['timings']['prompt_per_second'] for r in chosen for x in r['rows'][1:]]
        summary[key] = {'pp': statistics.mean(pp_samples), 'pp_sd': statistics.stdev(pp_samples),
                        'tg': statistics.mean(r['tg'] for r in chosen),
                        'decode_hashes': sorted({r['decode']['sha256'] for r in chosen})}
    summary['pp_delta_pct'] = 100 * (summary['fixed']['pp'] / summary['baseline']['pp'] - 1)
    summary['tg_delta_pct'] = 100 * (summary['fixed']['tg'] / summary['baseline']['tg'] - 1)
    report = {'profile': a.profile, 'summary': summary, 'arms': results,
              'build_mode': a.build_mode, 'volta_moe_env_on_both_arms': True}
    (ROOT / f'bench-{a.profile}.json').write_text(json.dumps(report, indent=2) + '\n')
    print('SUMMARY', a.profile, json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
