#!/usr/bin/env python3
"""Reproducible, fail-fast correctness and paired operator timing probes."""
import argparse
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ORNITH = '/models/Ornith-1.5-35B-A3B/Ornith-1.5-35B-A3B-AD-Q5_K-Q4_K.gguf'
NEXT = '/models/Qwen3.8-Flash-Next-GGUF/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf'
GPUS = {'v100': 'GPU-1d64d3a4-4aea-87ec-9048-33a5217efd79', '2080ti': 'GPU-30acbf2b-a41a-8b88-e298-fe66c4138b29'}

def run(case, index, folder, gpu, records):
    model, name, tokens, topk, route, host, enabled, group, repeats = case
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPUS[gpu], GGML_CUDA_VOLTA_FORCE_MMQ='moe',
               GGML_CUDA_MOE_STREAM=str(enabled), GGML_CUDA_MOE_STREAM_GROUP=str(group), GGML_CUDA_MOE_STREAM_TRACE='1')
    command = [str(ROOT / 'build/bin/test-moe-stream'), model, name, str(tokens), str(topk), route, host, str(repeats)]
    result = subprocess.run(command, env=env, text=True, capture_output=True, timeout=120)
    (folder / f'{index:04}.stderr').write_text(result.stderr)
    (folder / f'{index:04}.stdout').write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f'probe {index} failed ({result.returncode}): {command}\n{result.stderr[-3000:]}\n{result.stdout}')
    record = json.loads(result.stdout.strip().splitlines()[-1])
    record.update(index=index, model=model, command=command)
    record['stream_activated'] = 'moe-stream: calls=' in result.stderr
    if bool(enabled) != record['stream_activated']:
        raise RuntimeError(f'probe {index}: requested streaming={enabled}, activated={record["stream_activated"]}')
    records.write(json.dumps(record) + '\n')
    records.flush()
    print(f'{index:03} {gpu} {name} N={tokens} {route} {host} stream={enabled} group={group}: '
          f'{record["host_ms"]:.3f} ms, max_abs={record["max_abs"]:.3g}, pass={record["pass"]}', flush=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', choices=GPUS, default='v100')
    p.add_argument('--suite', choices=['correctness', 'timing'], default='correctness')
    p.add_argument('--tag', required=True)
    args = p.parse_args()
    folder = HERE / args.tag
    folder.mkdir(exist_ok=False)
    cases = []
    tensors = [(ORNITH, 'blk.0.ffn_gate_exps.weight', 8),
               (ORNITH, 'blk.0.ffn_down_exps.weight', 8),
               (NEXT, 'blk.0.ffn_gate_exps.weight', 10),
               (NEXT, 'blk.0.ffn_down_exps.weight', 10),
               (NEXT, 'blk.2.ffn_gate_exps.weight', 10),
               (NEXT, 'blk.2.ffn_down_exps.weight', 10)]
    if args.suite == 'correctness':
        for model, name, topk in tensors:
            for tokens, route, host, group in [(64, 'sparse', 'pinned', 7), (512, 'dense', 'pinned', 16),
                                              (1024, 'skew', 'pinned', 32), (4096, 'dense', 'pinned', 16),
                                              (512, 'sparse', 'pageable', 7)]:
                cases.append((model, name, tokens, topk, route, host, 1, group, 3))
    else:
        for model, name, topk in tensors[:4]:
            for tokens in [512, 2048, 4096]:
                # ABBA, using medians of seven repetitions inside each process.
                for enabled in [0, 1, 1, 0]:
                    cases.append((model, name, tokens, topk, 'dense', 'pinned', enabled, 16, 7))
    with (folder / 'results.jsonl').open('w') as records:
        for i, case in enumerate(cases):
            run(case, i, folder, args.gpu, records)
    print(f'Completed {len(cases)} probes.', flush=True)
