#!/usr/bin/env python3
"""Reproduce the 65k boundary and validate cold and restored Ornith serving.

Run under the Development Sandbox gpu:all resource lock. This script does not
modify existing model files or saved states. Both binaries must be fully built.
"""
from __future__ import annotations
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import struct
import subprocess
import threading
import time
import urllib.error
import urllib.request

LAB = Path('/models/.bench-ornith-mtp4')
FIXTURE_DIR = Path('/workspace/oai-qwen38-pp-lab/results/parallel-refined-0907')
FIXTURE = json.loads((FIXTURE_DIR / 'real-fixture.json').read_text())
MODEL = '/models/Ornith-1.5-35B-A3B/Ornith-1.5-35B-A3B-AD-Q6_K-Q5_K.gguf'
DRAFT = '/workspace/models/Ornith-1.5-35B-A3B/shisa-mtp/mtp-shisa-ornith15-all-q4.gguf'
V100 = 'GPU-1d64d3a4-4aea-87ec-9048-33a5217efd79'
RTX = 'GPU-30acbf2b-a41a-8b88-e298-fe66c4138b29'
ROOT = LAB / 'int8-streamk-validation/e2e'
ROOT.mkdir(parents=True, exist_ok=True)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def saved_tokens(path: Path) -> list[int]:
    with path.open('rb') as f:
        magic, version, n = struct.unpack('<III', f.read(12))
        if magic != 0x67677371 or version != 3 or n > 2000000:
            raise ValueError(f'Unsupported state header: {magic:x}, {version}, {n}')
        tokens = list(struct.unpack(f'<{n}i', f.read(4 * n)))
    if tokens and tokens[0] == -1:
        if tokens[1] != 1 or tokens[2] < 0:
            raise ValueError('Unsupported server-token wrapper')
        n = tokens[2]
        if len(tokens) != n + 4 or tokens[-1] != 0:
            raise ValueError('Expected a text-only saved state')
        return tokens[3:3 + n]
    return tokens


class Server:
    def __init__(self, binary: Path, label: str, port: int, state_dir: Path,
                 draft_kv: str | None = None, sharing: bool = True,
                 sanitizer: bool = False):
        self.binary = binary.resolve()
        self.label, self.port, self.sanitizer = label, port, sanitizer
        self.log_path = ROOT / f'{label}.log'
        self.draft_kv = draft_kv
        self.log = None
        self.p = None
        self.manifest = {}
        args = [str(self.binary), '-m', MODEL, '--host', '127.0.0.1', '--port', str(port),
                '--ctx-size', '1400000', '--parallel', '4', '--kv-unified',
                '--kv-unified-per-slot', '400000', '--slot-fork-prefix',
                '--cache-ram', '0', '--no-cache-idle-slots', '--slot-prompt-similarity', '0',
                '--override-kv', 'qwen35moe.context_length=int:400000',
                '--rope-scaling', 'yarn', '--rope-scale', '1.52587890625', '--yarn-orig-ctx', '262144',
                '--split-mode', 'layer', '--fit', 'off', '--gpu-layers', 'all',
                '--device', 'CUDA1,CUDA0', '--tensor-split', '14,35', '--flash-attn', 'on',
                '--batch-size', '512', '--ubatch-size', '128', '--pipeline-copies', '1',
                '--cache-type-k', 'q8_0', '--cache-type-v', 'q8_0',
                '--slot-save-path', str(state_dir), '--no-warmup', '--perf',
                '--reasoning', 'off', '--reasoning-format', 'none']
        if draft_kv:
            args += ['--spec-type', 'draft-mtp', '--spec-draft-model', DRAFT,
                     '--spec-draft-device', 'CUDA1,CUDA0' if sharing else 'CUDA1',
                     '--spec-draft-ngl', 'all', '--spec-draft-type-k', draft_kv,
                     '--spec-draft-type-v', draft_kv, '--spec-draft-ubatch', '64',
                     '--spec-draft-n-max', '1', '--spec-mtp-defer-prompt']
        env = os.environ.copy()
        for key in list(env):
            if key.startswith(('GGML_CUDA_', 'LLAMA_EXPERIMENT_', 'LLAMA_MTP_')) or key == 'CUDA_LAUNCH_BLOCKING':
                env.pop(key, None)
        env.update(CUDA_VISIBLE_DEVICES=f'{V100},{RTX}',
                   LD_LIBRARY_PATH=str(self.binary.parent),
                   GGML_CUDA_VOLTA_FORCE_MMQ='moe', GGML_CUDA_ALLREDUCE='internal',
                   GGML_CUDA_AR_COPY_THRESHOLD='131072')
        if draft_kv and sharing:
            env['LLAMA_MTP_SHARE_TARGET_IO'] = 'head'
        self.manifest = {'argv': args, 'cuda_library_sha256': sha(self.binary.parent / 'libggml-cuda.so.0'),
                         'env': {k: v for k, v in env.items() if k.startswith(('GGML_CUDA_', 'CUDA_VISIBLE_', 'LLAMA_MTP_'))}}
        self.args = (['compute-sanitizer', '--tool', 'memcheck', '--error-exitcode', '97',
                      '--report-api-errors', 'all'] + args) if sanitizer else args
        self.env = env

    def request(self, path: str, body: dict | None = None, timeout: float = 900) -> dict:
        data = None if body is None else json.dumps(body, separators=(',', ':')).encode()
        req = urllib.request.Request(f'http://127.0.0.1:{self.port}{path}', data=data,
                                     headers={} if data is None else {'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as f:
                return json.load(f)
        except urllib.error.HTTPError as e:
            raise RuntimeError(f'{path}: HTTP {e.code}: {e.read().decode(errors="replace")[:2000]}') from e

    def __enter__(self):
        self.log = self.log_path.open('wb')
        self.p = subprocess.Popen(self.args, env=self.env, stdout=self.log,
                                  stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                if self.p.poll() is not None:
                    raise RuntimeError('Server exited during startup')
                try:
                    if self.request('/health', timeout=.5).get('status') == 'ok':
                        return self
                except (OSError, ValueError, RuntimeError):
                    pass
                time.sleep(.15)
            raise TimeoutError('Server health timeout')
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.p is not None and self.p.poll() is None:
            if self.sanitizer:
                # Let memcheck finish and report ERROR SUMMARY: signal the application,
                # not the sanitizer wrapper or the entire process group.
                table = subprocess.check_output(['ps', '-eo', 'pid=,pgid=,comm='], text=True)
                victims = [int(fields[0]) for row in table.splitlines()
                           if len(fields := row.split()) == 3 and fields[1] == str(self.p.pid)
                           and fields[2] == 'llama-server']
                for pid in victims:
                    try:
                        os.kill(pid, signal.SIGINT)
                    except ProcessLookupError:
                        pass
                if not victims:
                    self.p.send_signal(signal.SIGINT)
            else:
                self.p.send_signal(signal.SIGINT)
            try:
                self.p.wait(35)
            except subprocess.TimeoutExpired:
                os.killpg(self.p.pid, signal.SIGKILL)
                self.p.wait()
        if self.log is not None:
            self.log.close()
        if self.p is not None:
            self.manifest['exit_code'] = self.p.returncode
            self.manifest['library_unchanged'] = self.manifest['cuda_library_sha256'] == sha(self.binary.parent / 'libggml-cuda.so.0')
            text = self.log_path.read_text(errors='replace')
            self.manifest['error_lines'] = [s for s in text.splitlines() if any(x in s for x in
                                           ('Invalid __', 'CUDA error:', 'ERROR SUMMARY:', 'GGML_ASSERT'))]
            self.manifest['memcheck_clean'] = 'ERROR SUMMARY: 0 errors' in text if self.sanitizer else None
            (ROOT / f'{self.label}.manifest.json').write_text(json.dumps(self.manifest, indent=2) + '\n')

    def __exit__(self, typ, value, tb):
        self.close()
        if typ is None:
            if self.p.returncode != 0 or not self.manifest.get('library_unchanged'):
                raise RuntimeError(f'{self.label}: bad exit/build changed: {self.manifest}')
            if self.sanitizer and not self.manifest.get('memcheck_clean'):
                raise RuntimeError(f'{self.label}: no clean memcheck summary')

    def restore(self, slot: int, name: str, expected: int):
        result = self.request(f'/slots/{slot}?action=restore', {'filename': name})
        if result.get('n_restored') != expected:
            raise AssertionError(result)

    def generate(self, tokens: list[int], slot: int = 0, count: int = 64,
                 expected_cache: int | None = None) -> dict:
        t0 = time.perf_counter()
        obj = {'prompt': tokens, 'id_slot': slot, 'cache_prompt': True, 'n_predict': count,
               'ignore_eos': True, 'temperature': 0, 'seed': 1234, 'return_tokens': True}
        if self.draft_kv:
            obj['speculative_n_max'] = 1
        result = self.request('/completion', obj)
        tm = result['timings']
        if expected_cache is not None and (tm['cache_n'] != expected_cache or tm['prompt_n'] != len(tokens) - expected_cache):
            raise AssertionError(f'Unexpected cache reuse: {tm}')
        output = result.get('tokens', [])
        if not output or tm.get('predicted_n') != count:
            raise AssertionError(f'Missing/short output: {tm}, {len(output)}')
        return {'wall_s': time.perf_counter() - t0, 'timings': tm, 'tokens': output,
                'sha256': hashlib.sha256(json.dumps(output, separators=(',', ':')).encode()).hexdigest()}


def boundary(binary: Path, label: str, port: int, sanitizer: bool) -> dict:
    state = LAB / 'cold-boundary'
    assert saved_tokens(state / 'ornith65024.bin') == FIXTURE['histories'][0][:65024]
    rows = []
    with Server(binary, label, port, state, sanitizer=sanitizer) as server:
        sizes = [512] if sanitizer else [128, 384, 512, 640, 1000, 512]
        for n in sizes:
            server.restore(0, 'ornith65024.bin', 65024)
            row = server.generate(FIXTURE['histories'][0][:65024 + n], count=1, expected_cache=65024)
            rows.append({'append': n, **row})
            print(label, 'append', n, row['timings'], flush=True)
    return {'rows': rows, 'manifest': server.manifest}


def cold(binary: Path, label: str, port: int, draft_kv: str | None) -> dict:
    with Server(binary, label, port, LAB / 'cold-boundary', draft_kv=draft_kv) as server:
        row = server.generate(FIXTURE['histories'][0][:100000], count=16, expected_cache=0)
        print(label, 'COLD100K', row['timings'], flush=True)
    return {'rows': [row], 'manifest': server.manifest}


def stress(binary: Path, label: str, port: int, draft_kv: str | None, rounds: int, sharing: bool) -> dict:
    state = LAB / f'draft-kv-states/{draft_kv}-{draft_kv}' if draft_kv and draft_kv != 'q8_0' else FIXTURE_DIR
    for i in range(4):
        assert saved_tokens(state / f'real100k-{i}.bin') == FIXTURE['histories'][i]
    rows = []
    with Server(binary, label, port, state, draft_kv=draft_kv, sharing=sharing) as server:
        for rep in range(rounds):
            for i in range(4):
                server.request(f'/slots/{i}?action=erase', {})
            for i in range(4):
                server.restore(i, f'real100k-{i}.bin', 100000)
            barrier = threading.Barrier(4)
            def one(i):
                barrier.wait(timeout=30)
                return server.generate(FIXTURE['prompts'][i], slot=i, count=128, expected_cache=100000)
            t0 = time.perf_counter()
            with cf.ThreadPoolExecutor(max_workers=4) as pool:
                outputs = list(pool.map(one, range(4)))
            row = {'round': rep, 'warmup': rep == 0, 'aggregate_tps': 512 / (time.perf_counter() - t0), 'outputs': outputs}
            rows.append(row)
            (ROOT / f'{label}.progress.json').write_text(json.dumps(rows, indent=2) + '\n')
            print(label, rep, 'aggregate_tps', row['aggregate_tps'], 'hashes', [x['sha256'][:12] for x in outputs], flush=True)
    keep = rows[1:] or rows
    return {'mean_aggregate_tps': statistics.mean(r['aggregate_tps'] for r in keep),
            'unique_hashes_per_slot': [len({r['outputs'][i]['sha256'] for r in rows}) for i in range(4)],
            'rows': rows, 'manifest': server.manifest}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['boundary', 'cold', 'stress'])
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--label', required=True)
    p.add_argument('--port', type=int, default=41050)
    p.add_argument('--draft-kv', choices=['q8_0', 'q4_0'])
    p.add_argument('--no-sharing', action='store_true')
    p.add_argument('--sanitizer', action='store_true')
    p.add_argument('--rounds', type=int, default=10)
    args = p.parse_args()
    out = {'mode': args.mode, 'label': args.label, 'binary': str(args.binary), 'passed': False}
    try:
        if args.mode == 'boundary':
            out.update(boundary(args.binary, args.label, args.port, args.sanitizer))
        elif args.mode == 'cold':
            out.update(cold(args.binary, args.label, args.port, args.draft_kv))
        else:
            out.update(stress(args.binary, args.label, args.port, args.draft_kv, args.rounds, not args.no_sharing))
        out['passed'] = True
    except Exception as e:
        out['error'] = repr(e)
        raise
    finally:
        (ROOT / f'{args.label}.json').write_text(json.dumps(out, indent=2) + '\n')
        print('RESULT', args.label, 'PASS' if out['passed'] else 'FAIL', flush=True)


if __name__ == '__main__':
    main()
