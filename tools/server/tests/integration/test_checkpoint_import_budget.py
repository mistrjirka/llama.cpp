#!/usr/bin/env python3
"""Test checkpoint appendix bounds on a tiny CPU model. Does not use a GPU."""
import argparse
import json
import pathlib
import struct
import subprocess
import tempfile
import time
import urllib.error
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument('--server', required=True, type=pathlib.Path)
ap.add_argument('--model', required=True, type=pathlib.Path)
ap.add_argument('--port', default=2347, type=int)
ap.add_argument('--output', type=pathlib.Path)
args = ap.parse_args()
url = f'http://127.0.0.1:{args.port}'
MIB = 1024 * 1024
CKPT_HEADER = struct.pack('<II', 0x504b4353, 1)
DRAFT_EMPTY = struct.pack('<IIQQ', 0x54464453, 1, 0, 0)

def request(path, body=None):
    req = urllib.request.Request(url + path,
        data=None if body is None else json.dumps(body).encode(),
        headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.load(response)

def record(number, target, draft=b'', spec=b''):
    return struct.pack('<qii', number, 0, number - 1) + b''.join(
        struct.pack('<Q', len(blob)) + blob for blob in (target, draft, spec))

def appendix(records):
    return CKPT_HEADER + struct.pack('<I', len(records)) + b''.join(records) if records else b''

with tempfile.TemporaryDirectory(prefix='checkpoint-import-test-') as directory:
    work = pathlib.Path(directory)
    log_path = work / 'server.log'
    result = []
    for cap in (1, 0):
        command = [str(args.server), '-m', str(args.model), '--host', '127.0.0.1', '--port', str(args.port),
            '--device', 'none', '--gpu-layers', '0', '--ctx-size', '512', '--parallel', '1',
            '--threads', '1', '--threads-batch', '1', '--cache-ram', '16', '--ctx-checkpoints', '3',
            '--ctx-checkpoint-ram', str(cap), '--slot-save-path', str(work), '--no-warmup']
        with log_path.open('w') as log:
            process = subprocess.Popen(command, stdout=log, stderr=log)
            try:
                for attempt in range(100):
                    if process.poll() is not None:
                        raise RuntimeError(log_path.read_text())
                    try:
                        if request('/health').get('status') == 'ok': break
                    except (OSError, urllib.error.URLError): pass
                    time.sleep(0.1)
                else: raise TimeoutError('CPU server did not become healthy')
                request('/completion', {'prompt':'Once upon a time, a small bird', 'n_predict':4, 'temperature':0})
                request('/slots/0?action=save', {'filename':'base.bin'})
                base = (work / 'base.bin').read_bytes()
                assert CKPT_HEADER not in base, 'Use an attention-only tiny fixture with no checkpoint appendix'
                tiny = [record(i, bytes([i]) * 4096) for i in range(1, 7)]
                cases = [
                    ('count', appendix(tiny), tiny[-3:] if cap else []),
                    ('oversized', appendix([record(1, b'x' * (2 * MIB))]), []),
                    ('combined', appendix([record(1, b'x' * (700 * 1024), b'y' * (400 * 1024))]), []),
                    ('eviction', appendix([record(1, b'a' * (700 * 1024)), record(2, b'b' * (700 * 1024))]),
                        [record(2, b'b' * (700 * 1024))] if cap else []),
                    ('truncated', CKPT_HEADER + struct.pack('<IqiiQ', 1, 1, 0, 0, 1000), []),
                    ('huge_length', CKPT_HEADER + struct.pack('<IqiiQ', 1, 1, 0, 0, 2**64 - 1), []),
                ]
                for name, data, expected in cases:
                    file = work / (name + '.bin')
                    file.write_bytes(base + data + DRAFT_EMPTY)
                    restored = request('/slots/0?action=restore', {'filename':file.name})
                    request('/slots/0?action=save', {'filename':'roundtrip.bin'})
                    actual = (work / 'roundtrip.bin').read_bytes()
                    assert actual == base + appendix(expected), (name, cap, len(actual), len(base + appendix(expected)))
                    if name not in ('truncated', 'huge_length'):
                        assert restored['n_read'] == file.stat().st_size, (name, restored)
                    result.append({'case':name, 'cap_mib':cap, 'kept':len(expected), 'status':'pass'})
                    request('/slots/0?action=erase', {})
                assert request('/health')['status'] == 'ok'
            finally:
                process.terminate()
                try: process.wait(timeout=10)
                except subprocess.TimeoutExpired: process.kill(); process.wait()
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.with_suffix('.log').write_text(log_path.read_text())
    if args.output: args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    print('PASS: count/byte limits, zero budget, oversized and truncated input, exact appendix bytes, and following draft offset')
