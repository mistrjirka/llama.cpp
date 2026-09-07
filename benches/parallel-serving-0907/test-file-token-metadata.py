#!/usr/bin/env python3
"""Regression checks for token-only snapshot inspection (no model/GPU needed)."""
import argparse
import ctypes as C
import json
from pathlib import Path
import re
import struct
import tempfile

p = argparse.ArgumentParser()
p.add_argument('--library', default='build-sm70-75/bin/libllama.so')
p.add_argument('--output')
a = p.parse_args()
header = Path('include/llama.h').read_text()
def define(name):
    value = re.search(r'^#define\s+' + name + r'\s+(\S+)', header, re.M)[1]
    return int(value.rstrip("uUlL"), 0) if value[0].isdigit() else define(value)
magic, version = define('LLAMA_STATE_SEQ_MAGIC'), define('LLAMA_STATE_SEQ_VERSION')
lib = C.CDLL(str(Path(a.library).resolve()))
f = lib.llama_state_seq_load_file_tokens
f.argtypes = [C.c_char_p, C.POINTER(C.c_int32), C.c_size_t, C.POINTER(C.c_size_t)]
f.restype = C.c_size_t
passed = []
with tempfile.TemporaryDirectory() as td:
    path = Path(td) / 'meta.bin'
    raw = str(path).encode()
    tokens, count = (C.c_int32 * 3)(), C.c_size_t()
    path.write_bytes(struct.pack('<IIIiii', magic, version, 3, 2, 4, 6))
    assert f(raw, None, 0, C.byref(count)) == 12 and count.value == 3
    passed.append('count_only')
    assert f(raw, tokens, 3, C.byref(count)) == 24 and list(tokens) == [2, 4, 6]
    passed.append('token_only')
    assert f(raw, tokens, 2, C.byref(count)) == 0
    passed.append('capacity_guard')
    for label, payload in [
        ('truncated_header', b'bad'),
        ('wrong_magic', struct.pack('<III', 0, version, 0)),
        ('wrong_version', struct.pack('<III', magic, version + 1, 0)),
        ('truncated_tokens', struct.pack('<IIIi', magic, version, 3, 2)),
    ]:
        path.write_bytes(payload)
        assert f(raw, tokens, 3, C.byref(count)) == 0, label
        passed.append(label)
    assert f(None, tokens, 3, C.byref(count)) == 0
    passed.append('null_path')
    assert f(raw, tokens, 3, None) == 0
    passed.append('null_count')
    path.write_bytes(struct.pack('<III', magic, version, 0))
    assert f(raw, tokens, 3, C.byref(count)) == 12 and count.value == 0
    passed.append('empty_tokens')
report = {'passed': passed, 'count': len(passed)}
if a.output:
    Path(a.output).write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
