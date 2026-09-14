#!/usr/bin/env python3
"""Read GGUF metadata without mapping/touching any weight data."""
import json
import struct
import sys
from pathlib import Path

FORMATS = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}

def read(path: Path):
    with path.open('rb') as f:
        def number(fmt):
            size = struct.calcsize('<' + fmt)
            data = f.read(size)
            if len(data) != size:
                raise ValueError('truncated GGUF')
            return struct.unpack('<' + fmt, data)[0]
        def string(keep=True):
            n = number('Q')
            if n > path.stat().st_size:
                raise ValueError('invalid GGUF string length')
            if keep:
                return f.read(n).decode('utf-8', errors='replace')
            f.seek(n, 1)
        def value(kind, keep):
            if kind == 8:
                return string(keep)
            if kind == 9:
                item_kind, count = number('I'), number('Q')
                if item_kind in FORMATS:
                    f.seek(count * struct.calcsize('<' + FORMATS[item_kind]), 1)
                else:
                    for _ in range(count):
                        value(item_kind, False)
                return None
            if kind not in FORMATS:
                raise ValueError(f'unknown GGUF type {kind}')
            return number(FORMATS[kind])
        if f.read(4) != b'GGUF':
            raise ValueError('not GGUF')
        version, tensors, pairs = number('I'), number('Q'), number('Q')
        metadata = {}
        for _ in range(pairs):
            key = string()
            keep = key == 'general.architecture' or any(s in key for s in ('expert_count', 'expert_used_count', 'ngram', 'feed_forward', 'block_count'))
            val = value(number('I'), keep)
            if keep:
                metadata[key] = val
        selected = []
        for _ in range(tensors):
            name, ndims = string(), number('I')
            shape = [number('Q') for _ in range(ndims)]
            kind, offset = number('I'), number('Q')
            if 'exps' in name and len(selected) < 9:
                selected.append({'name': name, 'shape': shape, 'ggml_type': kind, 'offset': offset})
        return {'path': str(path), 'version': version, 'metadata': metadata, 'sample_experts': selected}

if __name__ == '__main__':
    for arg in sys.argv[1:]:
        print(json.dumps(read(Path(arg))), flush=True)
