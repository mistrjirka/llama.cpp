#!/usr/bin/env python3
"""Screen lossless codecs on local SenseNova v4 slot files without loading a model.

Parses the known contiguous Q8 KV + standard recurrent + SCKP v1 layout.
Rejects unsupported layouts. Uses liblz4/libzstd through ctypes. No payloads
are emitted. --inspect only reads metadata and performs no compression.
"""
import argparse
from collections import Counter
import ctypes as C
import ctypes.util
import json
import pathlib
import statistics
import struct
import time

MIB = 1024 * 1024

class Reader:
    def __init__(self, file, end):
        self.file, self.end = file, end
    def pos(self): return self.file.tell()
    def read(self, fmt):
        size = struct.calcsize('<' + fmt)
        if size > self.end - self.pos(): raise ValueError('truncated metadata')
        return struct.unpack('<' + fmt, self.file.read(size))
    def skip(self, size):
        if size < 0 or size > self.end - self.pos(): raise ValueError('invalid payload range')
        self.file.seek(size, 1)
    def region(self, kind, size, **extra):
        result = dict(kind=kind, offset=self.pos(), bytes=size, **extra)
        self.skip(size)
        return result

def recurrent(reader, recurrent_layers, total_layers):
    cells, = reader.read('I')
    if cells != 1: raise ValueError('expected one recurrent cell')
    position, count = reader.read('iI')
    if position < 0 or count > 1: raise ValueError('unexpected recurrent cell metadata')
    reader.skip(count * 4)
    transpose, layers = reader.read('II')
    if transpose != 0 or layers != total_layers:
        raise ValueError('only standard recurrent rows are supported, not lazy/device envelopes')
    ranges = []
    for kind in ('recurrent_r', 'recurrent_s'):
        for layer in range(recurrent_layers):
            dtype, row_bytes = reader.read('IQ')
            if dtype not in (0, 1) or not 0 < row_bytes <= 32 * MIB:
                raise ValueError('invalid recurrent tensor row')
            ranges.append(reader.region(kind, row_bytes, dtype=dtype, layer=layer))
    return ranges

def parse_layout(path, recurrent_layers=42, total_layers=56):
    matches = []
    # Matches the existing snapshot-store parser and the optional 12-byte M-RoPE cell extension.
    for extension in (0, 12):
        try:
            with path.open('rb') as file:
                reader = Reader(file, path.stat().st_size)
                magic, version, token_words = reader.read('III')
                if (magic, version) != (0x67677371, 4): raise ValueError('not a v4 sequence file')
                reader.skip(token_words * 4)
                streams, cells = reader.read('II')
                if streams != 1 or not 0 < cells <= 1048576: raise ValueError('unsupported KV streams')
                previous = -1
                for _ in range(cells):
                    position, count = reader.read('iI')
                    if position < previous or position < 0 or count != 1: raise ValueError('invalid KV metadata')
                    previous = position
                    reader.skip(extension)
                    seq_id, = reader.read('i')
                    if seq_id != 0: raise ValueError('expected slot zero')
                transpose, layers, rotate_k, rotate_v = reader.read('IIII')
                if transpose or layers != total_layers - recurrent_layers or rotate_k > 4096 or rotate_v > 4096:
                    raise ValueError('unsupported KV tensor layout')
                kv = []
                for index in range(2 * layers):
                    dtype, row_bytes = reader.read('IQ')
                    if dtype != 8 or not 0 < row_bytes <= MIB: raise ValueError('expected Q8_0 rows')
                    kv.append(reader.region('q8_kv', row_bytes * cells, dtype=dtype, tensor=index))
                active_start = reader.pos()
                active_tensors = recurrent(reader, recurrent_layers, total_layers)
                active_end = reader.pos()
                checkpoint_ranges = []
                checkpoint_details = []
                if reader.pos() != reader.end:
                    magic, version, count = reader.read('III')
                    if (magic, version) != (0x504b4353, 1) or count > 1000:
                        raise ValueError('expected SCKP v1 after full state')
                    for index in range(count):
                        n_tokens, pos_min, pos_max = reader.read('qii')
                        target_size, = reader.read('Q')
                        target_start = reader.pos()
                        if target_size > reader.end - target_start: raise ValueError('checkpoint exceeds file')
                        target = Reader(file, target_start + target_size)
                        state_magic, seq_id = target.read('Ii')
                        if state_magic != 0xaf143cd8 or seq_id != 0: raise ValueError('not a host checkpoint')
                        tensors = recurrent(target, recurrent_layers, total_layers)
                        if target.pos() != target.end: raise ValueError('checkpoint size/layout mismatch')
                        checkpoint_ranges.append(dict(kind='recurrent_checkpoint', offset=target_start, bytes=target_size))
                        checkpoint_details.append(dict(index=index, n_tokens=n_tokens, pos_min=pos_min, pos_max=pos_max,
                            offset=target_start, bytes=target_size, tensor_payload=sum(t['bytes'] for t in tensors)))
                        for _ in range(2):
                            size, = reader.read('Q')
                            reader.skip(size)
                if reader.pos() != reader.end:
                    magic, version = reader.read('II')
                    if (magic, version) != (0x54464453, 1): raise ValueError('unexpected trailing data')
                    for _ in range(2):
                        size, = reader.read('Q')
                        reader.skip(size)
                if reader.pos() != reader.end: raise ValueError('unparsed trailing bytes')
                matches.append(dict(file=str(path), bytes=reader.end, token_words=token_words, kv_cells=cells,
                    cell_extension_bytes=extension, kv_payload_bytes=sum(r['bytes'] for r in kv),
                    active_recurrent_bytes=active_end-active_start, active_tensor_bytes=sum(r['bytes'] for r in active_tensors),
                    checkpoints=checkpoint_details, ranges={'q8_kv':kv, 'recurrent_checkpoint':checkpoint_ranges}))
        except (ValueError, struct.error) as error:
            failure = str(error)
    if len(matches) != 1: raise ValueError(f'{path.name}: unsupported/ambiguous layout ({failure if not matches else "multiple matches"})')
    return matches[0]

class Codec:
    def __init__(self, name):
        self.name = name
        self.lib = C.CDLL(ctypes.util.find_library('lz4' if name == 'lz4' else 'zstd'))
        self.cctx = self.dctx = None
        def setup(name, args, result):
            function = getattr(self.lib, name)
            function.argtypes, function.restype = args, result
            return function
        if name == 'lz4':
            self.bound = setup('LZ4_compressBound', [C.c_int], C.c_int)
            self.compress = setup('LZ4_compress_default', [C.c_void_p,C.c_void_p,C.c_int,C.c_int], C.c_int)
            self.decompress = setup('LZ4_decompress_safe', [C.c_void_p,C.c_void_p,C.c_int,C.c_int], C.c_int)
        else:
            self.bound = setup('ZSTD_compressBound', [C.c_size_t], C.c_size_t)
            self.compress = setup('ZSTD_compressCCtx', [C.c_void_p,C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t,C.c_int], C.c_size_t)
            self.decompress = setup('ZSTD_decompressDCtx', [C.c_void_p,C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t], C.c_size_t)
            self.is_error = setup('ZSTD_isError', [C.c_size_t], C.c_uint)
            self.cctx = setup('ZSTD_createCCtx', [], C.c_void_p)()
            self.dctx = setup('ZSTD_createDCtx', [], C.c_void_p)()
            self.csize = setup('ZSTD_sizeof_CCtx', [C.c_void_p], C.c_size_t)
            self.dsize = setup('ZSTD_sizeof_DCtx', [C.c_void_p], C.c_size_t)
            self.cfree = setup('ZSTD_freeCCtx', [C.c_void_p], C.c_size_t)
            self.dfree = setup('ZSTD_freeDCtx', [C.c_void_p], C.c_size_t)
            if not self.cctx or not self.dctx: raise MemoryError('codec context allocation failed')
        self.compare = C.CDLL(None).memcmp
        self.compare.argtypes, self.compare.restype = [C.c_void_p,C.c_void_p,C.c_size_t], C.c_int
    def close(self):
        if self.cctx: self.cfree(self.cctx); self.cctx = None
        if self.dctx: self.dfree(self.dctx); self.dctx = None
    def measure(self, raw, repeats):
        size = len(raw)
        source = (C.c_uint8 * size).from_buffer(raw)
        compressed = C.create_string_buffer(self.bound(size))
        restored = C.create_string_buffer(size)
        encode, decode = [], []
        for iteration in range(repeats + 1):
            start = time.perf_counter_ns()
            stored = self.compress(source, compressed, size, len(compressed)) if self.name == 'lz4' else \
                self.compress(self.cctx, compressed, len(compressed), source, size, 1)
            middle = time.perf_counter_ns()
            if stored <= 0 or (self.name != 'lz4' and self.is_error(stored)): raise RuntimeError('compression failed')
            length = self.decompress(compressed, restored, stored, size) if self.name == 'lz4' else \
                self.decompress(self.dctx, restored, size, compressed, stored)
            end = time.perf_counter_ns()
            if length != size or self.compare(source, restored, size): raise RuntimeError('restored bytes differ')
            if iteration:
                encode.append((middle-start)/1e6)
                decode.append((end-middle)/1e6)
        scratch = size + len(compressed) + size + (self.csize(self.cctx) + self.dsize(self.dctx) if self.cctx else 65536)
        if scratch > 128*MIB: raise MemoryError('scratch budget exceeded')
        return dict(raw_bytes=size, compressed_bytes=stored, stored_bytes=min(size, stored)+16,
                    encode_ms=statistics.median(encode), decode_ms=statistics.median(decode),
                    stored_decode_ms=statistics.median(decode) if stored < size else 0, scratch_bytes=scratch)

def windows(ranges, sample_bytes, window_bytes):
    if not ranges: return []
    window_bytes = min(window_bytes, sample_bytes)
    count = max(1, sample_bytes // window_bytes)
    selected = [min(len(ranges)-1, (2*i+1)*len(ranges)//(2*count)) for i in range(count)]
    multiplicities = Counter(selected)
    occurrences, result = {}, []
    for index in selected:
        region = ranges[index]
        occurrence = occurrences.get(index, 0)
        occurrences[index] = occurrence + 1
        multiplicity = multiplicities[index]
        size = min(window_bytes, region['bytes'] // multiplicity)
        offset = region['offset'] + (2*occurrence+1)*region['bytes']//(2*multiplicity) - size//2
        if size: result.append((offset, size))
    return result

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('files', nargs='+', type=pathlib.Path)
    ap.add_argument('--inspect', action='store_true')
    ap.add_argument('--sample-mib', type=int, default=32)
    ap.add_argument('--block-kib', type=int, nargs='+', default=[128, 4096])
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--output', type=pathlib.Path)
    args = ap.parse_args()
    if not 1 <= args.sample_mib <= 128 or not 1 <= args.repeats <= 20 or any(n < 4 or n > 16384 for n in args.block_kib):
        ap.error('samples 1..128 MiB, blocks 4..16384 KiB, repeats 1..20')
    layouts = [parse_layout(path) for path in args.files]
    result = {'layout':[ {k:v for k,v in layout.items() if k!='ranges'} for layout in layouts ], 'measurements':[],
              'notes':['Times exclude file I/O, buffer allocation, and byte verification. One warmup and median repeated codec calls per block.',
                       'Codec contexts are reused. Payloads stay local. Stored size includes raw fallback plus 16 bytes per block.',
                       'Samples are evenly spread across parsed tensor/checkpoint ranges. These are sample estimates, not whole-cache ratios.']}
    if not args.inspect:
        codecs = [Codec('lz4'), Codec('zstd1')]
        try:
            for layout in layouts:
                with pathlib.Path(layout['file']).open('rb') as file:
                    for group, ranges in layout['ranges'].items():
                        sample = windows(ranges, args.sample_mib*MIB, max(args.block_kib)*1024)
                        for block_kib in args.block_kib:
                            for codec in codecs:
                                sums = {key:0 for key in ['raw_bytes','compressed_bytes','stored_bytes','encode_ms','decode_ms','stored_decode_ms']}
                                peak = blocks = 0
                                for offset, available in sample:
                                    file.seek(offset)
                                    while available:
                                        size = min(available,block_kib*1024)
                                        raw = bytearray(size)
                                        if file.readinto(raw) != size: raise EOFError('sample read failed')
                                        measured = codec.measure(raw, args.repeats)
                                        for key in sums: sums[key] += measured[key]
                                        peak = max(peak,measured['scratch_bytes'])
                                        available -= size
                                        blocks += 1
                                if sums['raw_bytes']:
                                    row = dict(file=pathlib.Path(layout['file']).name, group=group, codec=codec.name,
                                        block_kib=block_kib, blocks=blocks, **sums, scratch_peak_bytes=peak,
                                        stored_fraction=sums['stored_bytes']/sums['raw_bytes'], verified_exact=True)
                                    result['measurements'].append(row)
                                    print(json.dumps(row),flush=True)
        finally:
            for codec in codecs: codec.close()
    if args.output: args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result if args.inspect else {'verified_rows':len(result['measurements'])},indent=2))

if __name__ == '__main__': main()
