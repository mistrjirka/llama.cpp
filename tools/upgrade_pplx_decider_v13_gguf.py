#!/usr/bin/env python3
"""Upgrade legacy pplx-decider-v1.3 GGUF metadata without touching quantized weights.

Writes a NEW file. Keeps all GGUF tensor information and tensor bytes unchanged.
Requires the accompanying llama.cpp Decision v4 runtime fix.
"""
import argparse
import os
import shutil
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "conversion"))
from pplx_decider_v4 import systemone_template_v4  # noqa: E402

SCALAR = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
ALIGNMENT = 32


class Scanner:
    def __init__(self, blob: bytes):
        self.buf = blob
        self.pos = 0

    def take(self, size: int) -> bytes:
        if size < 0 or self.pos + size > len(self.buf):
            raise ValueError("GGUF header is incomplete or corrupt")
        out = self.buf[self.pos:self.pos + size]
        self.pos += size
        return out

    def u32(self) -> int:
        return struct.unpack("<I", self.take(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.take(8))[0]

    def string(self) -> bytes:
        return self.take(self.u64())

    def value(self, typ: int):
        if typ in SCALAR:
            self.take(SCALAR[typ])
        elif typ == 8:  # STRING
            self.string()
        elif typ == 9:  # ARRAY
            elem_type = self.u32()
            count = self.u64()
            for _ in range(count):
                self.value(elem_type)
        else:
            raise ValueError(f"Unknown GGUF metadata type: {typ}")


def kv(key: str, typ: int, body: bytes) -> bytes:
    key_bytes = key.encode()
    return struct.pack("<Q", len(key_bytes)) + key_bytes + struct.pack("<I", typ) + body


def encoded_string(data: bytes) -> bytes:
    return struct.pack("<Q", len(data)) + data


def upgrade(src: Path, dst: Path):
    if src.resolve() == dst.resolve() or dst.exists() or dst.with_suffix(dst.suffix + ".partial").exists():
        raise ValueError("Output must be a new, nonexisting file")
    with src.open("rb") as stream:
        initial = stream.read(32 * 1024 * 1024)
    scan = Scanner(initial)
    magic, version, n_tensors, n_kv = struct.unpack("<4sIQQ", scan.take(24))
    if magic != b"GGUF" or version != 3:
        raise ValueError(f"Expected GGUF v3, got {magic} {version}")
    all_kv = []
    keys = set()
    for _ in range(n_kv):
        start = scan.pos
        key = scan.string().decode("utf-8")
        typ = scan.u32()
        scan.value(typ)
        if key in keys:
            raise ValueError(f"Duplicate metadata key: {key}")
        keys.add(key)
        all_kv.append((key, typ, initial[start:scan.pos]))
    tensor_info_start = scan.pos
    for _ in range(n_tensors):
        scan.string()
        n_dims = scan.u32()
        if n_dims > 4:
            raise ValueError(f"Invalid tensor dimensions: {n_dims}")
        scan.take(8 * n_dims + 4 + 8)
    tensor_info = initial[tensor_info_start:scan.pos]
    old_data_offset = (scan.pos + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT

    expected = {"qwen35.decision.type", "tokenizer.chat_template.systemone"}
    if not expected <= keys:
        raise ValueError(f"Missing Decision metadata: {expected - keys}")
    for key in ("qwen35.attention.causal", "qwen35.decision.protocol_version"):
        if key in keys:
            raise ValueError(f"Model is already configured for v4 or modified: {key}")
    template = systemone_template_v4().encode()
    old_templates = [raw for key, typ, raw in all_kv if key == "tokenizer.chat_template.systemone"]
    assert len(old_templates) == 1
    if b"Reply with only the selected option code" not in old_templates[0]:
        raise ValueError("Refusing to modify an unknown source template")
    new_meta = []
    for key, typ, raw in all_kv:
        if key == "tokenizer.chat_template.systemone":
            raw = kv(key, 8, encoded_string(template))
        new_meta.append(raw)
    new_meta += [
        kv("qwen35.attention.causal", 7, b"\x00"),
        kv("qwen35.decision.protocol_version", 4, struct.pack("<I", 4)),
    ]
    out_header = struct.pack("<4sIQQ", magic, version, n_tensors, n_kv + 2) + b"".join(new_meta) + tensor_info
    out_header += b"\x00" * (-len(out_header) % ALIGNMENT)
    print(f"GGUF tensor count={n_tensors}, metadata count {n_kv} -> {n_kv + 2}, tensor offset {old_data_offset} -> {len(out_header)}",flush=True)
    partial = dst.with_suffix(dst.suffix + ".partial")
    try:
        with src.open("rb") as ins, partial.open("xb") as out:
            out.write(out_header)
            ins.seek(old_data_offset)
            shutil.copyfileobj(ins, out, length=16 * 1024 * 1024)
            out.flush()
            os.fsync(out.fileno())
        if partial.stat().st_size - len(out_header) != src.stat().st_size - old_data_offset:
            raise ValueError("Tensor bytes copied incorrectly")
        partial.replace(dst)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    print(f"Created {dst}, {dst.stat().st_size} bytes. Original unchanged.", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    upgrade(a.source, a.output)
