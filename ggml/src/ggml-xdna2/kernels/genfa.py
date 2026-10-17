# Flash-attention prefill design (IRON, fa.cc): GQA with 4 kv heads x 4 q heads per kv head, head dim 256.  One "pair" = 32 queries (2 q tiles
# of 16) x 16 q heads; a dispatch runs `pairs` consecutive pairs (pair r: nb + 2r K/V blocks of 16 keys).
#   kv head g owns columns 2g, 2g+1; its 8 cores are (qi = column parity: q tile of the pair, hh = q head within the group).
#   K stream (shim 2g -> memtile 2g -> broadcast to the 8 cores):   per pair the Q prefix (8 objects, one per core), then the K blocks
#   V stream (shim 2g+1 -> memtile 2g+1 -> broadcast):              per pair the header object, then the V blocks
#   O: each core 2 x 32 tiles of 8x8 f32 + 128 row sums -> joined per column -> shim
# Buffers (all bfp16ebs8 host-encoded, 4608 B objects): qbuf [pair][g 4][core 8][4608] + [pair][header object 4608 (i32 nb, d0 at its start)];
# kbuf/vbuf = the bfp16 shadow of one layer's KV cache [kv head 4][kvmax/16 blocks][4608]; obuf f32 [pair][col 8][core 4][OC].
# The instruction stream depends on two host values: nb (BD lengths (nb + 2r) * 4608: affine) and kvmax (kv head offsets g * kvmax/16 * 4608: linear).
import argparse, sys
import numpy as np
from ml_dtypes import bfloat16
from aie.iron import ObjectFifo, Program, Runtime, Worker, Kernel, Buffer
from aie.iron.controlflow import range_
from aie.iron.runtime.taskgroup import TaskGroup
from aie.iron.device import NPU2, Tile
from aie.helpers.taplib import TensorAccessPattern
from aie.dialects import arith
from aie.extras import types as T
p = argparse.ArgumentParser()
p.add_argument("--obj", required=True)
p.add_argument("--nb", type=int, required=True)
p.add_argument("--kvmax", type=int, required=True)    # shadow KV cache rows per kv head
p.add_argument("--stack", type=int, default=4096)
p.add_argument("--pairs", type=int, default=1)       # 32-query pairs per dispatch (pair p: nb + 2p blocks)
a = p.parse_args()
G = 4; NKV = 4; D = 256; BQ = 16; BK = 16; NC = 8
OBJ = BQ * D * 9 // 8                                 # 4608 B bfp16 (= 16 keys too)
OC = 2 * 32 * 64 + 128                                # f32 per core: unnormalized O + row sums
obj_ty = np.ndarray[(OBJ,), np.dtype[np.int8]]
o_core_ty = np.ndarray[(OC,), np.dtype[np.float32]]
o_col_ty = np.ndarray[(4 * OC,), np.dtype[np.float32]]
i4_ty = np.ndarray[(4,), np.dtype[np.int32]]
k_hdr = Kernel("fa_hdr", a.obj, [obj_ty, i4_ty, i4_ty])
k_q = Kernel("fa_q", a.obj, [obj_ty, i4_ty])
k_init = Kernel("fa_init", a.obj, [o_core_ty])
k_blk = Kernel("fa_blk", a.obj, [obj_ty, obj_ty, o_core_ty, i4_ty, i4_ty])
k_end = Kernel("fa_end", a.obj, [o_core_ty])

def core_fn(of_k, of_v, of_o, khd, kq, kin, kbl, ken, hdr, ident):
    hv = of_v.acquire(1); khd(hv, hdr, ident); of_v.release(1)
    for _ in range_(NC):
        q = of_k.acquire(1); kq(q, ident); of_k.release(1)
    o = of_o.acquire(1)
    kin(o)
    nb = arith.index_cast(T.index(), hdr[0])
    for _ in range_(0, nb, 1):
        kk = of_k.acquire(1); vv = of_v.acquire(1)
        kbl(kk, vv, o, hdr, ident)
        of_k.release(1); of_v.release(1)
    ken(o)
    of_o.release(1)

workers, k_prods, v_prods, o_conses = [], [], [], []
for g in range(G):
    ck, cv = 2 * g, 2 * g + 1
    ofk = ObjectFifo(obj_ty, name=f"k{g}", depth=2)
    kf = ofk.cons().forward(tile=Tile(col=ck, row=1), obj_type=obj_ty, depth=2, name=f"km{g}")
    k_prods.append(ofk.prod(tile=Tile(col=ck, row=0)))
    ofv = ObjectFifo(obj_ty, name=f"v{g}", depth=2)
    vf = ofv.cons().forward(tile=Tile(col=cv, row=1), obj_type=obj_ty, depth=2, name=f"vm{g}")
    v_prods.append(ofv.prod(tile=Tile(col=cv, row=0)))
    for qi in range(2):
        c = 2 * g + qi
        of_out = ObjectFifo(o_col_ty, name=f"out{c}", depth=1)
        subs = of_out.prod().join([i * OC for i in range(4)], tile=Tile(col=c, row=1), obj_types=[o_core_ty] * 4,
                                  names=[f"o{c}_{i}" for i in range(4)], depths=[1] * 4)
        o_conses.append(of_out.cons(tile=Tile(col=c, row=0)))
        for hh in range(4):
            me = qi * 4 + hh
            workers.append(Worker(core_fn, [kf.cons(), vf.cons(), subs[hh].prod(), k_hdr, k_q, k_init, k_blk, k_end,
                                            Buffer(i4_ty, name=f"hdr{c}_{hh}"),
                                            Buffer(i4_ty, initial_value=np.array([me, qi, 0, 0], dtype=np.int32), name=f"id{c}_{hh}")],
                                  tile=Tile(col=c, row=2 + hh), stack_size=a.stack))

R = a.pairs
Q_EL = R * (G * NC * OBJ + OBJ)                       # per pair: Q objects, then the header objects of all pairs
HOFF = R * G * NC * OBJ
KVB = a.kvmax // BK * OBJ                             # bytes per kv head
KV_EL = NKV * KVB
O_EL = R * 2 * G * 4 * OC
def sequence(qbuf, kbuf, vbuf, obuf, *hs):
    # one task group per pair; pair r - 1 is awaited/freed only after pair r is queued, so the DMAs never run dry
    # and at most two pairs' BDs (3 per shim tile each) are live
    kp = hs[:G]; vp = hs[G:2 * G]; oc = hs[2 * G:]
    groups = []
    for r in range(R):
        tg = TaskGroup()
        nb = a.nb + 2 * r
        for g in range(G):
            kp[g].fill(qbuf, tap=TensorAccessPattern((Q_EL,), (r * G + g) * NC * OBJ, [1, 1, 1, NC * OBJ], [0, 0, 0, 1]), group=tg)
            kp[g].fill(kbuf, tap=TensorAccessPattern((KV_EL,), g * KVB, [1, 1, 1, nb * OBJ], [0, 0, 0, 1]), group=tg)
            vp[g].fill(qbuf, tap=TensorAccessPattern((Q_EL,), HOFF + r * OBJ, [1, 1, 1, OBJ], [0, 0, 0, 1]), group=tg)
            vp[g].fill(vbuf, tap=TensorAccessPattern((KV_EL,), g * KVB, [1, 1, 1, nb * OBJ], [0, 0, 0, 1]), group=tg)
        for c in range(2 * G):
            oc[c].drain(obuf, tap=TensorAccessPattern((O_EL,), (r * 2 * G + c) * 4 * OC, [1, 1, 1, 4 * OC], [0, 0, 0, 1]), wait=True, group=tg)
        groups.append(tg)
        if r >= 1:
            groups[r - 1].finish()
    groups[-1].finish()
rt = Runtime(sequence, [np.ndarray[(Q_EL,), np.dtype[np.int8]], np.ndarray[(KV_EL,), np.dtype[np.int8]],
                        np.ndarray[(KV_EL,), np.dtype[np.int8]], np.ndarray[(O_EL,), np.dtype[np.float32]],
                        *k_prods, *v_prods, *o_conses])
print(Program(NPU2(), rt, workers=workers).resolve_program())
sys.stderr.write(f"META nb={a.nb} groups={G} kvmax={a.kvmax} pairs={R}\n")
