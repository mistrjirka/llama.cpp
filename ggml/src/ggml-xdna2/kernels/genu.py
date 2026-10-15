# Unified decode kernel generator (IRON).  One xclbin for every (qtype, K <= KMAX, N <= 8*4*GMAX*16 rows).
# The runtime sequence (instruction stream) depends on three host values only:
#   --chain_words  : 32-bit words of the weight stream of one chain  (= n_w_chunks * 2 lanes * S / 4)
#   --a_words      : 32-bit words of the activation BO               (= n_act_chunks * 2 lanes * S / 4)
#   --g            : 16-row groups per core (output words per column = g * 4 cores * 16)
# which the host patches linearly (patch spec, see mkspec.py).
import argparse, sys
import numpy as np
from aie.iron import ObjectFifo, Program, Runtime, Worker, Kernel, Buffer
from aie.iron.controlflow import range_
from aie.iron.device import NPU2, Tile
from aie.helpers.taplib import TensorAccessPattern
from aie.dialects import arith
from aie.extras import types as T
p = argparse.ArgumentParser()
p.add_argument("--obj", required=True)
p.add_argument("--S", type=int, default=1024)            # chunk bytes (must equal -DQK_S)
p.add_argument("--kmax", type=int, default=12288)
p.add_argument("--gmax", type=int, default=128)
p.add_argument("--depth", type=int, default=8)      # core side
p.add_argument("--mdepth", type=int, default=2)     # memtile side
p.add_argument("--stack", type=int, default=9216)
p.add_argument("--chain_words", type=int, required=True)
p.add_argument("--a_words", type=int, required=True)
p.add_argument("--g", type=int, required=True)
a = p.parse_args()
C = 8; CHAINS = 1; CPC = 4
S = a.S
SVMAX = a.kmax + a.kmax // 256 * 4 + a.kmax // 8
ACT_MAX = ((64 + SVMAX + S - 1) // S) * S
chunk_ty = np.ndarray[(S,), np.dtype[np.int8]]
obj_ty = np.ndarray[(CPC * S,), np.dtype[np.int8]]
act_ty = np.ndarray[(ACT_MAX,), np.dtype[np.int8]]
wrec_ty = np.ndarray[(4352,), np.dtype[np.int8]]
ost_ty = np.ndarray[(a.gmax * 16,), np.dtype[np.float32]]
hdr_ty = np.ndarray[(16,), np.dtype[np.int32]]
out_core_ty = np.ndarray[(16,), np.dtype[np.float32]]
out_col_ty = np.ndarray[(64,), np.dtype[np.float32]]
k_first = Kernel("qk_first", a.obj, [chunk_ty, act_ty, hdr_ty])
k_act = Kernel("qk_act", a.obj, [chunk_ty, act_ty])
k_w = Kernel("qk_w", a.obj, [chunk_ty, act_ty, wrec_ty, ost_ty, hdr_ty])
k_flush = Kernel("qk_flush", a.obj, [out_core_ty, ost_ty])

def core_fn(of_w, of_o, kf, ka, kw, kfl, act, hdr, wrec, ost):
    e = of_w.acquire(1); kf(e, act, hdr); of_w.release(1)
    na = arith.index_cast(T.index(), hdr[3])
    for _ in range_(0, na, 1):
        e = of_w.acquire(1); ka(e, act); of_w.release(1)
    nw = arith.index_cast(T.index(), hdr[4])
    for _ in range_(0, nw, 1):
        e = of_w.acquire(1); kw(e, act, wrec, ost, hdr); of_w.release(1)
    g = arith.index_cast(T.index(), hdr[0])
    for _ in range_(0, g, 1):
        o = of_o.acquire(1); kfl(o, ost); of_o.release(1)

workers, in_prods, out_conses = [], [], []
for c in range(C):
    t = Tile(col=c)
    of_out = ObjectFifo(out_col_ty, name=f"out{c}", depth=2)
    subs = of_out.prod().join([i * 16 for i in range(4)], tile=t, obj_types=[out_core_ty] * 4,
                              names=[f"o{c}_{i}" for i in range(4)], depths=[2] * 4)
    out_conses.append(of_out.cons(tile=t))
    k = 0
    for ch in range(CHAINS):
        of_in = ObjectFifo(obj_ty, name=f"in{c}_{ch}", depth=a.mdepth)
        wf = of_in.cons().split([i * S for i in range(CPC)], tile=t, obj_types=[chunk_ty] * CPC,
                                names=[f"w{c}_{ch}_{i}" for i in range(CPC)], depths=[a.depth] * CPC)
        in_prods.append(of_in.prod(tile=t))
        for i in range(CPC):
            workers.append(Worker(core_fn, [wf[i].cons(), subs[k].prod(), k_first, k_act, k_w, k_flush,
                                            Buffer(act_ty, name=f"act{c}_{k}"), Buffer(hdr_ty, name=f"hdr{c}_{k}"),
                                            Buffer(wrec_ty, name=f"wrec{c}_{k}"), Buffer(ost_ty, name=f"ost{c}_{k}")],
                                   tile=t, stack_size=a.stack)); k += 1
n_in, n_out = len(in_prods), len(out_conses)
W_el = a.chain_words * n_in
A_el = a.a_words
per_col = a.g * 4 * 16
O_el = per_col * n_out
def sequence(wbuf, abuf, obuf, *hs):
    prods, conses = hs[:n_in], hs[n_in:]
    for i in range(n_in):
        prods[i].fill(abuf, tap=TensorAccessPattern((A_el,), 0, [1, 1, 1, A_el], [0, 0, 0, 1]), wait=False)
        prods[i].fill(wbuf, tap=TensorAccessPattern((W_el,), i * a.chain_words, [1, 1, 1, a.chain_words], [0, 0, 0, 1]), wait=False)
    for i in range(n_out):
        conses[i].drain(obuf, tap=TensorAccessPattern((O_el,), i * per_col, [1, 1, 1, per_col], [0, 0, 0, 1]), wait=True)
rt = Runtime(sequence, [np.ndarray[(W_el,), np.dtype[np.int32]], np.ndarray[(A_el,), np.dtype[np.int32]],
                        np.ndarray[(O_el,), np.dtype[np.float32]], *in_prods, *out_conses])
print(Program(NPU2(), rt, workers=workers).resolve_program())
sys.stderr.write(f"META S={S} kmax={a.kmax} gmax={a.gmax} act_max={ACT_MAX} depth={a.depth} chain_words={a.chain_words} a_words={a.a_words} g={a.g}\n")
