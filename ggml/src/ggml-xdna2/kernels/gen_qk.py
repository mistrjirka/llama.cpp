import argparse, sys
import numpy as np
from aie.iron import ObjectFifo, Program, Runtime, Worker, Kernel, Buffer
from aie.iron.controlflow import range_
from aie.iron.device import NPU2, Tile
from aie.helpers.taplib import TensorAccessPattern
p = argparse.ArgumentParser()
p.add_argument("--qt", type=int, choices=[4, 6], default=4)
p.add_argument("--chains", type=int, default=2)
p.add_argument("--depth", type=int, default=3)
p.add_argument("--N", type=int, required=True)
p.add_argument("--K", type=int, required=True)
p.add_argument("--M", type=int, default=1)
p.add_argument("--obj", required=True)
p.add_argument("--stack", type=int, default=8192)
p.add_argument("--reps", type=int, default=0)
a = p.parse_args()
C = 8; CPC = 4 // a.chains; cores_per_col = 4; ncores = 32
REC = 2304 if a.qt == 4 else 3360
NBK = a.K // 256; RPC = a.N // ncores; M = a.M
assert RPC % 16 == 0 and a.K % 4096 == 0
SV = a.K + NBK * 4 + a.K // 8
ACT_RECS = -(-M * SV // REC)
nw = (RPC // 16) * NBK
nrec = ACT_RECS + nw
chain_bytes = nrec * CPC * REC
fill_bytes = (ACT_RECS + 1) * CPC * REC if a.reps > 0 else chain_bytes
rec_ty = np.ndarray[(REC,), np.dtype[np.int8]]
obj_ty = np.ndarray[(CPC * REC,), np.dtype[np.int8]]
oc = M * RPC
out_core_ty = np.ndarray[(oc,), np.dtype[np.float32]]
out_col_ty = np.ndarray[(oc * cores_per_col,), np.dtype[np.float32]]
ACT_BYTES = (M * SV + 63) & ~63
act_ty = np.ndarray[(ACT_BYTES,), np.dtype[np.int8]]
k_act = Kernel("qk_act_rec", a.obj, [rec_ty, act_ty])
k_w = Kernel("qk_w_rec", a.obj, [rec_ty, out_core_ty, act_ty])
def core_fn(of_w, of_o, ka, kw, act):
    o = of_o.acquire(1)
    for _ in range_(ACT_RECS):
        e = of_w.acquire(1); ka(e, act); of_w.release(1)
    if a.reps > 0:
        e = of_w.acquire(1)
        for _ in range_(a.reps):
            kw(e, o, act)
        of_w.release(1)
    else:
        for _ in range_(nw):
            e = of_w.acquire(1); kw(e, o, act); of_w.release(1)
    of_o.release(1)
workers, in_prods, out_conses = [], [], []
for c in range(C):
    t = Tile(col=c)
    of_out = ObjectFifo(out_col_ty, name=f"out{c}", depth=1)
    subs = of_out.prod().join([i * oc for i in range(cores_per_col)], tile=t, obj_types=[out_core_ty] * cores_per_col,
                              names=[f"o{c}_{i}" for i in range(cores_per_col)], depths=[1] * cores_per_col)
    out_conses.append(of_out.cons(tile=t))
    k = 0
    for ch in range(a.chains):
        of_in = ObjectFifo(obj_ty, name=f"in{c}_{ch}", depth=a.depth)
        wf = of_in.cons().split([i * REC for i in range(CPC)], tile=t, obj_types=[rec_ty] * CPC,
                                names=[f"w{c}_{ch}_{i}" for i in range(CPC)], depths=[a.depth] * CPC)
        in_prods.append(of_in.prod(tile=t))
        for i in range(CPC):
            workers.append(Worker(core_fn, [wf[i].cons(), subs[k].prod(), k_act, k_w, Buffer(act_ty, name=f"act{c}_{k}")], tile=t, stack_size=a.stack)); k += 1
n_in, n_out = len(in_prods), len(out_conses)
W_el, O_el = chain_bytes * n_in // 4, a.N * M
def sequence(wbuf, obuf, *hs):
    prods, conses = hs[:n_in], hs[n_in:]
    for i in range(n_in):
        prods[i].fill(wbuf, tap=TensorAccessPattern((W_el,), i * chain_bytes // 4, [1, 1, 1, fill_bytes // 4], [0, 0, 0, 1]), wait=False)
    for i in range(n_out):
        conses[i].drain(obuf, tap=TensorAccessPattern((O_el,), i * oc * cores_per_col, [1, 1, 1, oc * cores_per_col], [0, 0, 0, 1]), wait=True)
rt = Runtime(sequence, [np.ndarray[(W_el,), np.dtype[np.int32]], np.ndarray[(O_el,), np.dtype[np.float32]], *in_prods, *out_conses])
print(Program(NPU2(), rt, workers=workers).resolve_program())
sys.stderr.write(f"META n_in={n_in} chain_bytes={chain_bytes} N={a.N} K={a.K} M={M} qt={a.qt} RPC={RPC} ACT_RECS={ACT_RECS} nrec={nrec} wbytes={chain_bytes*n_in} obytes={O_el*4}\n")
