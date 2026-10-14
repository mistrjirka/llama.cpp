import argparse, sys
import numpy as np
from aie.iron import ObjectFifo, Program, Runtime, Worker, Kernel, Buffer
from aie.iron.controlflow import range_
from aie.iron.device import NPU2, Tile
from aie.helpers.taplib import TensorAccessPattern
from aie.dialects import arith
from aie.extras import types as T
p = argparse.ArgumentParser()
p.add_argument("--qt", type=int, choices=[4, 5, 6, 8], default=4)
p.add_argument("--chains", type=int, default=2)
p.add_argument("--depth", type=int, default=3)
p.add_argument("--N", type=int, required=True)      # only shapes the runtime sequence (instruction stream), not the xclbin
p.add_argument("--K", type=int, required=True)
p.add_argument("--M", type=int, default=1)
p.add_argument("--obj", required=True)
p.add_argument("--stack", type=int, default=4096)
p.add_argument("--actrep", choices=["dma", "host"], default="host")   # replicate act per lane via stride-0 DMA or in the host BO
a = p.parse_args()
C = 8; CPC = 4 // a.chains; ncores = 32
REC = {4: 2304, 5: 2816, 6: 3360, 8: 4352}[a.qt]
NBK = a.K // 256; M = a.M
G = a.N // 512
assert a.N % 512 == 0 and a.K % 256 == 0
SV = a.K + a.K // 8 if a.qt == 8 else a.K + NBK * 4 + a.K // 8
ACT_BYTES = (64 + M * SV + 63) & ~63
ACT_RECS = -(-ACT_BYTES // REC)
nw = G * NBK
chain_bytes = nw * CPC * REC
rec_ty = np.ndarray[(REC,), np.dtype[np.int8]]
obj_ty = np.ndarray[(CPC * REC,), np.dtype[np.int8]]
oc = M * 16
out_core_ty = np.ndarray[(oc,), np.dtype[np.float32]]
out_col_ty = np.ndarray[(oc * 4,), np.dtype[np.float32]]
act_ty = np.ndarray[(ACT_BYTES,), np.dtype[np.int8]]
hdr_ty = np.ndarray[(16,), np.dtype[np.int32]]
k_act = Kernel("qk_act_rec", a.obj, [rec_ty, act_ty, hdr_ty])
k_w = Kernel("qk_w_rec", a.obj, [rec_ty, out_core_ty, act_ty])
def core_fn(of_w, of_o, ka, kw, act, hdr):
    for _ in range_(ACT_RECS):
        e = of_w.acquire(1); ka(e, act, hdr); of_w.release(1)
    g = arith.index_cast(T.index(), hdr[0])
    for _ in range_(0, g, 1):
        o = of_o.acquire(1)
        for _ in range_(NBK):
            e = of_w.acquire(1); kw(e, o, act); of_w.release(1)
        of_o.release(1)
workers, in_prods, out_conses = [], [], []
for c in range(C):
    t = Tile(col=c)
    of_out = ObjectFifo(out_col_ty, name=f"out{c}", depth=2)
    subs = of_out.prod().join([i * oc for i in range(4)], tile=t, obj_types=[out_core_ty] * 4,
                              names=[f"o{c}_{i}" for i in range(4)], depths=[2] * 4)
    out_conses.append(of_out.cons(tile=t))
    k = 0
    for ch in range(a.chains):
        of_in = ObjectFifo(obj_ty, name=f"in{c}_{ch}", depth=a.depth)
        wf = of_in.cons().split([i * REC for i in range(CPC)], tile=t, obj_types=[rec_ty] * CPC,
                                names=[f"w{c}_{ch}_{i}" for i in range(CPC)], depths=[a.depth] * CPC)
        in_prods.append(of_in.prod(tile=t))
        for i in range(CPC):
            workers.append(Worker(core_fn, [wf[i].cons(), subs[k].prod(), k_act, k_w, Buffer(act_ty, name=f"act{c}_{k}"),
                                            Buffer(hdr_ty, name=f"hdr{c}_{k}")], tile=t, stack_size=a.stack)); k += 1
n_in, n_out = len(in_prods), len(out_conses)
W_el = chain_bytes * n_in // 4
A_el = ACT_RECS * REC // 4 * (CPC if a.actrep == "host" else 1)
O_el = a.N * M
def sequence(wbuf, abuf, obuf, *hs):
    prods, conses = hs[:n_in], hs[n_in:]
    for i in range(n_in):
        if a.actrep == "dma":
            atap = TensorAccessPattern((A_el,), 0, [ACT_RECS, CPC, REC // 4], [REC // 4, 0, 1])
        else:
            atap = TensorAccessPattern((A_el,), 0, [1, 1, 1, A_el], [0, 0, 0, 1])
        prods[i].fill(abuf, tap=atap, wait=False)
        prods[i].fill(wbuf, tap=TensorAccessPattern((W_el,), i * chain_bytes // 4, [1, 1, 1, chain_bytes // 4], [0, 0, 0, 1]), wait=False)
    per_col = nw // NBK * 4 * oc
    for i in range(n_out):
        conses[i].drain(obuf, tap=TensorAccessPattern((O_el,), i * per_col, [1, 1, 1, per_col], [0, 0, 0, 1]), wait=True)
rt = Runtime(sequence, [np.ndarray[(W_el,), np.dtype[np.int32]], np.ndarray[(A_el,), np.dtype[np.int32]],
                        np.ndarray[(O_el,), np.dtype[np.float32]], *in_prods, *out_conses])
print(Program(NPU2(), rt, workers=workers).resolve_program())
sys.stderr.write(f"META n_in={n_in} chain_bytes={chain_bytes} N={a.N} K={a.K} M={M} qt={a.qt} G={G} ACT_RECS={ACT_RECS} nw={nw} wbytes={chain_bytes*n_in} abytes={A_el*4} obytes={O_el*4}\n")
