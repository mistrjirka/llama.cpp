# Universal prefill kernel generator (IRON), layout v2.  One xclbin for every (qtype, K <= KMAX, N); one dispatch = one 64-token tile x one split.
#   W (weights, layout v2 chain region of a column: [n_w][4 lanes][1 KB]) : one shim channel per column -> memtile (4 KB objects) -> split to the 4 cores (1 KB chunks)
#   A (activation tile, bfp16ebs8; [hdr object][nsb x nbk slabs of 18432 B]) : shim col 0 -> memtile col 0 -> broadcast to all 32 cores
#   Y (fp32): per core nsb objects of GSB*1024 floats (the C accumulators live in the output object) -> joined per column -> shim
# Instruction stream depends on three host values (each linear, zero intercept): chain_words (W words per column), a_words, o_words (Y words per column)
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
p.add_argument("--stack", type=int, default=6144)
p.add_argument("--wdepth", type=int, default=4)       # core side chunk fifo depth
p.add_argument("--mdepth", type=int, default=2)       # memtile side
p.add_argument("--chain_words", type=int, required=True)
p.add_argument("--a_words", type=int, required=True)
p.add_argument("--o_words", type=int, required=True)  # f32 words drained per column
a = p.parse_args()
C = 8; CPC = 4; S = 1024; GSB = 5; AOBJ = 18432
chunk_ty = np.ndarray[(S,), np.dtype[np.int8]]
obj_ty = np.ndarray[(CPC * S,), np.dtype[np.int8]]
A_ty = np.ndarray[(AOBJ,), np.dtype[np.int8]]
oc = GSB * 1024
out_core_ty = np.ndarray[(oc,), np.dtype[np.float32]]
out_col_ty = np.ndarray[(oc * CPC,), np.dtype[np.float32]]
hdr_ty = np.ndarray[(16,), np.dtype[np.int32]]
wrec_ty = np.ndarray[(4352,), np.dtype[np.int8]]
bt_ty = np.ndarray[(64 * 72,), np.dtype[np.int8]]
k_hdr = Kernel("pf_hdr", a.obj, [A_ty, hdr_ty])
k_zero = Kernel("pf_zero", a.obj, [out_core_ty])
k_begin = Kernel("pf_slab_begin", a.obj, [hdr_ty, hdr_ty])
k_chunk = Kernel("pf_chunk", a.obj, [chunk_ty, A_ty, out_core_ty, wrec_ty, bt_ty, hdr_ty])

def core_fn(of_w, of_a, of_o, khd, kz, kbg, kch, hdr, ctl, wrec, Bt):
    ao = of_a.acquire(1); khd(ao, hdr); of_a.release(1)
    nsb = arith.index_cast(T.index(), hdr[0])
    nbk = arith.index_cast(T.index(), hdr[1])
    for _ in range_(0, nsb, 1):
        o = of_o.acquire(1)
        kz(o)
        for _ in range_(0, nbk, 1):
            av = of_a.acquire(1)
            kbg(hdr, ctl)
            nc = arith.index_cast(T.index(), ctl[0])
            for _ in range_(0, nc, 1):
                w = of_w.acquire(1); kch(w, av, o, wrec, Bt, hdr); of_w.release(1)
            of_a.release(1)
        of_o.release(1)

workers, w_prods, out_conses = [], [], []
of_a = ObjectFifo(A_ty, name="ain", depth=1)
a_prod = of_a.prod(tile=Tile(col=0))
af = of_a.cons().forward(tile=Tile(col=0), obj_type=A_ty, depth=1, name="a2_shared")
for c in range(C):
    t = Tile(col=c)
    of_out = ObjectFifo(out_col_ty, name=f"out{c}", depth=1)
    subs = of_out.prod().join([i * oc for i in range(CPC)], tile=t, obj_types=[out_core_ty] * CPC,
                              names=[f"o{c}_{i}" for i in range(CPC)], depths=[1] * CPC)
    out_conses.append(of_out.cons(tile=t))
    of_in = ObjectFifo(obj_ty, name=f"win{c}", depth=a.mdepth)
    wf = of_in.cons().split([i * S for i in range(CPC)], tile=t, obj_types=[chunk_ty] * CPC,
                            names=[f"w{c}_{i}" for i in range(CPC)], depths=[a.wdepth] * CPC)
    w_prods.append(of_in.prod(tile=t))
    for i in range(CPC):
        workers.append(Worker(core_fn, [wf[i].cons(), af.cons(), subs[i].prod(), k_hdr, k_zero, k_begin, k_chunk,
                                        Buffer(hdr_ty, name=f"hdr{c}_{i}"), Buffer(hdr_ty, name=f"ctl{c}_{i}"),
                                        Buffer(wrec_ty, name=f"wrec{c}_{i}"), Buffer(bt_ty, name=f"bt{c}_{i}")],
                              tile=t, stack_size=a.stack))
W_el = a.chain_words * C
A_el = a.a_words
O_el = a.o_words * C
def sequence(wbuf, abuf, obuf, *hs):
    wp = hs[:C]; ap = hs[C]; oc_ = hs[C + 1:]
    ap.fill(abuf, tap=TensorAccessPattern((A_el,), 0, [1, 1, 1, A_el], [0, 0, 0, 1]), wait=False)
    for i in range(C):
        wp[i].fill(wbuf, tap=TensorAccessPattern((W_el,), i * a.chain_words, [1, 1, 1, a.chain_words], [0, 0, 0, 1]), wait=False)
    for i in range(C):
        oc_[i].drain(obuf, tap=TensorAccessPattern((O_el,), i * a.o_words, [1, 1, 1, a.o_words], [0, 0, 0, 1]), wait=True)
rt = Runtime(sequence, [np.ndarray[(W_el,), np.dtype[np.int32]], np.ndarray[(A_el,), np.dtype[np.int32]],
                        np.ndarray[(O_el,), np.dtype[np.float32]], *w_prods, a_prod, *out_conses])
print(Program(NPU2(), rt, workers=workers).resolve_program())
sys.stderr.write(f"META chain_words={a.chain_words} a_words={a.a_words} o_words={a.o_words} wdepth={a.wdepth}\n")
