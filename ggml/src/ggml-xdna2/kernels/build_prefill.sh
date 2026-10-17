#!/bin/bash
# Build the UNIVERSAL NPU PREFILL kernel (pf.cc + genp.py) for ggml-xdna2: one xclbin for every (qtype, K <= 12288, N), one dispatch = one 64-token tile
# of one weight-matrix split. Like build_kernel.sh, only the instruction stream depends on host values (chain_words, a_words, o_words: linear, zero
# intercept; a_rep: affine) in exactly one set of words: the xclbin is built once together with the base stream and one stream per varied parameter.
# Does not touch the NPU.
#
# One dispatch runs T consecutive 64-token tiles of the same split (T = 1 2 4 8 16 32: one stream set per T; the per-tile offsets make the a_words
# words affine, so every parameter is patched with the affine rule).
#
# usage: build_prefill.sh <mlir-aie-env-dir> <outdir>
# writes <outdir>/kernel.xclbin, per T: insts_t<T>_base.bin (2048,2048,2048, a_rep 2), _cw (4096,..), _aw, _ow, _rw (a_rep 4), meta.txt
export LC_ALL=C
H=$(cd "$(dirname "$0")" && pwd)
TOOL=$1; OUT=$2
. "$H/env.sh" "$TOOL" || exit 10
mkdir -p "$OUT/work" && cd "$OUT/work" || exit 11
# the BFP pipeliner guard (lab stage4): the software pipeliner is disabled for this kernel
$PEANOCC -mllvm -enable-pipeliner=false -I"$H" -c "$H/pf.cc" -o k.o || { echo CCFAIL; exit 12; }
build() {   # name chain_words a_words o_words a_rep tiles
  python "$H/genp.py" --obj k.o --chain_words $2 --a_words $3 --o_words $4 --a_rep $5 --tiles $6 > $1.mlir 2> $1.meta || { echo GENFAIL $1; cat $1.meta; return 13; }
  if [ $1 = t1_base ]; then XB="--get-xclbin --xclbin-name=$1.xclbin --dump-intermediates"; else XB=""; fi
  aiecc --no-xchesscc --peano=$PEANO_INSTALL_DIR $XB --get-npu-insts --npu-insts-name=$1.insts.bin $1.mlir > $1.log 2>&1 || { echo AIECCFAIL $1; tail -c 2000 $1.log; return 14; }
}
TILES="1 2 4 8 16 32"
pids=()
for T in $TILES; do
  build t${T}_base 2048 2048 2048 2 $T & pids+=($!)
  build t${T}_cw 4096 2048 2048 2 $T & pids+=($!)
  build t${T}_aw 2048 4096 2048 2 $T & pids+=($!)
  build t${T}_ow 2048 2048 4096 2 $T & pids+=($!)
  build t${T}_rw 2048 2048 2048 4 $T & pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait $p || rc=$?; done
[ $rc = 0 ] || exit $rc
# L1 alignment audit: 512-bit vector accesses silently round the address DOWN to 64 B, so every aie.buffer must be 64 B aligned
python - <<PY || { echo "ALIGNMENT AUDIT FAILED"; exit 16; }
import re, sys
s = open("t1_base.mlir.prj/input_with_addresses.mlir").read()
bufs = re.findall(r'aie.buffer\(%tile_\d+_\d+\) \{address = (\d+) : i32.*?sym_name = "([^"]+)"', s)
bad = [(n, a) for a, n in bufs if int(a) % 64]
print("alignment audit:", len(bufs), "buffers,", "FAIL %s" % bad[:4] if bad else "all 64B aligned")
sys.exit(1 if bad else 0)
PY
mv t1_base.xclbin "$OUT/kernel.xclbin" || exit 15
for T in $TILES; do for v in base cw aw ow rw; do mv t${T}_$v.insts.bin "$OUT/insts_t${T}_$v.bin" || exit 15; done; done
cp t1_base.meta "$OUT/meta.txt"
cd "$OUT" && rm -rf work
echo "OK prefill kernel"
