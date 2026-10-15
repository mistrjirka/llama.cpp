#!/bin/bash
# Build the UNIFIED NPU decode kernel (lab stage5 ABI: qku.cc + genu.py) for ggml-xdna2. One xclbin serves every (qtype, K <= 12288, N).
# Only the instruction stream depends on three host values (chain_words, a_words, g), each linear in exactly one set of words, so the
# xclbin is built once together with the base stream and one stream per varied parameter; the backend derives the stream of a dispatch.
# Called by the kernel cache (build at load). Does not touch the NPU.
#
# usage: build_kernel.sh <mlir-aie-env-dir> <outdir>
# writes <outdir>/kernel.xclbin, insts_base.bin (2048,2048,2), insts_cw.bin (4096,2048,2), insts_aw.bin (2048,4096,2), insts_g.bin (2048,2048,4)
# and meta.txt; intermediates go to <outdir>/work (removed on success).
export LC_ALL=C
H=$(cd "$(dirname "$0")" && pwd)
TOOL=$1; OUT=$2
. "$H/env.sh" "$TOOL" || exit 10
mkdir -p "$OUT/work" && cd "$OUT/work" || exit 11
S=1024
$PEANOCC -DQK_S=$S -c "$H/qku.cc" -o k.o || { echo CCFAIL; exit 12; }
build() {   # name chain_words a_words g
  python "$H/genu.py" --obj k.o --S $S --stack 9216 --depth 8 --chain_words $2 --a_words $3 --g $4 > $1.mlir 2> $1.meta || { echo GENFAIL $1; cat $1.meta; return 13; }
  if [ $1 = base ]; then XB="--get-xclbin --xclbin-name=$1.xclbin --dump-intermediates"; else XB=""; fi
  aiecc --no-xchesscc --peano=$PEANO_INSTALL_DIR $XB --get-npu-insts --npu-insts-name=$1.insts.bin $1.mlir > $1.log 2>&1 || { echo AIECCFAIL $1; tail -c 2000 $1.log; return 14; }
}
build base 2048 2048 2 & p0=$!
build cw 4096 2048 2 & p1=$!
build aw 2048 4096 2 & p2=$!
build g 2048 2048 4 & p3=$!
rc=0; for p in $p0 $p1 $p2 $p3; do wait $p || rc=$?; done
[ $rc = 0 ] || exit $rc
# L1 alignment audit: 512-bit vector accesses silently round the address DOWN to 64 B, so every aie.buffer must be 64 B aligned
python - <<PY || { echo "ALIGNMENT AUDIT FAILED"; exit 16; }
import re, sys
s = open("base.mlir.prj/input_with_addresses.mlir").read()
bufs = re.findall(r'aie.buffer\(%tile_\d+_\d+\) \{address = (\d+) : i32.*?sym_name = "([^"]+)"', s)
bad = [(n, a) for a, n in bufs if int(a) % 64]
print("alignment audit:", len(bufs), "buffers,", "FAIL %s" % bad[:4] if bad else "all 64B aligned")
sys.exit(1 if bad else 0)
PY
mv base.xclbin "$OUT/kernel.xclbin" || exit 15
mv base.insts.bin "$OUT/insts_base.bin" && mv cw.insts.bin "$OUT/insts_cw.bin" && mv aw.insts.bin "$OUT/insts_aw.bin" && mv g.insts.bin "$OUT/insts_g.bin" || exit 15
cp base.meta "$OUT/meta.txt"
cd "$OUT" && rm -rf work
echo "OK unified kernel S=$S"
