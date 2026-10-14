#!/bin/bash
# Build one NPU GEMV kernel (stage3 ABI) for ggml-xdna2: the xclbin is specific to (qtype, K, M) but serves any N % 512 == 0;
# only the instruction stream depends on N, so it is produced for a few reference N and patched by the backend.
# Called by the kernel cache (build at load). Does not touch the NPU.
#
# usage: build_kernel.sh <mlir-aie-env-dir> <outdir> <qt: 4|5|6|8> <K> <M> <N_ref0> [<N_ref1> ...]
# writes <outdir>/kernel.xclbin (from N_ref0), <outdir>/insts_n<N>.bin for every N, <outdir>/meta.txt;
# intermediates go to <outdir>/work (removed on success).
# Adapted from the lab's stage3/build3.sh.
export LC_ALL=C
H=$(cd "$(dirname "$0")" && pwd)
TOOL=$1; OUT=$2; QT=$3; K=$4; M=$5; shift 5
. "$H/env.sh" "$TOOL" || exit 10
mkdir -p "$OUT/work" && cd "$OUT/work" || exit 11
STACK=5120; [ "$QT" = 4 ] && STACK=4096
DEPTH=3; [ "$M" -ge 2 ] && DEPTH=2          # lab: depth 2 for M >= 2 (memtile BD budget)
n=k
$PEANOCC -DQK_T=$QT -DQK_K=$K -DQK_M=$M -c "$H/qk3.cc" -o $n.o || { echo CCFAIL; exit 12; }
first=1
for N in "$@"; do
  python "$H/gen3.py" --qt $QT --N $N --K $K --M $M --depth $DEPTH --stack $STACK --obj $n.o > n$N.mlir 2> n$N.meta || { echo GENFAIL; cat n$N.meta; exit 13; }
  if [ $first = 1 ]; then XB="--get-xclbin --xclbin-name=n$N.xclbin"; else XB=""; fi
  aiecc --no-xchesscc --peano=$PEANO_INSTALL_DIR $XB --get-npu-insts --npu-insts-name=n$N.insts.bin n$N.mlir > n$N.log 2>&1 || { echo AIECCFAIL; tail -c 2000 n$N.log; exit 14; }
  if [ $first = 1 ]; then mv n$N.xclbin "$OUT/kernel.xclbin" || exit 15; cp n$N.meta "$OUT/meta.txt"; first=0; fi
  mv n$N.insts.bin "$OUT/insts_n$N.bin" || exit 15
  rm -rf n$N.mlir.prj
done
cd "$OUT" && rm -rf work
echo "OK qt=$QT K=$K M=$M N=$*"
