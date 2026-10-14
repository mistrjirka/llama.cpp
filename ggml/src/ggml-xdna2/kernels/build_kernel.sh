#!/bin/bash
# Build one NPU GEMV kernel (xclbin + instruction stream). Called by ggml-xdna2 (kernel cache, build at load).
# usage: build_kernel.sh <mlir-aie-env-dir> <outdir> <qt: 4|6 (Q4_K|Q6_K)> <N> <K> <M>
# Writes <outdir>/kernel.xclbin and <outdir>/insts.bin; intermediates go to <outdir>/work (removed on success).
# Adapted from the lab's stage2/build_qk.sh. Does not touch the NPU.
export LC_ALL=C
H=$(cd "$(dirname "$0")" && pwd)
TOOL=$1; OUT=$2; QT=$3; N=$4; K=$5; M=$6
. "$H/env.sh" "$TOOL" || exit 10
mkdir -p "$OUT/work" && cd "$OUT/work" || exit 11
DEPTH=3; [ "$M" -ge 2 ] && DEPTH=2          # stage2: depth 2 for M >= 2 (memtile BD budget)
n=k
$PEANOCC -DQK_T=$QT -DQK_K=$K -DQK_RPC=$((N/32)) -DQK_M=$M -c "$H/qk.cc" -o $n.o || { echo CCFAIL; exit 12; }
python "$H/gen_qk.py" --qt $QT --N $N --K $K --M $M --depth $DEPTH --obj $n.o > $n.mlir 2> $n.meta || { echo GENFAIL; cat $n.meta; exit 13; }
aiecc --no-xchesscc --peano=$PEANO_INSTALL_DIR --get-xclbin --xclbin-name=$n.xclbin --get-npu-insts --npu-insts-name=$n.insts.bin $n.mlir > $n.log 2>&1 || { echo AIECCFAIL; tail -c 2000 $n.log; exit 14; }
mv $n.xclbin "$OUT/kernel.xclbin" && mv $n.insts.bin "$OUT/insts.bin" && mv $n.meta "$OUT/meta.txt" || exit 15
cd "$OUT" && rm -rf work
echo "OK qt=$QT N=$N K=$K M=$M"
