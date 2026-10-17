#!/bin/bash
# Build the NPU flash-attention kernel (fa.cc + genfa.py) for ggml-xdna2: one xclbin; per supported pair count R (1 2 4 8 16) three reference
# instruction streams: base (nb 2, kvmax 4096), nb (nb 4: affine), kv (kvmax 8192: linear).  Does not touch the NPU.
#
# usage: build_fa.sh <mlir-aie-env-dir> <outdir> [D (128|256, default 256)] [kv heads (4|8, default 4)]
# writes <outdir>/kernel.xclbin, insts_r<R>_{base,nb,kv}.bin, meta.txt
export LC_ALL=C
H=$(cd "$(dirname "$0")" && pwd)
TOOL=$1; OUT=$2; D=${3:-256}; NKV=${4:-4}
. "$H/env.sh" "$TOOL" || exit 10
mkdir -p "$OUT/work" && cd "$OUT/work" || exit 11
STACK=4096
$PEANOCC -fstack-usage -DD=$D -I"$H" -c "$H/fa.cc" -o k.o || { echo CCFAIL; exit 12; }
# stack guard: the deepest kernel frame plus the IRON core main must fit the core stack (an overflow silently corrupts the fifo buffers)
SU=$(awk '{print $2}' k.su | sort -n | tail -1)
[ $((SU + 1024)) -le $STACK ] || { echo "STACK GUARD: kernel frame $SU B + 1 KB > stack $STACK"; exit 17; }
build() {   # name pairs nb kvmax
  python "$H/genfa.py" --obj k.o --pairs $2 --nb $3 --kvmax $4 --D $D --nkv $NKV --stack $STACK > $1.mlir 2> $1.meta || { echo GENFAIL $1; cat $1.meta; return 13; }
  if [ $1 = r16_base ]; then XB="--get-xclbin --xclbin-name=$1.xclbin --dump-intermediates"; else XB=""; fi
  aiecc --no-xchesscc --peano=$PEANO_INSTALL_DIR $XB --get-npu-insts --npu-insts-name=$1.insts.bin $1.mlir > $1.log 2>&1 || { echo AIECCFAIL $1; tail -c 2000 $1.log; return 14; }
}
pids=()
for R in 1 2 4 8 16; do
  build r${R}_base $R 2 4096 & pids+=($!)
  build r${R}_nb $R 4 4096 & pids+=($!)
  build r${R}_kv $R 2 8192 & pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait $p || rc=$?; done
[ $rc = 0 ] || exit $rc
# L1 alignment audit: 512-bit vector accesses silently round the address DOWN to 64 B (the only unaligned buffers allowed are the
# compiler's scalar loop-counter buffers, memref<Nxi32> with N < 16)
python - <<PY || { echo "ALIGNMENT AUDIT FAILED"; exit 16; }
import re, sys
s = open("r16_base.mlir.prj/input_with_addresses.mlir").read()
bufs = re.findall(r'aie.buffer\(%tile_\d+_\d+\) \{address = (\d+) : i32.*?sym_name = "([^"]+)"\} : memref<(\d+)x', s)
bad = [(n, a) for a, n, cnt in bufs if int(a) % 64 and int(cnt) >= 16]
print("alignment audit:", len(bufs), "buffers,", "FAIL %s" % bad[:4] if bad else "ok")
sys.exit(1 if bad else 0)
PY
mv r16_base.xclbin "$OUT/kernel.xclbin" || exit 15
for R in 1 2 4 8 16; do for v in base nb kv; do mv r${R}_$v.insts.bin "$OUT/insts_r${R}_$v.bin" || exit 15; done; done
cp r16_base.meta "$OUT/meta.txt"
cd "$OUT" && rm -rf work
echo "OK attention kernel"
