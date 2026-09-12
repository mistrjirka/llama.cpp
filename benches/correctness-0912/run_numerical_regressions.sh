#!/usr/bin/env bash
set -euo pipefail
: "${BACKEND_TEST:?Set BACKEND_TEST to the test-backend-ops executable}"
: "${CUDA_VISIBLE_DEVICES:?Select one GPU by UUID}"
log=${1:-streamk-numerical.log}
filter='hsk=256,hsv=256,nh=2,nr23=\[(6|8),1\],kv=(65408|65536|65664),nb=(128|129|512),.*type_K=q8_0,type_V=q8_0'
"$BACKEND_TEST" test -b CUDA0 -o FLASH_ATTN_EXT -p "$filter" 2>&1 | tee "$log"
grep -q '10/10 tests passed' "$log"
