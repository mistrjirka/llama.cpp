#!/usr/bin/env bash
set -u

MODEL=${MODEL:-/models/Qwen3.5-122B-A10B-GGUF/UD-Q3_K_XL/Qwen3.5-122B-A10B-UD-Q3_K_XL-00001-of-00003.gguf}
PROMPT=${PROMPT:-'Write a minimal C++20 program that prints hello and return only code.'}
OUTDIR=${OUTDIR:-vertical-cache-capacity-profile}
TOKENS=${TOKENS:-16}
BUDGETS=("${@:-baseline 5000 6000 7000 7600 8000}")

# Expand the default string into separate arguments when no explicit budgets were supplied.
if [ "$#" -eq 0 ]; then
    BUDGETS=(baseline 5000 6000 7000 7600 8000)
else
    BUDGETS=("$@")
fi

mkdir -p "$OUTDIR"
printf 'case\texit\tdecode_tps\tprompt_tps\twall_s\tallocated_mib\tcompute_mib\tselected_layers\tslots\tmin_free_mib\tmax_used_mib\tmean_gpu_util\tsha256\n' > "$OUTDIR/results.tsv"

run_case() {
    local case_name=$1
    local out="$OUTDIR/$case_name.out"
    local err="$OUTDIR/$case_name.err"
    local mon="$OUTDIR/$case_name.nvidia.csv"
    rm -f "$out" "$err" "$mon"

    nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu \
        --format=csv,noheader,nounits -lms 200 > "$mon" 2>/dev/null &
    local monitor_pid=$!

    local start end rc=0
    start=$(date +%s%N)

    local cache_env=()
    if [ "$case_name" != baseline ]; then
        cache_env=(
            "GGML_EXPERT_CACHE_MIB=$case_name"
            GGML_EXPERT_CACHE_PROFILE=1
            GGML_MOE_DYNAMIC_SPLIT_SLOTS=auto
            GGML_MOE_DYNAMIC_THRESHOLD=999
            GGML_MOE_DYNAMIC_MIN_HOT_ROUTES=4
            GGML_MOE_DYNAMIC_MAX_ADMISSIONS_PER_TOKEN=0
            GGML_MOE_DYNAMIC_SKIP_HOT_BRANCH=1
        )
    fi

    env -u GGML_EXPERT_CACHE_MIB -u GGML_EXPERT_CACHE_RESERVE_MIB \
        -u GGML_EXPERT_CACHE_PROFILE -u GGML_EXPERT_CACHE_VMM -u GGML_EXPERT_CACHE_MANAGED \
        -u GGML_MOE_DYNAMIC_SPLIT_SLOTS -u GGML_MOE_DYNAMIC_SPLIT_LAYERS \
        -u GGML_MOE_DYNAMIC_THRESHOLD -u GGML_MOE_DYNAMIC_MIN_HOT_ROUTES \
        -u GGML_MOE_DYNAMIC_MAX_ADMISSIONS_PER_TOKEN -u GGML_MOE_DYNAMIC_SKIP_HOT_BRANCH \
        -u GGML_MOE_DYNAMIC_TRACE \
        CUDA_VISIBLE_DEVICES=0 GGML_CUDA_DISABLE_GRAPHS=1 "${cache_env[@]}" \
        timeout 1800s build-v100/bin/llama-completion \
        -m "$MODEL" -p "$PROMPT" -n "$TOKENS" -c 256 -b 2048 -ub 512 -fa on -dev CUDA0 \
        -t 40 -tb 48 -fit on -fitt 8192 -fitc 512 \
        -s 1 --temp 0 --single-turn --no-conversation --no-display-prompt --simple-io -v \
        > "$out" 2> "$err" || rc=$?

    end=$(date +%s%N)
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true

    python3 - "$case_name" "$rc" "$start" "$end" "$err" "$out" "$mon" <<'PY' | tee -a "$OUTDIR/results.tsv"
import hashlib
import re
import statistics
import sys

case_name, rc, start, end, err_path, out_path, mon_path = sys.argv[1:]
text = open(err_path, errors='replace').read()

def last(pattern, default=''):
    matches = re.findall(pattern, text)
    return matches[-1] if matches else default

samples = []
for line in open(mon_path, errors='replace'):
    fields = [field.strip() for field in line.split(',')]
    if len(fields) != 4:
        continue
    try:
        index, used, free, util = map(float, fields)
    except ValueError:
        continue
    if used >= 1024:
        samples.append((index, used, free, util))

min_free = min((sample[2] for sample in samples), default=0.0)
max_used = max((sample[1] for sample in samples), default=0.0)
mean_util = statistics.mean(sample[3] for sample in samples) if samples else 0.0
slots = sorted(set(re.findall(r'moe-dynamic-vram-plan:.*?slots=(\d+)/256', text)), key=int)
sha = hashlib.sha256(open(out_path, 'rb').read()).hexdigest()

row = [
    case_name,
    rc,
    last(r'(?<!prompt )eval time\s*=.*?\(.*?([0-9.]+) tokens per second\)'),
    last(r'prompt eval time\s*=.*?\(.*?([0-9.]+) tokens per second\)'),
    f'{(int(end) - int(start)) / 1e9:.3f}',
    last(r'expert-cache: entries=\d+ allocated=([0-9.]+) MiB', '0'),
    last(r'CUDA0 compute buffer size =\s*([0-9.]+) MiB', '0'),
    last(r'moe-dynamic-vram-plan:.*?selected-layers=(\d+)', '0'),
    ','.join(slots),
    f'{min_free:.0f}',
    f'{max_used:.0f}',
    f'{mean_util:.2f}',
    sha,
]
print('\t'.join(map(str, row)))
PY
}

for budget in "${BUDGETS[@]}"; do
    run_case "$budget"
done

cat "$OUTDIR/results.tsv"
