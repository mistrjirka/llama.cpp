#!/usr/bin/env python3
import os
"""Attribute kernel launches to target/draft/cache refresh scopes. Profiler only."""
import bisect, collections, json, pathlib, sqlite3
R = pathlib.Path(os.environ.get('MTP_REFRESH_RESULTS_DIR', '/workspace/oai-qwen38-pp-lab/results/mtp-kv-refresh-0907'))
c = sqlite3.connect(R / 'refresh.sqlite'); c.row_factory = sqlite3.Row
strings = dict(c.execute('select id,value from StringIds')); out = []
for phase in ['ordinary/TG', 'cache-only/TG']:
    windows = c.execute('select start,end from NVTX_EVENTS where text=?', (phase,)).fetchall()
    assert len(windows) == 1, (phase, len(windows))
    start, end = windows[0]
    ranges = collections.defaultdict(list)
    for r in c.execute("select start,end,globalTid,text from NVTX_EVENTS where start>=? and end<=? and text like 'decode/%' order by start", (start, end)):
        ranges[r['globalTid']].append((r['start'], r['end'], r['text'].split('/')[1]))
    indices = {t: [v[0] for v in rs] for t, rs in ranges.items()}
    def tag(tid, tm):
        rs = ranges.get(tid, []); i = bisect.bisect_right(indices.get(tid, []), tm) - 1
        return rs[i][2] if i >= 0 and tm <= rs[i][1] else 'outside'
    api = {r['correlationId']: tag(r['globalTid'], r['start']) for r in c.execute('select * from CUPTI_ACTIVITY_KIND_RUNTIME where start>=? and end<=?', (start, end))}
    groups = collections.defaultdict(lambda: [0, 0.]); detail = collections.defaultdict(lambda: [0, 0.]); unknown = 0
    for r in c.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL where start>=? and end<=?', (start, end)):
        comp = api.get(r['correlationId'], 'unknown'); unknown += comp == 'unknown'; name = strings[r['demangledName']]
        fam = 'attention' if 'flash_attn' in name else 'q8-to-f16' if 'dequantize_block_q8_0_f16' in name else 'matmul' if 'mul_mat' in name or 'gemm' in name else 'other'
        ms = (r['end'] - r['start']) / 1e6
        groups[(comp, fam)][0] += 1; groups[(comp, fam)][1] += ms
        if comp == 'process': detail[name][0] += 1; detail[name][1] += ms
    n = c.execute("select count(*) from NVTX_EVENTS where start>=? and end<=? and text like 'decode/process/%'", (start, end)).fetchone()[0]
    assert n > 0 and unknown == 0, (phase, n, unknown)
    refresh_ms = sum(v[1] for k, v in groups.items() if k[0] == 'process')
    out.append({'phase': phase, 'profiled_wall_ms': (end-start)/1e6, 'unmatched_kernels': unknown,
                'refresh_calls': n, 'refresh_gpu_ms_per_call': refresh_ms/n,
                'kernels': [{'stage': k[0], 'family': k[1], 'calls': v[0], 'gpu_ms': v[1]} for k, v in groups.items()],
                'refresh_kernels': [{'name': k, 'calls': v[0], 'gpu_ms': v[1]} for k, v in sorted(detail.items(), key=lambda x: -x[1][1])]})
    print(phase, 'refresh calls', n, 'GPU ms/call', refresh_ms/n)
(R/'profile-summary.json').write_text(json.dumps(out, indent=2)+'\n')
