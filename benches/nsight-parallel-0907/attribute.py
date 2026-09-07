#!/usr/bin/env python3
"""Attribute CUDA launches to interposed MTP/target NVTX scopes, scoped by TG window."""
import bisect, collections, json, pathlib, sqlite3
R=pathlib.Path('/workspace/oai-qwen38-pp-lab/results/nsight-parallel-0907');c=sqlite3.connect(R/'attributed.sqlite');c.row_factory=sqlite3.Row
strings=dict(c.execute('select id,value from StringIds'));gpus=dict(c.execute('select id,name from TARGET_INFO_GPU'))
start,end=c.execute("select start,end from NVTX_EVENTS where text='attributed-mtp3/TG'").fetchone()
ranges=collections.defaultdict(list);sums=collections.defaultdict(lambda:[0,0.])
for r in c.execute("select start,end,globalTid,text from NVTX_EVENTS where start>=? and end<=? and (text like 'MTP/%' or text like 'decode/%') order by start",(start,end)):
    name=r['text'];sums[name][0]+=1;sums[name][1]+=(r['end']-r['start'])/1e6
    if name.startswith('decode/'):
        comp=name.split('/')[1];ranges[r['globalTid']].append((r['start'],r['end'],comp))
indices={t:[v[0] for v in vs] for t,vs in ranges.items()}
def tag(tid,tm):
    vs=ranges.get(tid,[]);ix=bisect.bisect_right(indices.get(tid,[]),tm)-1
    return vs[ix][2] if ix>=0 and tm<=vs[ix][1] else 'outside-decode'
api={};api_groups=collections.defaultdict(lambda:[0,0.])
for r in c.execute('select * from CUPTI_ACTIVITY_KIND_RUNTIME where start>=? and end<=?',(start,end)):
    comp=tag(r['globalTid'],r['start']);api[r['correlationId']]=comp
    k=(comp,strings[r['nameId']]);api_groups[k][0]+=1;api_groups[k][1]+=(r['end']-r['start'])/1e6
kernel_groups=collections.defaultdict(lambda:[0,0.]);copy_groups=collections.defaultdict(lambda:[0,0.,0]);missing=0
for r in c.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL where start>=? and end<=?',(start,end)):
    comp=api.get(r['correlationId'],'unknown');missing+=comp=='unknown'
    name=strings[r['demangledName']]
    fam='attention' if 'flash_attn' in name or 'ggml_q8multi' in name else 'q8-to-f16' if 'dequantize_block_q8_0_f16' in name else 'matmul' if 'mul_mat' in name or 'gemm' in name else 'other'
    k=(comp,gpus[r['deviceId']],fam);kernel_groups[k][0]+=1;kernel_groups[k][1]+=(r['end']-r['start'])/1e6
for r in c.execute('select * from CUPTI_ACTIVITY_KIND_MEMCPY where start>=? and end<=?',(start,end)):
    comp=api.get(r['correlationId'],'unknown');k=(comp,gpus[r['deviceId']],r['copyKind']);copy_groups[k][0]+=1;copy_groups[k][1]+=(r['end']-r['start'])/1e6;copy_groups[k][2]+=r['bytes']
out={'wall_ms':(end-start)/1e6,'unmatched_kernels':missing,'cpu_scopes':{k:{'calls':v[0],'summed_ms':v[1]} for k,v in sums.items()},'gpu_kernels':[{'component':k[0],'gpu':k[1],'family':k[2],'calls':v[0],'summed_ms':v[1]} for k,v in kernel_groups.items()], 'api':[{'component':k[0],'name':k[1],'calls':v[0],'summed_ms':v[1]} for k,v in sorted(api_groups.items(),key=lambda x:-x[1][1])], 'copies':[{'component':k[0],'gpu':k[1],'kind':k[2],'calls':v[0],'summed_ms':v[1],'bytes':v[2]} for k,v in copy_groups.items()]}
(R/'mtp-attribution.json').write_text(json.dumps(out,indent=2))
print('CPU SCOPES',json.dumps(out['cpu_scopes'],indent=2));print('UNMATCHED KERNELS',missing)
comps=collections.defaultdict(float)
for x in out['gpu_kernels']:comps[x['component']]+=x['summed_ms']
print('GPU TIME BY COMPONENT',dict(comps));print('KERNEL GROUPS',json.dumps(out['gpu_kernels'],indent=2));print('API',json.dumps(out['api'][:15],indent=2));print('COPIES',json.dumps(out['copies'],indent=2))
