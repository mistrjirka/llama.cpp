#!/usr/bin/env python3
"""Summarize GPU kernels inside driver NVTX intervals. Kernel sums are NOT wall time."""
import collections, json, pathlib, sqlite3, sys
R=pathlib.Path('/workspace/oai-qwen38-pp-lab/results/nsight-parallel-0907')
stem=sys.argv[1] if len(sys.argv)>1 else 'parallel'
c=sqlite3.connect(R/(stem+'.sqlite'));c.row_factory=sqlite3.Row
strings=dict(c.execute('select id,value from StringIds'))
gpus={r['id']:r['name'] for r in c.execute('select id,name from TARGET_INFO_GPU')}
phases=[dict(r) for r in c.execute("select start,end,coalesce(text,s.value) as name from NVTX_EVENTS n left join StringIds s on n.textId=s.id where n.end is not null and (n.text like '%/PP' or n.text like '%/TG') order by start")]
def union_ns(intervals):
    t=0;last=None
    for a,b in sorted(intervals):
        if last is None:last=[a,b]
        elif a<=last[1]:last[1]=max(last[1],b)
        else:t+=last[1]-last[0];last=[a,b]
    return t+(0 if last is None else last[1]-last[0])
result=[]
for ph in phases:
    start,end=ph['start'],ph['end'];dur=end-start
    ks=list(c.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL where start>=? and end<=?',(start,end)))
    ms=list(c.execute('select * from CUPTI_ACTIVITY_KIND_MEMCPY where start>=? and end<=?',(start,end)))
    rows=collections.defaultdict(lambda:{'count':0,'total_ms':0.,'registers':set(),'blocks':set(),'shared_bytes':set()})
    for k in ks:
        name=strings[k['demangledName']];key=(k['deviceId'],name)
        r=rows[key];r['count']+=1;r['total_ms']+=(k['end']-k['start'])/1e6
        r['registers'].add(k['registersPerThread']);r['blocks'].add(k['blockX']*k['blockY']*k['blockZ']);r['shared_bytes'].add(k['staticSharedMemory']+k['dynamicSharedMemory'])
    top=[]
    for (gpu,name),r in sorted(rows.items(),key=lambda x:-x[1]['total_ms']):
        top.append({'gpu':gpus[gpu],'gpu_id':gpu,'name':name,**{k:sorted(v) if isinstance(v,set) else v for k,v in r.items()}})
    intervals={g:[] for g in gpus}
    for k in ks+ms:intervals[k['deviceId']].append((k['start'],k['end']))
    busy={g:union_ns(xs) for g,xs in intervals.items()}
    allbusy=union_ns([x for xs in intervals.values() for x in xs]);both=sum(busy.values())-allbusy
    apis=collections.defaultdict(lambda:[0,0.])
    for k in c.execute('select start,end,nameId from CUPTI_ACTIVITY_KIND_RUNTIME where start>=? and end<=?',(start,end)):
        r=apis[strings[k['nameId']]];r[0]+=1;r[1]+=(k['end']-k['start'])/1e6
    copies=collections.defaultdict(lambda:[0,0.,0])
    for k in ms:
        r=copies[(k['deviceId'],k['copyKind'])];r[0]+=1;r[1]+=(k['end']-k['start'])/1e6;r[2]+=k['bytes']
    row={'phase':ph['name'],'wall_ms':dur/1e6,'kernel_ms':sum(x['total_ms'] for x in top),'kernel_count':len(ks),'gpu_busy_pct':{gpus[g]:100*t/dur for g,t in busy.items()},'both_gpu_active_pct':100*both/dur,'neither_gpu_active_pct':100*(dur-allbusy)/dur,'top_kernels':top,'cuda_apis':[{'name':k,'calls':v[0],'summed_cpu_api_ms':v[1]} for k,v in sorted(apis.items(),key=lambda x:-x[1][1])],'copies':[{'gpu':gpus[g],'kind':kind,'calls':v[0],'summed_ms':v[1],'bytes':v[2]} for (g,kind),v in copies.items()]}
    result.append(row)
    print('\nPHASE',ph['name'],'wall_ms',round(row['wall_ms'],2),'kernel_sum_ms',round(row['kernel_ms'],2),'busy',row['gpu_busy_pct'],'both',round(row['both_gpu_active_pct'],2),'neither',round(row['neither_gpu_active_pct'],2))
    for k in top[:9]:print(round(k['total_ms'],2),k['gpu'],k['count'],k['name'][:175])
    print('API',row['cuda_apis'][:4]);print('COPIES',row['copies'])
(R/(stem+'-summary.json')).write_text(json.dumps(result,indent=2))
