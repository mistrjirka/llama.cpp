#!/usr/bin/env python3
"""Use device execution intervals, not CPU NVTX duration, to measure GPU activity."""
import argparse,bisect,collections,html,json,pathlib,sqlite3
R=pathlib.Path('/workspace/oai-qwen38-pp-lab/results/mtp-gantt-0907')
p=argparse.ArgumentParser();p.add_argument('name',nargs='?',default='baseline');a=p.parse_args()
c=sqlite3.connect(R/(a.name+'.sqlite'));c.row_factory=sqlite3.Row
strings=dict(c.execute('select id,value from StringIds'));gpu=dict(c.execute('select id,name from TARGET_INFO_GPU'))
def merge(xs):
 out=[]
 for s,e in sorted(xs):
  if e<=s:continue
  if out and s<=out[-1][1]:out[-1][1]=max(out[-1][1],e)
  else:out.append([s,e])
 return out
def total(xs):return sum(e-s for s,e in merge(xs))
def intersect(xs,ys):
 x,y=merge(xs),merge(ys);i=j=0;v=0
 while i<len(x) and j<len(y):
  v+=max(0,min(x[i][1],y[j][1])-max(x[i][0],y[j][0]))
  if x[i][1]<y[j][1]:i+=1
  else:j+=1
 return v
allout=[]
for ph in c.execute("select text,start,end from NVTX_EVENTS where text like ? and text like '%/TG'",(a.name+'/%',)).fetchall():
 start,end=ph['start'],ph['end'];duration=end-start
 nv=[dict(r) for r in c.execute('select start,end,globalTid,text from NVTX_EVENTS where start>=? and end<=? and end is not null',(start,end)) if r['text']]
 ranges=collections.defaultdict(list)
 for r in nv:
  if r['text'].startswith(('decode/','sample/')):ranges[r['globalTid']].append(r)
 for t in ranges:ranges[t].sort(key=lambda r:r['start'])
 indices={t:[r['start'] for r in v] for t,v in ranges.items()}
 def tag(t,tm):
  rows=ranges.get(t,[]);i=bisect.bisect_right(indices.get(t,[]),tm)-1
  if i>=0 and tm<=rows[i]['end']:return rows[i]['text'].split('/')[1]
  return 'host-other'
 runtime=[dict(r) for r in c.execute('select * from CUPTI_ACTIVITY_KIND_RUNTIME where start>=? and end<=?',(start,end))]
 corr={r['correlationId']:tag(r['globalTid'],r['start']) for r in runtime}
 events=[];unknown=0;stage=collections.defaultdict(lambda:[0,0.0]);active=collections.defaultdict(list)
 for r in c.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL where start>=? and end<=?',(start,end)):
  name=strings[r['demangledName']];kind=corr.get(r['correlationId'],'unknown');unknown+=kind=='unknown'
  fam='attention' if 'flash_attn' in name else 'KV dequant' if 'dequantize_block_q8_0_f16' in name else 'matmul' if 'mul_mat' in name or 'gemm' in name else 'other'
  stage[kind][0]+=1;stage[kind][1]+=(r['end']-r['start'])/1e6
  active[r['deviceId']].append((r['start'],r['end']))
  events.append({'lane':gpu[r['deviceId']],'s':(r['start']-start)/1e6,'e':(r['end']-start)/1e6,'kind':kind,'detail':fam,'name':name})
 copies=[]
 for r in c.execute('select * from CUPTI_ACTIVITY_KIND_MEMCPY where start>=? and end<=?',(start,end)):
  active[r['deviceId']].append((r['start'],r['end']));copies.append(dict(r))
  events.append({'lane':gpu[r['deviceId']],'s':(r['start']-start)/1e6,'e':(r['end']-start)/1e6,'kind':'copy','detail':f"kind {r['copyKind']} bytes {r['bytes']}",'name':'CUDA transfer'})
 union=merge([r for rows in active.values() for r in rows]);busy=total(union);idle=duration-busy
 cpu={}
 for label,prefix in [('Draft orchestration','MTP/draft'),('KV tail removal','KV/remove/'),('Prompt copies','CPU/copy_prompt'),('Target readback/sampling','sample/target'),('Draft readback/sampling','sample/draft'),('Cache refresh','MTP/refresh')]:
  rs=[r for r in nv if r['text'].startswith(prefix)]
  intervals=[(r['start'],r['end']) for r in rs];time=total(intervals);overlap=intersect(intervals,union)
  cpu[label]={'calls':len(rs),'wall_ms_union':time/1e6,'gpu_idle_overlap_ms':(time-overlap)/1e6}
  for r in rs:events.append({'lane':label,'s':(r['start']-start)/1e6,'e':(r['end']-start)/1e6,'kind':'cpu','detail':r['text'],'name':r['text']})
 waits=[r for r in runtime if 'Synchronize' in strings[r['nameId']]]
 wtime=total([(r['start'],r['end']) for r in waits]);wover=intersect([(r['start'],r['end']) for r in waits],union)
 for r in waits:events.append({'lane':'CUDA synchronization','s':(r['start']-start)/1e6,'e':(r['end']-start)/1e6,'kind':'wait','detail':strings[r['nameId']],'name':strings[r['nameId']]})
 copy_summary=collections.defaultdict(lambda:[0,0,0.0])
 for r in copies:
  z=copy_summary[r['copyKind']];z[0]+=1;z[1]+=r['bytes'];z[2]+=(r['end']-r['start'])/1e6
 out={'phase':ph['text'],'profiled_wall_ms':duration/1e6,'gpu_busy_any_ms':busy/1e6,'neither_gpu_active_ms':idle/1e6,'neither_gpu_active_pct':idle/duration*100,'simultaneous_gpu_ms':(sum(total(v) for v in active.values())-busy)/1e6,'gpu_busy_ms':{gpu[k]:total(v)/1e6 for k,v in active.items()},'cpu_scopes':cpu,'sync':{'calls':len(waits),'wall_ms_union':wtime/1e6,'gpu_idle_overlap_ms':(wtime-wover)/1e6},'kernels_by_launch_stage':{k:{'calls':v[0],'summed_gpu_ms':v[1]} for k,v in stage.items()},'copies':{str(k):{'calls':v[0],'bytes':v[1],'summed_gpu_ms':v[2]} for k,v in copy_summary.items()},'unmatched_kernels':unknown}
 allout.append(out)
 mode=ph['text'].split('/')[1];(R/(a.name+'-'+mode+'-events.json')).write_text(json.dumps(events,separators=(',',':')))
 print(json.dumps(out,indent=2),flush=True)
 # Exact steady-state timeline of three draft rounds; all positions are true trace times.
 drafts=sorted([r for r in nv if r['text']=='MTP/draft'],key=lambda r:r['start'])
 if len(drafts)>=8:lo=(drafts[4]['start']-start)/1e6;hi=(drafts[7]['start']-start)/1e6
 else:lo=duration/1e6*.2;hi=lo+min(250,duration/1e6*.4)
 lanes=['Draft orchestration','Prompt copies','KV tail removal','Target readback/sampling','Draft readback/sampling','Cache refresh','CUDA synchronization']+list(gpu.values())
 width=1500;left=235;right=30;rowheight=46;height=170+len(lanes)*rowheight
 # SVG uses actual event spans; no invented parallelism or stage widths.
 palette={'target':'#4e79a7','draft':'#f28e2b','refresh':'#59a14f','copy':'#b07aa1','cpu':'#808080','wait':'#b8b8b8','host-other':'#76b7b2','unknown':'#e15759'}
 esc=html.escape
 svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>','<style>text{font-family:Arial,sans-serif;fill:#222}</style>',f'<text x="22" y="32" font-size="22" font-weight="bold">{esc(a.name)} — {esc(mode)} — measured CPU/GPU timeline</text>',f'<text x="22" y="56" font-size="14">100k cached code tokens/agent • Q6/Q5 target • Q8 KV • V100 + 2080 Ti • profiled, not benchmark latency</text>']
 for tick in range(7):
  tm=lo+(hi-lo)*tick/6;x=left+(width-left-right)*tick/6
  svg += [f'<line x1="{x:.2f}" y1="95" x2="{x:.2f}" y2="{height-62}" stroke="#ddd"/>',f'<text x="{x:.2f}" y="86" text-anchor="middle" font-size="12">{tm-lo:.1f} ms</text>']
 for i,lane in enumerate(lanes):
  y=103+i*rowheight
  svg.append(f'<text x="{left-12}" y="{y+19}" text-anchor="end" font-size="13">{esc(lane)}</text>')
  svg.append(f'<line x1="{left}" y1="{y+28}" x2="{width-right}" y2="{y+28}" stroke="#eee"/>')
  for e in events:
   if e['lane']!=lane or e['e']<=lo or e['s']>=hi:continue
   x=left+(max(e['s'],lo)-lo)/(hi-lo)*(width-left-right);w=(min(e['e'],hi)-max(e['s'],lo))/(hi-lo)*(width-left-right)
   svg.append(f'<rect x="{x:.3f}" y="{y}" width="{max(w,0.15):.3f}" height="27" fill="{palette[e["kind"]]}"><title>{esc(e["detail"])} | {e["e"]-e["s"]:.4f} ms</title></rect>')
 for j,(kind,label) in enumerate([('target','Target / verification'),('draft','Draft forward'),('refresh','K/V refresh'),('copy','CUDA transfer'),('cpu','CPU scope / waits')]):
  x=left+j*225;svg.extend([f'<rect x="{x}" y="{height-44}" width="16" height="14" fill="{palette[kind]}"/>',f'<text x="{x+23}" y="{height-32}" font-size="12">{label}</text>'])
 svg.append(f'<text x="22" y="{height-8}" font-size="12">GPU bars are actual kernels/copies; blank GPU lanes mean no recorded GPU work. CPU scopes can include waits. Zoom starts {lo:.2f} ms into captured turn.</text></svg>')
 (R/(a.name+'-'+mode+'-gantt.svg')).write_text('\n'.join(svg))
(R/(a.name+'-timeline-summary.json')).write_text(json.dumps(allout,indent=2))
