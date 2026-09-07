#!/usr/bin/env python3
"""Independent verification groups vs scheduler buffering, on fixed real-code KV."""
from __future__ import annotations
import argparse, concurrent.futures as cf, hashlib, importlib.util, json, os, pathlib, statistics, subprocess, threading, time
R=pathlib.Path(os.environ.get('GROUP_RESULTS_DIR', '/workspace/oai-qwen38-pp-lab/results/group-pipeline-0907')); F=R.parent/'parallel-refined-0907'
P=pathlib.Path('/workspace/llama-parallel-group-pipeline-0907')
z=importlib.util.spec_from_file_location('fixture',R.parent/'parallel-research-0907/sweep.py');s=importlib.util.module_from_spec(z);z.loader.exec_module(s)
s.PORT=32538;s.OUT=R;s.BIN=str(R/'validated128-bin/llama-server');s.ENV['LD_LIBRARY_PATH']=str(R/'validated128-bin')
for k in list(s.ENV):
 if k.startswith('LLAMA_EXPERIMENT_') or k.startswith('GGML_CUDA_VOLTA_Q8_MULTI') or k.startswith('GGML_CUDA_VOLTA_Q8_REF') or k in ['LLAMA_ARG_BACKEND_SAMPLING','GGML_CUDA_VOLTA_Q8_COOP']:s.ENV.pop(k)
f=json.loads((F/'real-fixture.json').read_text());prompts=f['prompts']
for i in range(4):
 for ext in ['', '.draft','.spec']:
  link=R/f'real100k-{i}.bin{ext}'
  if not link.exists():link.symlink_to(F/link.name)
original_args=s.args
copies=1;pool=1400000

def arguments(depth: int) -> list[str]:
 a=original_args(depth)
 a[a.index('--pipeline-copies')+1]=str(copies)
 a[a.index('--ctx-size')+1]=str(pool)
 a[a.index('--ubatch-size')+1]='128'
 return a
s.args=arguments
ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['capacity','bench','teacher','profile']);a=ap.parse_args()
if a.mode=='capacity':
 copies=2
 p,log=s.start(3,'capacity128-copies2')
 try:
  slots=s.req('/slots');assert len(slots)==4
  (R/'capacity128.json').write_text(json.dumps({'slots':slots,'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.used,memory.free','--format=csv,noheader'],text=True)},indent=2))
  print('CAPACITY', (R/'capacity128.json').read_text(),flush=True)
 finally:s.stop(p,log)
if a.mode=='bench':
 rows=[];variants={'A':(0,1),'B':(2,1),'C':(0,2),'D':(2,2),'E':(1,2)}
 for block,arm in enumerate(['A','B','D','C','E','E','C','D','B','A']):
  limit,copies=variants[arm];s.ENV.pop('LLAMA_EXPERIMENT_VERIFY_GROUP_SEQS',None)
  s.ENV['LLAMA_EXPERIMENT_VERIFY_SCHED_DIAGNOSTIC']='1'
  if limit:s.ENV['LLAMA_EXPERIMENT_VERIFY_GROUP_SEQS']=str(limit)
  p,log=s.start(3,f'bench128-{block}-{arm}')
  try:
   assert str(R/'validated128-bin/libllama') in pathlib.Path(f'/proc/{p.pid}/maps').read_text()
   log.flush()
   if copies==2:
    assert 'retrying without pipeline' not in (R/f'bench128-{block}-{arm}.log').read_text(), 'pipeline disabled by allocator'
   for rep in range(3):
    for i in range(4):s.req(f'/slots/{i}?action=erase',{})
    for i in range(4):assert s.req(f'/slots/{i}?action=restore',{'filename':f'real100k-{i}.bin'})['n_restored']==100000
    barrier=threading.Barrier(4);t=time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=4) as ex:rr=list(ex.map(lambda i:s.comp(i,prompts[i],128,barrier),range(4)))
    wall=time.perf_counter()-t
    for i,r in enumerate(rr):
     assert r['timings']['cache_n']==100000 and r['timings']['prompt_n']==len(prompts[i])-100000
     assert r['timings']['predicted_n']==128
    row={'arm':arm,'groups':limit,'copies':copies,'block':block,'rep':rep,'warmup':rep==0,'wall_s':wall,'output_tps':512/wall,'mean_tg':statistics.mean(r['timings']['predicted_per_second'] for r in rr),'rows':rr}
    rows.append(row);(R/'bench128-results.json').write_text(json.dumps(rows,indent=2));print(arm,rep,round(wall,4),round(row['mean_tg'],3),flush=True)
  finally:s.stop(p,log)
 summary={}
 for arm in variants:
  group=[r for r in rows if r['arm']==arm and not r['warmup']]
  summary[arm]={'group_sequences':variants[arm][0],'pipeline_copies':variants[arm][1],'n':len(group)}
  for k in ['wall_s','output_tps','mean_tg']:
   v=[r[k] for r in group];summary[arm][k]={'mean':statistics.mean(v),'min':min(v),'max':max(v)}
 (R/'bench128-summary.json').write_text(json.dumps(summary,indent=2));print('SUMMARY',json.dumps(summary),flush=True)
if a.mode=='teacher':
 results=[]
 for arm,limit,copies in [('A',0,1),('C',0,2),('D',2,2),('E',1,2),('A2',0,1)]:
  env=s.ENV.copy();env.update({'GROUP_FIXTURE':str(R/'teacher-fixture.json'),'GROUP_SNAPSHOTS':str(R),'GROUP_LOGITS':str(R/f'logits128-{arm}.f32')})
  if limit:env.update({'LLAMA_EXPERIMENT_VERIFY_GROUP_SEQS':str(limit),'LLAMA_EXPERIMENT_VERIFY_GROUP_TRACE':'1'})
  args=arguments(3);args[0]=str(R/'teacher-groups')
  with (R/f'teacher128-{arm}.log').open('wb') as log:
   subprocess.run(args,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
  if arm!='A':
   result=json.loads(subprocess.check_output([str(R/'compare-logits'),str(R/'logits128-A.f32'),str(R/f'logits128-{arm}.f32')],text=True))
   results.append({'arm':arm,**result});print(arm,result,flush=True)
   (R/'teacher128-results.json').write_text(json.dumps(results,indent=2))

if a.mode=='profile':
 import ctypes
 markers=ctypes.CDLL(str(R.parent/'nsight-parallel-0907/markers.so'))
 markers.mark_push.argtypes=[ctypes.c_char_p]
 measurements=[]
 for arm,limit,copies in [('A',0,1),('B',2,1),('D',2,2),('E',1,2)]:
  s.ENV.pop('LLAMA_EXPERIMENT_VERIFY_GROUP_SEQS',None)
  s.ENV['LLAMA_EXPERIMENT_VERIFY_SCHED_DIAGNOSTIC']='1'
  if limit:s.ENV['LLAMA_EXPERIMENT_VERIFY_GROUP_SEQS']=str(limit)
  # Trace logging is only used in this diagnostic run, never in timings.
  if limit:s.ENV['LLAMA_EXPERIMENT_VERIFY_GROUP_TRACE']='1'
  p,log=s.start(3,f'profile128-{arm}')
  try:
   for i in range(4):s.req(f'/slots/{i}?action=restore',{'filename':f'real100k-{i}.bin'})
   barrier=threading.Barrier(4)
   with cf.ThreadPoolExecutor(max_workers=4) as ex:pp=list(ex.map(lambda i:s.comp(i,prompts[i],1,barrier),range(4)))
   qs=[prompts[i]+pp[i]['tokens'] for i in range(4)]
   barrier=threading.Barrier(4);markers.mark_push((arm+'/TG').encode());t=time.perf_counter()
   try:
    with cf.ThreadPoolExecutor(max_workers=4) as ex:rr=list(ex.map(lambda i:s.comp(i,qs[i],128,barrier),range(4)))
   finally:markers.mark_pop()
   measurements.append({'arm':arm,'wall_s':time.perf_counter()-t,'rows':rr})
   (R/'profile128-results.json').write_text(json.dumps(measurements,indent=2))
   print('PROFILE',arm,measurements[-1]['wall_s'],flush=True)
  finally:s.stop(p,log)
