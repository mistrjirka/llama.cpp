#!/usr/bin/env python3
"""Opt-in serving controller screen; identical model/KV/snapshots for all arms."""
import importlib.util,pathlib,json,time,threading,concurrent.futures as cf,urllib.request,hashlib,statistics
import os
R=pathlib.Path(os.environ.get('PARALLEL_RESULTS_DIR','/workspace/oai-qwen38-pp-lab/results/parallel-experiments-0907'))
O=pathlib.Path(os.environ.get('PARALLEL_FIXTURE_DIR',str(R.parent/'parallel-research-0907')))
z=importlib.util.spec_from_file_location('s',O/'sweep.py');s=importlib.util.module_from_spec(z);z.loader.exec_module(s)
s.PORT=32487;s.BIN=str(R/'initial-prototype-bin/llama-server');s.ENV['LD_LIBRARY_PATH']=str(R/'initial-prototype-bin')
all_rows=[]
def summary_stream(i,prompt,n,ready):
 body={'prompt':prompt,'id_slot':i,'cache_prompt':True,'n_predict':n,'ignore_eos':True,'temperature':0.0,'seed':1234,'return_tokens':True,'stream':True}
 request=urllib.request.Request(f'http://127.0.0.1:{s.PORT}/completion',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 start=time.perf_counter();times=[];text=[];final={}
 with urllib.request.urlopen(request,timeout=180) as response:
  for line in response:
   if not line.startswith(b'data: '):continue
   line=line[6:].strip()
   if line==b'[DONE]':continue
   r=json.loads(line)
   if 'error' in r:raise RuntimeError(r)
   if r.get('content'):
    times.append(time.perf_counter());text.append(r['content']);ready.set()
   if r.get('stop'):final=r
 gaps=[b-a for a,b in zip(times,times[1:]) if b-a>=0.001]
 return {'slot':i,'wall_s':time.perf_counter()-start,'ttft_s':times[0]-start if times else None,'max_visible_gap_s':max(gaps,default=0),'p95_visible_gap_s':sorted(gaps)[min(len(gaps)-1,int(.95*len(gaps)))] if gaps else 0,'timings':final.get('timings',{}),'sha':hashlib.sha256(''.join(text).encode()).hexdigest()}
for index,(label,flags) in enumerate([
 ('baseline',{}),('adaptive',{'LLAMA_EXPERIMENT_ADAPTIVE_MTP':'1'}),
 ('budget100',{'LLAMA_EXPERIMENT_PREFILL_MS':'100'}),
 ('combined',{'LLAMA_EXPERIMENT_ADAPTIVE_MTP':'1','LLAMA_EXPERIMENT_PREFILL_MS':'100'}),('baseline-repeat',{})]):
 for key in ['LLAMA_EXPERIMENT_ADAPTIVE_MTP','LLAMA_EXPERIMENT_PREFILL_MS']:s.ENV.pop(key,None)
 s.ENV.update(flags);p,l=s.start(3,f'../parallel-experiments-0907/controller-{label}')
 try:
  if label in ['baseline','adaptive','baseline-repeat']:
   s.restore();barrier=threading.Barrier(4);start=time.perf_counter()
   with cf.ThreadPoolExecutor(max_workers=4) as ex:rr=list(ex.map(lambda i:s.comp(i,s.PROMPTS[i],256,barrier),range(4)))
   wall=time.perf_counter()-start
   for r in rr:r.pop('tokens',None)
   row={'label':label,'workload':'uniform','wall_s':wall,'rows':rr};all_rows.append(row)
   (R/'controller-results.json').write_text(json.dumps(all_rows,indent=2));print('UNIFORM',label,wall,[round(r['timings']['predicted_per_second'],2) for r in rr],flush=True)
  s.restore();ev=[threading.Event() for _ in range(3)];start=time.perf_counter()
  with cf.ThreadPoolExecutor(max_workers=4) as ex:
   fs=[ex.submit(summary_stream,i,s.H[i]+s.SEQ[80000:80032],192,ev[i]) for i in range(3)]
   for e in ev:assert e.wait(60),'decode did not begin'
   heavy=ex.submit(s.comp,3,s.H[3]+s.SEQ[50000:58000],16)
   rr=[f.result() for f in fs];other=heavy.result();other.pop('tokens',None)
  wall=time.perf_counter()-start
  row={'label':label,'workload':'three-decode-plus-8k-prefill','wall_s':wall,'decoders':rr,'prefill':other};all_rows.append(row)
  (R/'controller-results.json').write_text(json.dumps(all_rows,indent=2));print('MIXED',label,wall,'maxgap',[round(r['max_visible_gap_s'],3) for r in rr],'p95',[round(r['p95_visible_gap_s'],3) for r in rr],flush=True)
 finally:s.stop(p,l)
print('DONE',flush=True)
