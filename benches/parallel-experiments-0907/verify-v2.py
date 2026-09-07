#!/usr/bin/env python3
import concurrent.futures as cf
import importlib.util, json, pathlib, threading, hashlib, time
ROOT=pathlib.Path('/workspace/oai-qwen38-pp-lab/results/prefix-first-0907')
OLD=pathlib.Path('/workspace/oai-qwen38-pp-lab/results/parallel-research-0907')
spec=importlib.util.spec_from_file_location('screen',OLD/'sweep.py');s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
s.PORT=32484
bins={'A':ROOT/'baseline-bin','B':pathlib.Path('/workspace/llama-multiagent-cache/build-sm70-75/bin')}
pr=json.loads((ROOT/'code-prompts.json').read_text());hs=pr['histories'];prompts=pr['prompts'];results=[]
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
for run,label in enumerate(['A','B','A']):
 s.BIN=str(bins[label]/'llama-server');s.ENV['LD_LIBRARY_PATH']=str(bins[label])
 p,log=s.start(3,f'../prefix-first-0907/verify-v2-{run}-{label}')
 try:
  for i in range(4):s.req(f'/slots/{i}?action=restore',{'filename':f'codebase{i}.bin'})
  states={}
  for i in range(4):
   name=f'canonical-{run}-{i}.bin';s.req(f'/slots/{i}?action=save',{'filename':name})
   states[str(i)]={suffix:digest(OLD/(name+suffix)) for suffix in ['', '.draft', '.spec']}
  barrier=threading.Barrier(4);t=time.perf_counter()
  with cf.ThreadPoolExecutor(max_workers=4) as ex:rr=list(ex.map(lambda i:s.comp(i,prompts[i],96,barrier),range(4)))
  wall=time.perf_counter()-t
  r={'run':run,'arm':label,'states':states,'requests':rr,'wall_s':wall};results.append(r)
  (ROOT/'verify-v2-raw.json').write_text(json.dumps(results,indent=2))
  print('V2',label,'wall',wall,'hashes',[x['sha'] for x in rr],flush=True)
 finally:s.stop(p,log)
summary={'candidate_states_match_baseline':results[0]['states']==results[1]['states'],'baseline_states_repeat':results[0]['states']==results[2]['states']}
for i in range(4):
 a,b,c=[r['requests'][i]['tokens'] for r in results]
 summary[str(i)]={'A_B_same':a==b,'A_A_same':a==c,'AB_first_diff':next((j for j,(x,y) in enumerate(zip(a,b)) if x!=y),None),'AA_first_diff':next((j for j,(x,y) in enumerate(zip(a,c)) if x!=y),None)}
(ROOT/'verify-v2-summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
