#!/usr/bin/env python3
import concurrent.futures as cf
import importlib.util,json,os,pathlib,signal,subprocess,threading,time,argparse
R=pathlib.Path(os.environ.get('MTP_RESULT_DIR', '/workspace/oai-qwen38-pp-lab/results/mtp-study-0907'));R.mkdir(parents=True,exist_ok=True)
F=R.parent/'parallel-research-0907'
z=importlib.util.spec_from_file_location('s',pathlib.Path(__file__).resolve().parent/'fixture.py');s=importlib.util.module_from_spec(z);z.loader.exec_module(s)
p=argparse.ArgumentParser();p.add_argument('--nsys',action='store_true');p.add_argument('--corpus',default='code',choices=['code','synthetic']);p.add_argument('--parallel',type=int,default=4);p.add_argument('--depth',type=int,default=3);p.add_argument('--tag',default='');a=p.parse_args()
s.PORT=32501
B=R.parent/'prefix-first-0907'/'baseline-bin'
s.BIN=str(B/'llama-server');s.ENV['LD_LIBRARY_PATH']=str(B)
label=f'{a.corpus}-p{a.parallel}-mtp{a.depth}'+('-nsys' if a.nsys else '')+a.tag
gate=R/(label+'.gate');gate.unlink(missing_ok=True)
s.ENV.update({'LD_PRELOAD':str(R/'profile.so'),'MTP_PROFILE_GATE':str(gate)})
serverargs=s.args(a.depth)
log=open(R/(label+'.log'),'wb')
cmd=serverargs
if a.nsys:
 cmd=['nsys','profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none','--cuda-graph-trace=node','--force-overwrite=true','--output',str(R/label),'--']+cmd
proc=subprocess.Popen(cmd,env=s.ENV,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
try:
 for _ in range(700):
  if proc.poll() is not None:raise RuntimeError('server died, '+label)
  try:
   if s.req('/health',timeout=.5).get('status')=='ok':break
  except Exception:pass
  time.sleep(.1)
 else:raise TimeoutError('startup')
 if a.corpus=='code':
  fixture=json.loads((R.parent/'prefix-first-0907'/'code-prompts.json').read_text())
  hs=fixture['histories'];prompts=fixture['prompts'];names=[f'codebase{i}.bin' for i in range(4)]
 else:
  hs=s.H;prompts=s.PROMPTS;names=[f'base{i}.bin' for i in range(4)]
 def restore():
  for i in range(4):s.req(f'/slots/{i}?action=erase',{})
  for i,name in enumerate(names):s.req(f'/slots/{i}?action=restore',{'filename':name})
 restore()
 # Use a separate request for warm-up, restore all histories before measurement.
 s.comp(0,prompts[0],16)
 restore()
 gate.touch()
 bar=threading.Barrier(a.parallel);t=time.perf_counter()
 with cf.ThreadPoolExecutor(max_workers=a.parallel) as ex:
  rows=list(ex.map(lambda i:s.comp(i,prompts[i],192,bar),range(a.parallel)))
 elapsed=time.perf_counter()-t;gate.unlink()
 for i,r in enumerate(rows):assert r['timings']['cache_n']==len(hs[i]),r['timings']
 out={'label':label,'wall_s':elapsed,'whole_turn_output_tps':sum(r['timings']['predicted_n'] for r in rows)/elapsed,'rows':rows,'profiled':a.nsys,'safe_baseline':'7571e2e17','weights':'AD-Q6_K-Q5_K','target_kv':'q8_0/q8_0','draft_kv':'q8_0/q8_0'}
 (R/(label+'.json')).write_text(json.dumps(out,indent=2));print(label,elapsed,[r['timings']['predicted_per_second'] for r in rows],flush=True)
finally:
 gate.unlink(missing_ok=True)
 if proc.poll() is None:
  os.killpg(proc.pid,signal.SIGTERM)
  try:proc.wait(80)
  except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
 log.close()
print('\n'.join(l for l in (R/(label+'.log')).read_text(errors='replace').splitlines() if 'MTP_PROFILE ' in l),flush=True)
