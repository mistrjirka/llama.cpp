#!/usr/bin/env python3
import concurrent.futures as cf,contextlib,ctypes,importlib.util,json,pathlib,threading,time
import os
R=pathlib.Path(os.environ.get('REFINED_RESULTS_DIR','/workspace/oai-qwen38-pp-lab/results/parallel-refined-0907'));O=R.parent/'parallel-research-0907'
z=importlib.util.spec_from_file_location('s',pathlib.Path(__file__).resolve().parent/'fixture.py');s=importlib.util.module_from_spec(z);z.loader.exec_module(s)
s.PORT=32523;s.OUT=R;s.BIN=str(R/'host-bin/llama-server');s.ENV['LD_LIBRARY_PATH']=str(R/'host-bin')
markers=ctypes.CDLL(str(R.parent/'nsight-parallel-0907/markers.so'));markers.mark_push.argtypes=[ctypes.c_char_p]
@contextlib.contextmanager
def phase(n):
 markers.mark_push(n.encode())
 try:yield
 finally:markers.mark_pop()
f=json.loads((R/'real-fixture.json').read_text());prompts=f['prompts'];rows=[]
def batch(prompts,n):
 b=threading.Barrier(4)
 with cf.ThreadPoolExecutor(max_workers=4) as ex:return list(ex.map(lambda i:s.comp(i,prompts[i],n,b),range(4)))
for arm in ['off','both','gpu']:
 for k in ['LLAMA_EXPERIMENT_SAMPLING_VIEW','LLAMA_EXPERIMENT_MTP_BULK_HIDDEN','LLAMA_ARG_BACKEND_SAMPLING']:s.ENV.pop(k,None)
 if arm=='both':s.ENV.update({'LLAMA_EXPERIMENT_SAMPLING_VIEW':'1','LLAMA_EXPERIMENT_MTP_BULK_HIDDEN':'1'})
 if arm=='gpu':s.ENV['LLAMA_ARG_BACKEND_SAMPLING']='1'
 p,log=s.start(3,f'profile-host-{arm}')
 try:
  for i in range(4):s.req(f'/slots/{i}?action=restore',{'filename':f'real100k-{i}.bin'})
  pp=batch(prompts,1)
  qs=[q+r['tokens'] for q,r in zip(prompts,pp)]
  with phase(arm+'/TG'):
   t=time.perf_counter();rr=batch(qs,128);wall=time.perf_counter()-t
  rows.append({'arm':arm,'wall_s':wall,'results':rr});(R/'profile-host-results.json').write_text(json.dumps(rows,indent=2))
  print('profile',arm,round(wall,3),flush=True)
 finally:s.stop(p,log)
