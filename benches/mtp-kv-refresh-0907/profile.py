#!/usr/bin/env python3
import os
import concurrent.futures as cf,ctypes,importlib.util,json,pathlib,threading,time
R=pathlib.Path(os.environ.get('MTP_REFRESH_RESULTS_DIR', '/workspace/oai-qwen38-pp-lab/results/mtp-kv-refresh-0907'));F=R.parent/'parallel-refined-0907';B=R/'candidate-bin'
z=importlib.util.spec_from_file_location('fixture',R.parent/'parallel-research-0907/sweep.py');s=importlib.util.module_from_spec(z);z.loader.exec_module(s)
s.PORT=32575;s.OUT=R;s.BIN=str(B/'llama-server');s.ENV['LD_LIBRARY_PATH']=str(B)
for k in list(s.ENV):
 if k.startswith('LLAMA_EXPERIMENT_') or k.startswith('GGML_CUDA_VOLTA_Q8_MULTI') or k.startswith('GGML_CUDA_VOLTA_Q8_REF') or k=='LD_PRELOAD':s.ENV.pop(k)
s.ENV['LD_PRELOAD']=str(R/'profile-markers.so')
f=json.loads((F/'real-fixture.json').read_text());mark=ctypes.CDLL(str(R.parent/'nsight-parallel-0907/markers.so'));mark.mark_push.argtypes=[ctypes.c_char_p];rows=[]
def batch(q,n):
 b=threading.Barrier(4)
 with cf.ThreadPoolExecutor(max_workers=4) as ex:return list(ex.map(lambda i:s.comp(i,q[i],n,b),range(4)))
for arm in ['ordinary','cache-only']:
 s.ENV.pop('LLAMA_EXPERIMENT_MTP_KV_ONLY',None)
 if arm=='cache-only':s.ENV['LLAMA_EXPERIMENT_MTP_KV_ONLY']='1'
 p,log=s.start(3,'profile-'+arm)
 try:
  for i in range(4):s.req(f'/slots/{i}?action=restore',{'filename':f'real100k-{i}.bin'})
  pp=batch(f['prompts'],16);q=[p+x['tokens'] for p,x in zip(f['prompts'],pp)]
  mark.mark_push((arm+'/TG').encode())
  try:
   t=time.perf_counter();out=batch(q,64);wall=time.perf_counter()-t
  finally:mark.mark_pop()
  rows.append({'arm':arm,'wall_s':wall,'requests':out});(R/'profile-raw.json').write_text(json.dumps(rows,indent=2));print('PROFILE',arm,wall,flush=True)
 finally:s.stop(p,log)
