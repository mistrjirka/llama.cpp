#!/usr/bin/env python3
import importlib.util,pathlib,json,time,threading,concurrent.futures as cf,statistics
import os
R=pathlib.Path(os.environ.get('PARALLEL_RESULTS_DIR','/workspace/oai-qwen38-pp-lab/results/parallel-experiments-0907'))
O=pathlib.Path(os.environ.get('PARALLEL_FIXTURE_DIR',str(R.parent/'parallel-research-0907')))
z=importlib.util.spec_from_file_location('s',O/'sweep.py');s=importlib.util.module_from_spec(z);z.loader.exec_module(s)
s.PORT=32488;s.BIN=str(R/'candidate-bin/llama-server');s.ENV['LD_LIBRARY_PATH']=str(R/'candidate-bin')
rows=[];expected=None
for run,arm in enumerate(['off','on','on','off']):
 s.ENV.pop('GGML_CUDA_VOLTA_Q8_MULTI',None)
 if arm=='on':s.ENV['GGML_CUDA_VOLTA_Q8_MULTI']='1'
 p,l=s.start(3,f'../parallel-experiments-0907/q8-{run}-{arm}')
 try:
  for rep in range(2):
   s.restore();barrier=threading.Barrier(4);start=time.perf_counter()
   with cf.ThreadPoolExecutor(max_workers=4) as ex: rr=list(ex.map(lambda i:s.comp(i,s.PROMPTS[i],128,barrier),range(4)))
   wall=time.perf_counter()-start
   for r in rr:
    assert r['timings']['cache_n']==100000 and r['timings']['prompt_n']==1000,r
    r.pop('tokens',None)
   sha=[x['sha'] for x in rr]
   if expected is None:expected=sha
   row={'arm':arm,'run':run,'warmup':rep==0,'wall_s':wall,'whole_turn_tps':512/wall,'matching_sha':sha==expected,'mean_request_pp':statistics.mean(x['timings']['prompt_per_second'] for x in rr),'mean_request_tg':statistics.mean(x['timings']['predicted_per_second'] for x in rr),'rows':rr}
   rows.append(row);(R/'q8-results.json').write_text(json.dumps(rows,indent=2))
   print('Q8',arm,rep,round(wall,3),'PP',round(row['mean_request_pp'],2),'TG',round(row['mean_request_tg'],2),'same',row['matching_sha'],flush=True)
 finally:s.stop(p,l)
print('DONE',flush=True)
