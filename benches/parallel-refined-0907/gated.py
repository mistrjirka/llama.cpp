#!/usr/bin/env python3
"""ABBA trials of shape-gated cooperative attention, normal restore and unchanged sampling."""
import concurrent.futures as cf,importlib.util,json,pathlib,statistics,threading,time
import os
R=pathlib.Path(os.environ.get('REFINED_RESULTS_DIR','/workspace/oai-qwen38-pp-lab/results/parallel-refined-0907'));O=R.parent/'parallel-research-0907'
z=importlib.util.spec_from_file_location('s',pathlib.Path(__file__).resolve().parent/'fixture.py');s=importlib.util.module_from_spec(z);z.loader.exec_module(s)
s.PORT=32525;s.OUT=R;s.BIN=str(R/'gated-bin/llama-server');s.ENV['LD_LIBRARY_PATH']=str(R/'gated-bin')
f=json.loads((R/'real-fixture.json').read_text());prompts=f['prompts'];rows=[]
for concurrency,depth in [(1,3),(4,0),(4,3)]:
 for block,arm in enumerate(['off','on','on','off']):
  for k in list(s.ENV):
   if k.startswith('LLAMA_EXPERIMENT_') or k.startswith('GGML_CUDA_VOLTA_Q8_MULTI') or k in ['GGML_CUDA_VOLTA_Q8_REFINED','GGML_CUDA_VOLTA_Q8_REFINED_AUTO','GGML_CUDA_VOLTA_Q8_COOP','LLAMA_ARG_BACKEND_SAMPLING']:s.ENV.pop(k)
  if arm=='on':s.ENV.update({'GGML_CUDA_VOLTA_Q8_MULTI':'1','GGML_CUDA_VOLTA_Q8_MULTI_PACK':'4','GGML_CUDA_VOLTA_Q8_REFINED':'1','GGML_CUDA_VOLTA_Q8_REFINED_AUTO':'1','GGML_CUDA_VOLTA_Q8_COOP':'1'})
  p,l=s.start(depth,f'gated-c{concurrency}-d{depth}-{block}-{arm}')
  try:
   for rep in range(3):
    for i in range(4):s.req(f'/slots/{i}?action=erase',{})
    for i in range(4):s.req(f'/slots/{i}?action=restore',{'filename':f'real100k-{i}.bin'})
    b=threading.Barrier(concurrency);t=time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=concurrency) as ex:rr=list(ex.map(lambda i:s.comp(i,prompts[i],128,b),range(concurrency)))
    elapsed=time.perf_counter()-t
    for i,r in enumerate(rr):
     assert r['timings']['cache_n']==100000 and r['timings']['predicted_n']==128,r['timings']
    row={'concurrency':concurrency,'depth':depth,'arm':arm,'block':block,'rep':rep,'warmup':rep==0,'wall_s':elapsed,'mean_tg':statistics.mean(r['timings']['predicted_per_second'] for r in rr),'aggregate_turn_tps':concurrency*128/elapsed,'rows':rr}
    rows.append(row);(R/'gated-results.json').write_text(json.dumps(rows,indent=2))
    print('gated',concurrency,depth,arm,rep,round(elapsed,4),round(row['mean_tg'],3),flush=True)
  finally:s.stop(p,l)
summary=[]
for c,d in [(1,3),(4,0),(4,3)]:
 out={'concurrency':c,'depth':d}
 for arm in ['off','on']:
  rs=[r for r in rows if r['concurrency']==c and r['depth']==d and r['arm']==arm and not r['warmup']]
  out[arm]={'n':len(rs),'wall_mean':statistics.mean(x['wall_s'] for x in rs),'wall_min':min(x['wall_s'] for x in rs),'wall_max':max(x['wall_s'] for x in rs),'tg_mean':statistics.mean(x['mean_tg'] for x in rs)}
 out['speedup']=out['off']['wall_mean']/out['on']['wall_mean'];summary.append(out)
(R/'gated-summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
