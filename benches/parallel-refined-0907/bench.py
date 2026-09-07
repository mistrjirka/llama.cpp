#!/usr/bin/env python3
"""Symmetrically bracketed ablations on identical real-code snapshots, no profiler overhead."""
import argparse,concurrent.futures as cf,importlib.util,json,pathlib,statistics,threading,time,random
import os
R=pathlib.Path(os.environ.get('REFINED_RESULTS_DIR','/workspace/oai-qwen38-pp-lab/results/parallel-refined-0907'));O=R.parent/'parallel-research-0907'
z=importlib.util.spec_from_file_location('s',pathlib.Path(__file__).resolve().parent/'fixture.py');s=importlib.util.module_from_spec(z);z.loader.exec_module(s)
s.PORT=32521;s.OUT=R
args=argparse.ArgumentParser();args.add_argument('--suite',default='host');args.add_argument('--repeats',type=int,default=2);a=args.parse_args()
fixture=json.loads((R/'real-fixture.json').read_text());prompts=fixture['prompts']
flags={
 'off':{},
 'sample':{'LLAMA_EXPERIMENT_SAMPLING_VIEW':'1'},
 'hidden':{'LLAMA_EXPERIMENT_MTP_BULK_HIDDEN':'1'},
 'both':{'LLAMA_EXPERIMENT_SAMPLING_VIEW':'1','LLAMA_EXPERIMENT_MTP_BULK_HIDDEN':'1'},
 'q8old':{'GGML_CUDA_VOLTA_Q8_MULTI':'1','GGML_CUDA_VOLTA_Q8_MULTI_PACK':'4'},
 'q8new':{'GGML_CUDA_VOLTA_Q8_MULTI':'1','GGML_CUDA_VOLTA_Q8_MULTI_PACK':'4','GGML_CUDA_VOLTA_Q8_REFINED':'1'}
}
if a.suite=='host':arms=['off','sample','both','hidden','hidden','both','sample','off'];binary='host-bin'
else:arms=['off','q8old','q8new','q8new','q8old','off'];binary='candidate-bin'
s.BIN=str(R/binary/'llama-server');s.ENV['LD_LIBRARY_PATH']=str(R/binary)
rows=[]
for block,arm in enumerate(arms):
 for k in list(s.ENV):
  if k.startswith('LLAMA_EXPERIMENT_') or k.startswith('GGML_CUDA_VOLTA_Q8_MULTI') or k=='GGML_CUDA_VOLTA_Q8_REFINED':s.ENV.pop(k)
 s.ENV.update(flags[arm]);p,log=s.start(3,f'{a.suite}-{block}-{arm}')
 try:
  maps=pathlib.Path(f'/proc/{p.pid}/maps').read_text()
  assert str(R/binary/'libllama') in maps,'wrong loaded library'
  for rep in range(a.repeats+1):
   for i in range(4):s.req(f'/slots/{i}?action=erase',{})
   for i in range(4):
    d=s.req(f'/slots/{i}?action=restore',{'filename':f'real100k-{i}.bin'});assert d['n_restored']==100000
   barrier=threading.Barrier(4);start=time.perf_counter()
   with cf.ThreadPoolExecutor(max_workers=4) as ex:results=list(ex.map(lambda i:s.comp(i,prompts[i],128,barrier),range(4)))
   wall=time.perf_counter()-start
   for i,r in enumerate(results):
    assert r['timings']['cache_n']==100000,r['timings']
    assert r['timings']['prompt_n']==len(prompts[i])-100000,r['timings']
    assert r['timings']['predicted_n']==128,r['timings']
   row={'arm':arm,'block':block,'rep':rep,'warmup':rep==0,'wall_s':wall,'aggregate_output_tps':512/wall,'mean_tg':statistics.mean(r['timings']['predicted_per_second'] for r in results),'results':results}
   rows.append(row);(R/f'{a.suite}-results.json').write_text(json.dumps(rows,indent=2))
   print(a.suite,arm,rep,'wall',round(wall,4),'TG',round(row['mean_tg'],3),'sha',[r['sha'][:8] for r in results],flush=True)
 finally:s.stop(p,log)
summary={}
for arm in set(arms):
 rr=[r for r in rows if r['arm']==arm and not r['warmup']]
 summary[arm]={k:{'mean':statistics.mean(r[k] for r in rr),'min':min(r[k] for r in rr),'max':max(r[k] for r in rr)} for k in ['wall_s','mean_tg','aggregate_output_tps']}
 summary[arm]['n']=len(rr)
(R/f'{a.suite}-summary.json').write_text(json.dumps(summary,indent=2));print('SUMMARY',json.dumps(summary),flush=True)
