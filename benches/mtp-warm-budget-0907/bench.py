#!/usr/bin/env python3
"""True off vs request-local warm0/MTP1/MTP3, identical MTP3 contexts for all capped arms.
Mirror process order, excluded warmup, no profiler or background compilation.
"""
import os
import concurrent.futures as cf,hashlib,importlib.util,json,pathlib,statistics,threading,time
R=pathlib.Path(os.environ.get('MTP_BUDGET_RESULTS_DIR', '/workspace/oai-qwen38-pp-lab/results/mtp-warm-budget-0907'));F=R.parent/'parallel-refined-0907';B=R/'candidate-bin'
z=importlib.util.spec_from_file_location('fixture',R.parent/'parallel-research-0907/sweep.py');s=importlib.util.module_from_spec(z);z.loader.exec_module(s)
s.PORT=32592;s.OUT=R;s.BIN=str(B/'llama-server');s.ENV['LD_LIBRARY_PATH']=str(B)
for k in list(s.ENV):
 if k.startswith('LLAMA_EXPERIMENT_') or k.startswith('GGML_CUDA_VOLTA_Q8_MULTI') or k.startswith('GGML_CUDA_VOLTA_Q8_REF') or k in ['GGML_CUDA_VOLTA_Q8_COOP','GGML_CUDA_VOLTA_Q8_PV_F32','LD_PRELOAD','LLAMA_ARG_BACKEND_SAMPLING']:s.ENV.pop(k)
s.ENV.update({'LLAMA_EXPERIMENT_MTP_KV_ONLY':'1','LLAMA_EXPERIMENT_MTP_REQUEST_BUDGET':'1'})
f=json.loads((F/'real-fixture.json').read_text());rows=[]
for block,arm in enumerate(['off','cap0','cap1','cap3','cap3','cap1','cap0','off']):
 cap=None if arm=='off' else int(arm[-1])
 p,l=s.start(0 if arm=='off' else 3,'bench-'+str(block)+'-'+arm)
 try:
  maps=pathlib.Path(f'/proc/{p.pid}/maps').read_text();assert str(B/'libllama-server-impl.so') in maps
  for rep in range(3):
   for i in range(4):s.req(f'/slots/{i}?action=erase',{})
   for i in range(4):assert s.req(f'/slots/{i}?action=restore',{'filename':f'real100k-{i}.bin'})['n_restored']==100000
   barrier=threading.Barrier(4)
   def work(i):
    body={'prompt':f['prompts'][i],'id_slot':i,'n_predict':128,'return_tokens':True,'cache_prompt':True,'ignore_eos':True,'temperature':0.0,'seed':1234}
    if cap is not None:body['speculative_n_max']=cap
    barrier.wait();r=s.req('/completion',body)
    assert len(r['tokens'])==128 and r['timings']['predicted_n']==128
    assert r['timings']['cache_n']==100000 and r['timings']['prompt_n']==len(f['prompts'][i])-100000,r['timings']
    if cap is not None:assert r['generation_settings']['speculative_n_max']==cap
    if cap==0:assert r['timings'].get('draft_n',0)==0
    return {'slot':i,'timings':r['timings'],'sha':hashlib.sha256(','.join(map(str,r['tokens'])).encode()).hexdigest(),'tokens':r['tokens']}
   t=time.perf_counter()
   with cf.ThreadPoolExecutor(max_workers=4) as ex:out=list(ex.map(work,range(4)))
   wall=time.perf_counter()-t
   row={'arm':arm,'block':block,'rep':rep,'warmup':rep==0,'wall_s':wall,'mean_tg':statistics.mean(x['timings']['predicted_per_second'] for x in out),'turn_output_tps':512/wall,'requests':out,'argv':s.args(0 if arm=='off' else 3)}
   rows.append(row);(R/'bench-raw.json').write_text(json.dumps(rows,indent=2))
   print(arm,rep,round(wall,4),round(row['mean_tg'],3),flush=True)
 finally:s.stop(p,l)
summary={}
for arm in ['off','cap0','cap1','cap3']:
 group=[r for r in rows if r['arm']==arm and not r['warmup']]
 summary[arm]={'n':len(group),**{k:{'mean':statistics.mean(r[k] for r in group),'min':min(r[k] for r in group),'max':max(r[k] for r in group)} for k in ['wall_s','mean_tg','turn_output_tps']}}
(R/'bench-summary.json').write_text(json.dumps(summary,indent=2));print('SUMMARY',json.dumps(summary),flush=True)
