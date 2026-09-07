#!/usr/bin/env python3
import concurrent.futures as cf,importlib.util,json,pathlib,threading,time
import os
R=pathlib.Path(os.environ.get('REFINED_RESULTS_DIR','/workspace/oai-qwen38-pp-lab/results/parallel-refined-0907'));O=R.parent/'parallel-research-0907'
z=importlib.util.spec_from_file_location('s',pathlib.Path(__file__).resolve().parent/'fixture.py');s=importlib.util.module_from_spec(z);z.loader.exec_module(s)
s.PORT=32522;s.OUT=R;s.BIN=str(R/'host-bin/llama-server');s.ENV['LD_LIBRARY_PATH']=str(R/'host-bin')
fixture=json.loads((R/'real-fixture.json').read_text());prompts=fixture['prompts']
rows=[];forced=None
for arm in ['off','both','off']:
 for k in ['LLAMA_EXPERIMENT_SAMPLING_VIEW','LLAMA_EXPERIMENT_MTP_BULK_HIDDEN','LD_PRELOAD']:s.ENV.pop(k,None)
 if arm=='both':s.ENV.update({'LLAMA_EXPERIMENT_SAMPLING_VIEW':'1','LLAMA_EXPERIMENT_MTP_BULK_HIDDEN':'1','LD_PRELOAD':str(R/'verify_views.so')})
 p,log=s.start(3,f'verify-{len(rows)}-{arm}')
 try:
  for i in range(4):s.req(f'/slots/{i}?action=restore',{'filename':f'real100k-{i}.bin'})
  vals=[];seq=[[] for _ in range(4)]
  for step in range(12):
   batch=[]
   for i in range(4):
    x=s.req('/completion',{'prompt':prompts[i]+seq[i],'id_slot':i,'cache_prompt':True,'n_predict':1,'temperature':0.0,'seed':1234,'n_probs':20,'post_sampling_probs':False,'return_tokens':True,'ignore_eos':True})
    batch.append({'token':x['tokens'][0],'probs':x.get('completion_probabilities')})
   vals.append(batch)
   for i in range(4):seq[i].append(batch[i]['token'] if forced is None else forced[step][i])
  if forced is None:forced=[[v['token'] for v in b] for b in vals]
  rows.append({'arm':arm,'values':vals})
  if arm=='both':
   barrier=threading.Barrier(4)
   with cf.ThreadPoolExecutor(max_workers=4) as ex:
    rr=list(ex.map(lambda i:s.comp(i,prompts[i]+seq[i],64,barrier),range(4)))
   # Save complete target/draft/carry companions, restore into empty slots, continue.
   for i in range(4):s.req(f'/slots/{i}?action=save',{'filename':f'views-roundtrip-{i}.bin'})
   for i in range(4):s.req(f'/slots/{i}?action=erase',{})
   for i in range(4):s.req(f'/slots/{i}?action=restore',{'filename':f'views-roundtrip-{i}.bin'})
   s.comp(0,prompts[0]+seq[0]+rr[0]['tokens'],16)
 finally:s.stop(p,log)
report={'A_B_token_mismatch':sum(x['token']!=y['token'] for b,c in zip(rows[0]['values'],rows[1]['values']) for x,y in zip(b,c)),'A_A_token_mismatch':sum(x['token']!=y['token'] for b,c in zip(rows[0]['values'],rows[2]['values']) for x,y in zip(b,c)),'A_B_probability_records_identical':rows[0]['values']==rows[1]['values'],'steps':12,'sequences':4}
(R/'views-validation.json').write_text(json.dumps({'summary':report,'raw':rows},indent=2));print(json.dumps(report),flush=True)
