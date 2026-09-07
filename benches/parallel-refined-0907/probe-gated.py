#!/usr/bin/env python3
"""Fixed-input target-distribution probe. Four-token appends trigger exactly the gated shape.
Only top-128 logprobs are returned: coarse KL merges unreported tokens into OTHER,
so it is a lower bound rather than an estimate of full-vocabulary KL.
"""
import importlib.util,json,math,pathlib,statistics
import os
R=pathlib.Path(os.environ.get('REFINED_RESULTS_DIR','/workspace/oai-qwen38-pp-lab/results/parallel-refined-0907'));O=R.parent/'parallel-research-0907'
z=importlib.util.spec_from_file_location('s',pathlib.Path(__file__).resolve().parent/'fixture.py');s=importlib.util.module_from_spec(z);z.loader.exec_module(s)
s.PORT=32526;s.OUT=R;s.BIN=str(R/'gated-bin/llama-server');s.ENV['LD_LIBRARY_PATH']=str(R/'gated-bin')
f=json.loads((R/'real-fixture.json').read_text());hs=f['histories'];heldout=f['parent'][95000:95128];rows=[]
for arm in ['off','on','off']:
 for k in list(s.ENV):
  if k.startswith('LLAMA_EXPERIMENT_') or k.startswith('GGML_CUDA_VOLTA_Q8_') and any(x in k for x in ['MULTI','REFINED','COOP']):s.ENV.pop(k)
 if arm=='on':s.ENV.update({'GGML_CUDA_VOLTA_Q8_MULTI':'1','GGML_CUDA_VOLTA_Q8_MULTI_PACK':'4','GGML_CUDA_VOLTA_Q8_REFINED':'1','GGML_CUDA_VOLTA_Q8_REFINED_AUTO':'1','GGML_CUDA_VOLTA_Q8_COOP':'1'})
 p,l=s.start(0,f'probability-{len(rows)}-{arm}')
 try:
  for i in range(4):s.req(f'/slots/{i}?action=restore',{'filename':f'real100k-{i}.bin'})
  vals=[]
  for step in range(16):
   for i in range(4):
    prompt=hs[i]+heldout[:4*(step+1)]
    r=s.req('/completion',{'prompt':prompt,'id_slot':i,'cache_prompt':True,'n_predict':1,'temperature':0.0,'seed':1234,'n_probs':128,'post_sampling_probs':False,'return_tokens':True,'ignore_eos':True})
    assert r['timings']['prompt_n']==4,r['timings']
    vals.append({'slot':i,'step':step,'token':r['tokens'][0],'probs':r['completion_probabilities'][0]['top_logprobs']})
  rows.append({'arm':arm,'values':vals});(R/'gated-probabilities.json').write_text(json.dumps(rows,indent=2));print('probe',arm,'done',flush=True)
 finally:s.stop(p,l)
report={}
for j,label in [(1,'off_on'),(2,'off_off')]:
 disagreements=0;deltas=[];kls=[]
 for x,y in zip(rows[0]['values'],rows[j]['values']):
  disagreements+=x['token']!=y['token']
  px={z['id']:math.exp(z['logprob']) for z in x['probs']};py={z['id']:math.exp(z['logprob']) for z in y['probs']}
  common=px.keys()&py.keys();a=[px[k] for k in common];b=[py[k] for k in common]
  deltas.append(max(abs(p-q) for p,q in zip(a,b)))
  a.append(max(1e-30,1-sum(a)));b.append(max(1e-30,1-sum(b)))
  kls.append(sum(p*math.log(p/q) for p,q in zip(a,b)))
 report[label]={'n':len(rows[0]['values']),'greedy_disagreements':disagreements,'max_common_probability_delta':max(deltas),'mean_common_probability_delta':statistics.mean(deltas),'mean_coarse_KL_nats':statistics.mean(kls),'max_coarse_KL_nats':max(kls)}
(R/'gated-probability-summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
