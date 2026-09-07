import importlib.util,pathlib,json,concurrent.futures as cf,statistics,math
R=pathlib.Path('/workspace/oai-qwen38-pp-lab/results/prefix-first-0907');O=R.parent/'parallel-research-0907'
z=importlib.util.spec_from_file_location('s',O/'sweep.py');s=importlib.util.module_from_spec(z);z.loader.exec_module(s);s.PORT=32491
pr=json.loads((R/'code-prompts.json').read_text());hs=pr['histories'];base=pr['prompts'];out=[];forced=None
for label in ['A','B','A']:
 d=R/'baseline-bin' if label=='A' else R.parent/'parallel-experiments-0907/final-bin';s.BIN=str(d/'llama-server');s.ENV['LD_LIBRARY_PATH']=str(d)
 p,l=s.start(0,f'../prefix-first-0907/default-teacher-{label}-{len(out)}')
 try:
  for i in range(4):s.req(f'/slots/{i}?action=restore',{'filename':f'codebase{i}.bin'})
  seq=[[] for _ in range(4)];values=[]
  for step in range(24):
   batch=[]
   for i in range(4):
    r=s.req('/completion',{'prompt':base[i]+seq[i],'id_slot':i,'n_predict':1,'temperature':0.0,'seed':1234,'n_probs':20,'post_sampling_probs':False,'cache_prompt':True,'return_tokens':True,'ignore_eos':True})
    if step==0 and i==0:(R/f'default-example-{label}.json').write_text(json.dumps(r,indent=2))
    batch.append({'tokens':r.get('tokens'),'probs':r.get('completion_probabilities'),'timings':r['timings']})
   values.append(batch)
   for i in range(4):seq[i].append((forced[step][i] if forced is not None else batch[i]['tokens'][0]))
  if forced is None:forced=[[x['tokens'][0] for x in b] for b in values]
  out.append({'label':label,'values':values});(R/'default-teacher-raw.json').write_text(json.dumps(out,indent=2))
  print(label,'done',flush=True)
 finally:s.stop(p,l)
summary={'n_forced_steps':24,'sequences':4}
for j in [1,2]:
 mismatches=sum(a['tokens']!=b['tokens'] for aa,bb in zip(out[0]['values'],out[j]['values']) for a,b in zip(aa,bb))
 summary['A_'+('B' if j==1 else 'A')+'_greedy_mismatches']=mismatches
(R/'default-teacher-summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
