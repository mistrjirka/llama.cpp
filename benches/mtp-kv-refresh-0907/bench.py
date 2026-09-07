#!/usr/bin/env python3
import os
"""Matched within-binary cache-only ablations, bracketed with genuine MTP-off.
Restore time excluded. Each process has one warmup plus two measured turns.
No profiler/interposition/build runs during these timings.
"""
import argparse,concurrent.futures as cf,importlib.util,json,pathlib,statistics,threading,time
R=pathlib.Path(os.environ.get('MTP_REFRESH_RESULTS_DIR', '/workspace/oai-qwen38-pp-lab/results/mtp-kv-refresh-0907'));F=R.parent/'parallel-refined-0907';B=R/'candidate-bin'
p=argparse.ArgumentParser();p.add_argument('--suite',choices=['four','one','warm'],default='four');a=p.parse_args()
z=importlib.util.spec_from_file_location('fixture',R.parent/'parallel-research-0907/sweep.py');s=importlib.util.module_from_spec(z);z.loader.exec_module(s)
s.PORT=32572;s.OUT=R;s.BIN=str(B/'llama-server');s.ENV['LD_LIBRARY_PATH']=str(B)
for k in list(s.ENV):
 if k.startswith('LLAMA_EXPERIMENT_') or k.startswith('GGML_CUDA_VOLTA_Q8_MULTI') or k.startswith('GGML_CUDA_VOLTA_Q8_REF') or k in ['GGML_CUDA_VOLTA_Q8_COOP','GGML_CUDA_VOLTA_Q8_PV_F32','LD_PRELOAD','LLAMA_ARG_BACKEND_SAMPLING']:s.ENV.pop(k)
f=json.loads((F/'real-fixture.json').read_text());original_args=s.args;rows=[]
if a.suite=='four':
 order=['off','mtp3','kv3','kv1','mtp1','mtp1','kv1','kv3','mtp3','off'];concurrency=4
elif a.suite=='one':order=['mtp3','kv3','kv3','mtp3'];concurrency=1
else:order=['off','warm0','kv0','kv0','warm0','off'];concurrency=4
for block,arm in enumerate(order):
 s.ENV.pop('LLAMA_EXPERIMENT_MTP_KV_ONLY',None)
 if arm.startswith('kv'):s.ENV['LLAMA_EXPERIMENT_MTP_KV_ONLY']='1'
 def arguments(_):
  cmd=original_args(0 if arm=='off' else 3 if arm.endswith('0') else int(arm[-1]))
  if arm.endswith('0'):cmd[cmd.index('--spec-draft-n-max')+1]='0'
  return cmd
 s.args=arguments
 p,log=s.start(0,f'bench-{a.suite}-{block}-{arm}')
 try:
  maps=pathlib.Path(f'/proc/{p.pid}/maps').read_text();assert str(B/'libllama') in maps
  for rep in range(3):
   for i in range(4):s.req(f'/slots/{i}?action=erase',{})
   for i in range(4):assert s.req(f'/slots/{i}?action=restore',{'filename':f'real100k-{i}.bin'})['n_restored']==100000
   barrier=threading.Barrier(concurrency);t=time.perf_counter()
   with cf.ThreadPoolExecutor(max_workers=concurrency) as ex:rr=list(ex.map(lambda i:s.comp(i,f['prompts'][i],128,barrier),range(concurrency)))
   wall=time.perf_counter()-t
   for i,x in enumerate(rr):
    assert x['timings']['cache_n']==100000 and x['timings']['prompt_n']==len(f['prompts'][i])-100000,x
    assert x['timings']['predicted_n']==128,x
    if arm.endswith('0'):assert x['timings'].get('draft_n',0)==0,x
   row={'arm':arm,'block':block,'warmup':rep==0,'concurrency':concurrency,'wall_s':wall,'mean_tg':statistics.mean(x['timings']['predicted_per_second'] for x in rr),'turn_output_tps':128*concurrency/wall,'requests':rr,'argv':s.args(0)}
   rows.append(row);(R/f'{a.suite}-raw.json').write_text(json.dumps(rows,indent=2))
   print(a.suite,arm,rep,round(wall,4),round(row['mean_tg'],3),flush=True)
 finally:s.stop(p,log);s.args=original_args
summary={}
for arm in dict.fromkeys(order):
 rs=[r for r in rows if r['arm']==arm and not r['warmup']]
 summary[arm]={'n':len(rs),**{k:{'mean':statistics.mean(r[k] for r in rs),'min':min(r[k] for r in rs),'max':max(r[k] for r in rs)} for k in ['wall_s','mean_tg','turn_output_tps']}}
(R/f'{a.suite}-summary.json').write_text(json.dumps(summary,indent=2));print('SUMMARY',json.dumps(summary),flush=True)
