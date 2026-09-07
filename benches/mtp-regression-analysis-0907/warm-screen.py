#!/usr/bin/env python3
"""Diagnostic: genuine off vs warm zero vs MTP1/2/3 on immutable 100k source states.
No experimental attention/grouping; all weights, pool and timing definitions fixed.
The warm0 arm retains the draft context but limits proposals to zero.
"""
import concurrent.futures as cf
import hashlib, importlib.util, json, pathlib, statistics, threading, time
R=pathlib.Path('/workspace/oai-qwen38-pp-lab/results/mtp-regression-analysis-0907')
F=R.parent/'parallel-refined-0907'
B=F/'final-bin'
z=importlib.util.spec_from_file_location('fixture',R.parent/'parallel-research-0907/sweep.py')
s=importlib.util.module_from_spec(z); z.loader.exec_module(s)
s.PORT=32560;s.OUT=F;s.BIN=str(B/'llama-server');s.ENV['LD_LIBRARY_PATH']=str(B)
for k in list(s.ENV):
 if k.startswith('LLAMA_EXPERIMENT_') or k.startswith('GGML_CUDA_VOLTA_Q8_MULTI') or k.startswith('GGML_CUDA_VOLTA_Q8_REF') or k in ['GGML_CUDA_VOLTA_Q8_COOP','GGML_CUDA_VOLTA_Q8_PV_F32','LLAMA_ARG_BACKEND_SAMPLING']:
  s.ENV.pop(k)
f=json.loads((F/'real-fixture.json').read_text());base_args=s.args;rows=[];errors=[]
order=['off','warm','warm','off']
for block,arm in enumerate(order):
 s.ENV.pop('LD_PRELOAD',None)
 if arm=='warm':s.ENV['LD_PRELOAD']=str(R/'no-proposals.so')
 def args(_):
  n=0 if arm=='off' else (3 if arm=='warm' else int(arm[-1]))
  a=base_args(n)
  if arm=='warm0':a[a.index('--spec-draft-n-max')+1]='0'
  return a
 s.args=args;p=log=None
 try:
  p,log=s.start(0,f'../mtp-regression-analysis-0907/warm-{block}-{arm}')
  assert str(B/'libllama') in pathlib.Path(f'/proc/{p.pid}/maps').read_text()
  for rep in range(2):
   for i in range(4):s.req(f'/slots/{i}?action=erase',{})
   for i in range(4):
    x=s.req(f'/slots/{i}?action=restore',{'filename':f'real100k-{i}.bin'})
    assert x['n_restored']==100000,x
   barrier=threading.Barrier(4);t=time.perf_counter()
   with cf.ThreadPoolExecutor(max_workers=4) as ex:out=list(ex.map(lambda i:s.comp(i,f['prompts'][i],128,barrier),range(4)))
   wall=time.perf_counter()-t
   for i,x in enumerate(out):
    assert x['timings']['cache_n']==100000 and x['timings']['prompt_n']==len(f['prompts'][i])-100000,x
    assert x['timings']['predicted_n']==128,x
   if arm=='warm':
    assert all(x['timings'].get('draft_n',0)==0 for x in out), 'proposal suppression failed'
   row={'arm':arm,'block':block,'warmup':rep==0,'wall_s':wall,'mean_tg':statistics.mean(x['timings']['predicted_per_second'] for x in out),'output_tps':512/wall,'requests':out,'argv':s.args(0)}
   rows.append(row);(R/'warm-raw.json').write_text(json.dumps(rows,indent=2))
   print(arm,rep,round(wall,4),round(row['mean_tg'],3),[(x['timings'].get('draft_n',0),x['timings'].get('draft_n_accepted',0)) for x in out],flush=True)
 except Exception as e:
  errors.append({'arm':arm,'block':block,'error':repr(e)})
  (R/'warm-errors.json').write_text(json.dumps(errors,indent=2));print('ERROR',arm,repr(e),flush=True)
 finally:
  if p is not None:
   s.stop(p,log)
   if arm=='warm':
    import re
    text=(R/f'warm-{block}-{arm}.log').read_text(errors='replace')
    match=re.search(r'MTP_NO_PROPOSALS_DIAGNOSTIC calls=(\d+)',text)
    assert match and int(match[1])>0, 'no-op shim was not invoked'
  s.args=base_args
summary={'scope':'4x100k C++ code histories, 38/42/44/41 input, 128 output each; two retained runs per arm; screening not a quality study','errors':errors,'arms':{}}
for arm in dict.fromkeys(order):
 rs=[r for r in rows if r['arm']==arm and not r['warmup']]
 if rs:
  summary['arms'][arm]={'n':len(rs), **{key:{'mean':statistics.mean(r[key] for r in rs),'min':min(r[key] for r in rs),'max':max(r[key] for r in rs)} for key in ['wall_s','mean_tg','output_tps']}}
(R/'warm-summary.json').write_text(json.dumps(summary,indent=2));print('SUMMARY',json.dumps(summary),flush=True)
