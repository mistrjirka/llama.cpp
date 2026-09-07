#!/usr/bin/env python3
"""Ablate only CPU bookkeeping. Latest validated cache-refresh/readback base is on.
All other weights, attention kernels, target/draft KV, contexts and MTP depth fixed.
"""
import argparse,concurrent.futures as cf,importlib.util,json,pathlib,statistics,threading,time
R=pathlib.Path('/workspace/oai-qwen38-pp-lab/results/mtp-final-integration-0907');F=R.parent/'parallel-refined-0907';B=R/'candidate-bin'
p=argparse.ArgumentParser();p.add_argument('--suite',choices=['four','one','off'],default='four');a=p.parse_args()
z=importlib.util.spec_from_file_location('fixture',R.parent/'parallel-research-0907/sweep.py');s=importlib.util.module_from_spec(z);z.loader.exec_module(s)
s.PORT=32707;s.OUT=R;s.BIN=str(B/'llama-server');s.ENV['LD_LIBRARY_PATH']=str(B)
for k in list(s.ENV):
 if k.startswith('LLAMA_EXPERIMENT_') or k.startswith('LLAMA_MTP_') or k=='LLAMA_KV_INDEXED_RM' or k.startswith('GGML_CUDA_VOLTA_Q8_MULTI') or k.startswith('GGML_CUDA_VOLTA_Q8_REF') or k in ['GGML_CUDA_VOLTA_Q8_COOP','GGML_CUDA_VOLTA_Q8_PV_F32','LD_PRELOAD']:s.ENV.pop(k)
f=json.loads((F/'real-fixture.json').read_text());rows=[]
order=['baseline','candidate','candidate','baseline'];concurrency=1 if a.suite=='one' else 4
for block,arm in enumerate(order):
 B=R/('baseline-bin' if arm=='baseline' else 'candidate-bin');s.BIN=str(B/'llama-server');s.ENV['LD_LIBRARY_PATH']=str(B)
 depth=0 if a.suite=='off' else 3
 p,log=s.start(depth,f'bench-{a.suite}-{block}-{arm}')
 try:
  assert str(B.resolve()/'libllama') in pathlib.Path(f'/proc/{p.pid}/maps').read_text()
  for rep in range(3):
   for i in range(4):s.req(f'/slots/{i}?action=erase',{})
   for i in range(4):assert s.req(f'/slots/{i}?action=restore',{'filename':f'real100k-{i}.bin'})['n_restored']==100000
   barrier=threading.Barrier(concurrency);t=time.perf_counter()
   with cf.ThreadPoolExecutor(max_workers=concurrency) as ex:out=list(ex.map(lambda i:s.comp(i,f['prompts'][i],128,barrier),range(concurrency)))
   elapsed=time.perf_counter()-t
   for i,r in enumerate(out):
    assert r['timings']['predicted_n']==128 and r['timings']['cache_n']==100000 and r['timings']['prompt_n']==len(f['prompts'][i])-100000,r
   row={'arm':arm,'block':block,'rep':rep,'warmup':rep==0,'concurrency':concurrency,'depth':depth,'wall_s':elapsed,'mean_tg':statistics.mean(x['timings']['predicted_per_second'] for x in out),'output_tps_turn':128*concurrency/elapsed,'requests':out,'argv':s.args(depth)}
   rows.append(row);(R/(a.suite+'-bench-raw.json')).write_text(json.dumps(rows,indent=2))
   print(a.suite,arm,rep,round(elapsed,4),round(row['mean_tg'],3),flush=True)
 finally:s.stop(p,log)
summary={}
for arm in dict.fromkeys(order):
 rs=[r for r in rows if r['arm']==arm and not r['warmup']]
 summary[arm]={'n':len(rs),**{k:{'mean':statistics.mean(r[k] for r in rs),'min':min(r[k] for r in rs),'max':max(r[k] for r in rs)} for k in ['wall_s','mean_tg','output_tps_turn']}}
(R/(a.suite+'-bench-summary.json')).write_text(json.dumps(summary,indent=2));print('SUMMARY',json.dumps(summary),flush=True)
