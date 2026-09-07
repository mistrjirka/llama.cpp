#!/usr/bin/env python3
"""Identical serialized histories and parameters; compare only prefix-first restore."""
import concurrent.futures as cf
import importlib.util, json, pathlib, statistics, threading, time
ROOT=pathlib.Path(__file__).resolve().parent
OLD=pathlib.Path('/workspace/oai-qwen38-pp-lab/results/parallel-research-0907')
spec=importlib.util.spec_from_file_location('screen',ROOT/'fixture.py')
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
# All input snapshots stay in OLD. Logs and raw results go into this experiment.
orig_start=s.start
s.PORT=32482
bins={'A':ROOT/'baseline-bin','B':pathlib.Path('/workspace/llama-multiagent-cache/build-sm70-75/bin')}
rows=[]; hash_ref=None
for arm,label in enumerate(['A','B','B','A']):
    s.BIN=str(bins[label]/'llama-server');s.ENV['LD_LIBRARY_PATH']=str(bins[label])
    p,log=s.start(3,f'../prefix-first-0907/{label}-{arm}')
    try:
        maps=pathlib.Path(f'/proc/{p.pid}/maps').read_text()
        libs=sorted({l.split()[-1] for l in maps.splitlines() if 'libllama' in l})
        assert any(str(bins[label]) in x and 'libllama-server' in x for x in libs),libs
        for rep in range(3):
            t=time.perf_counter();s.restore();restore_s=time.perf_counter()-t
            barrier=threading.Barrier(4);t=time.perf_counter()
            with cf.ThreadPoolExecutor(max_workers=4) as ex:
                rr=list(ex.map(lambda i:s.comp(i,s.PROMPTS[i],128,barrier),range(4)))
            wall=time.perf_counter()-t
            for r in rr:
                assert r['timings']['cache_n']==100000,r['timings']
                assert r['timings']['prompt_n']==1000,r['timings']
                assert r['timings']['predicted_n']==128,r['timings']
                r.pop('tokens',None)
            hashes=[r['sha'] for r in rr]
            if hash_ref is None: hash_ref=hashes
            assert hashes==hash_ref,(label,rep,hashes,hash_ref)
            row={'arm':label,'round':arm,'repeat':rep,'warmup':rep==0,'wall_s':wall,'restore_s':restore_s,'output_tps_whole_turn':512/wall,'mean_request_pp':statistics.mean(r['timings']['prompt_per_second'] for r in rr),'mean_request_tg':statistics.mean(r['timings']['predicted_per_second'] for r in rr),'rows':rr,'libraries':libs}
            rows.append(row);(ROOT/'raw.json').write_text(json.dumps(rows,indent=2))
            print(json.dumps({k:v for k,v in row.items() if k not in ('rows','libraries')}),flush=True)
    finally:s.stop(p,log)
summary={}
for label in bins:
    group=[r for r in rows if r['arm']==label and not r['warmup']]
    summary[label]={'n':len(group)}
    for key in ['wall_s','restore_s','output_tps_whole_turn','mean_request_pp','mean_request_tg']:
        v=[r[key] for r in group]
        summary[label][key]={'mean':statistics.mean(v),'min':min(v),'max':max(v)}
summary['matching_hashes']=True
summary['speedup']=summary['A']['wall_s']['mean']/summary['B']['wall_s']['mean']
summary['context']='4 x 100000 cached, shared prefix 98000, +1000 input and +128 output each; synthetic repetitive corpus'
(ROOT/'summary.json').write_text(json.dumps(summary,indent=2));print('SUMMARY',json.dumps(summary),flush=True)
