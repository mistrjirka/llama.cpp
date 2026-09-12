#!/usr/bin/env python3
"""Matched long-context Gemma 4 PP/TG benchmarks on V100 using llama-bench depth mode."""
from __future__ import annotations
import json, os, pathlib, statistics, subprocess

ROOT=pathlib.Path('/workspace/muse-glimmer-v100/benches/upstream-sync-0912'); ROOT.mkdir(parents=True,exist_ok=True)
GPU='GPU-1d64d3a4-4aea-87ec-9048-33a5217efd79'
BINS={
 'upstream':pathlib.Path('/workspace/llama-upstream-3057bb66/build-sm70-75/bin/llama-bench'),
 'current':pathlib.Path('/workspace/muse-glimmer-v100/build-sync-sm70-75/bin/llama-bench'),
}
MODELS={
 'gemma31':('/models/Gemma4-31B/gemma-4-31B-it-UD-Q4_K_XL.gguf','Gemma 4 31B'),
 'gemma26':('/models/Gemma-4-26B-A4B/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf','Gemma 4 26B-A4B'),
 'muse-v100':('/models/Muse-Glimmer-30B/Muse-Glimmer-30B-UD-Q4_K_XL.gguf','Muse Glimmer 30B'),
}

def run(which,arm,idx):
    b=BINS[arm]; model,_=MODELS[which]; assert b.exists(),b
    env=os.environ.copy()
    for k in list(env):
        if k.startswith(('GGML_CUDA_','LLAMA_EXPERIMENT_','LLAMA_MTP_')): env.pop(k,None)
    env['CUDA_VISIBLE_DEVICES']=GPU; env['LD_LIBRARY_PATH']=str(b.parent)
    cmd=[str(b),'-m',model,'-d','100000','-p','1000','-n','128','-b','4096','-ub','1024','-ctk','q8_0','-ctv','q8_0','-ngl','999','-fa','on','-sm','none','-dev','CUDA0','-r','5','--no-warmup','-o','json']
    cp=subprocess.run(cmd,env=env,text=True,capture_output=True,check=True)
    p=ROOT/f'{which}-{arm}-{idx}.json'; p.write_text(cp.stdout); (ROOT/f'{which}-{arm}-{idx}.err').write_text(cp.stderr)
    data=json.loads(cp.stdout)
    pp=next(x for x in data if x.get('n_prompt')==1000 and x.get('n_gen')==0)
    tg=next(x for x in data if x.get('n_gen')==128)
    return {'arm':arm,'pp':pp['avg_ts'],'pp_sd':pp['stddev_ts'],'tg':tg['avg_ts'],'tg_sd':tg['stddev_ts'],'pp_ms':pp['avg_ns']/1e6,'tg_ms':tg['avg_ns']/1e6,'cmd':cmd}

def main():
    allout={}
    for which in MODELS:
        rows=[]
        for idx,arm in enumerate(['upstream','current']):
            r=run(which,arm,idx); rows.append(r); print(which,arm,f"PP={r['pp']:.3f}",f"TG={r['tg']:.3f}",flush=True)
        sm={}
        for arm in ('upstream','current'):
            rr=[r for r in rows if r['arm']==arm]
            sm[arm]={'pp':statistics.mean(r['pp'] for r in rr),'tg':statistics.mean(r['tg'] for r in rr),'pp_between_process_sd':statistics.stdev(r['pp'] for r in rr) if len(rr)>1 else 0.0,'tg_between_process_sd':statistics.stdev(r['tg'] for r in rr) if len(rr)>1 else 0.0}
        sm['pp_gain_pct']=100*(sm['current']['pp']/sm['upstream']['pp']-1); sm['tg_gain_pct']=100*(sm['current']['tg']/sm['upstream']['tg']-1)
        rep={'model':MODELS[which][1],'depth':100000,'append':1000,'generated':128,'summary':sm,'rows':rows}
        (ROOT/f'{which}.json').write_text(json.dumps(rep,indent=2)+'\n'); allout[which]=rep
        print('SUMMARY',which,json.dumps(sm,sort_keys=True),flush=True)
    (ROOT/'headline-gemma-summary.json').write_text(json.dumps(allout,indent=2)+'\n')
if __name__=='__main__': main()
