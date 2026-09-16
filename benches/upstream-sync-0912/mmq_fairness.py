#!/usr/bin/env python3
from __future__ import annotations
import importlib.util, json, os, pathlib, statistics, subprocess

ROOT = pathlib.Path('/workspace/muse-glimmer-v100/benches/upstream-sync-0912/mmq-fairness')
ROOT.mkdir(parents=True, exist_ok=True)
GPU='GPU-1d64d3a4-4aea-87ec-9048-33a5217efd79'

# Reuse the exact saved-state server harness/config for Ornith V100.
spec=importlib.util.spec_from_file_location('hsb','/workspace/muse-glimmer-v100/benches/upstream-sync-0912/headline_server_bench.py')
hsb=importlib.util.module_from_spec(spec); spec.loader.exec_module(hsb)
cfg=dict(hsb.PROFILES['ornith-v100'])

# Policies: upstream normal; upstream global FORCE_MMQ; current selective MoE.
# A-B-C-C-B-A minimizes clock/order drift.
server_plan=[
 ('upstream-normal','upstream','normal',False),
 ('upstream-force','upstream','force',False),
 ('current-moe','current','normal',True),
 ('current-moe','current','normal',True),
 ('upstream-force','upstream','force',False),
 ('upstream-normal','upstream','normal',False),
]
server_rows=[]
for i,(name,arm,build,moe) in enumerate(server_plan):
    c=dict(cfg); c['build']=build; c['current_moe']=moe
    r=hsb.run_arm('ornith-v100-mmq',c,arm,f'{i}-{name}',43100+i,2)
    r['policy']=name
    server_rows.append(r)

def summarize_server(name):
    rows=[x for a in server_rows if a['policy']==name for x in a['rows']]
    return {k:statistics.mean(r[k] for r in rows) for k in ('pp','prompt_ms','tg','ttft_wall_ms','e2e_wall_ms')}
server_summary={k:summarize_server(k) for k in ('upstream-normal','upstream-force','current-moe')}
(ROOT/'ornith-v100.json').write_text(json.dumps({'summary':server_summary,'arms':server_rows},indent=2)+'\n')
print('ORNITH',json.dumps(server_summary,sort_keys=True),flush=True)

# Gemma 26B depth mode: upstream normal; upstream global FORCE_MMQ; current selective MoE.
MODEL='/models/Gemma-4-26B-A4B/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf'
BINS={
 'upstream-normal': pathlib.Path('/workspace/llama-upstream-3057bb66/build-sm70-75/bin/llama-bench'),
 'upstream-force': pathlib.Path('/workspace/llama-upstream-3057bb66/build-sm70-75-force/bin/llama-bench'),
 'current-moe': pathlib.Path('/workspace/muse-glimmer-v100/build-sync-sm70-75/bin/llama-bench'),
}
plan=['upstream-normal','upstream-force','current-moe','current-moe','upstream-force','upstream-normal']
gemma_rows=[]
for i,name in enumerate(plan):
    b=BINS[name]; assert b.exists(),b
    env=os.environ.copy()
    for k in list(env):
        if k.startswith(('GGML_CUDA_','LLAMA_EXPERIMENT_','LLAMA_MTP_')): env.pop(k,None)
    env['CUDA_VISIBLE_DEVICES']=GPU; env['LD_LIBRARY_PATH']=str(b.parent)
    if name=='current-moe': env['GGML_CUDA_VOLTA_FORCE_MMQ']='moe'
    cmd=[str(b),'-m',MODEL,'-d','100000','-p','1000','-n','128','-b','4096','-ub','1024','-ctk','q8_0','-ctv','q8_0','-ngl','999','-fa','on','-sm','none','-dev','CUDA0','-r','3','--no-warmup','-o','json']
    cp=subprocess.run(cmd,env=env,text=True,capture_output=True,check=True)
    (ROOT/f'gemma26-{i}-{name}.json').write_text(cp.stdout); (ROOT/f'gemma26-{i}-{name}.err').write_text(cp.stderr)
    arr=json.loads(cp.stdout); pp=next(x for x in arr if x.get('n_prompt')==1000 and x.get('n_gen')==0); tg=next(x for x in arr if x.get('n_gen')==128)
    row={'policy':name,'pp':pp['avg_ts'],'pp_ms':pp['avg_ns']/1e6,'tg':tg['avg_ts'],'tg_ms':tg['avg_ns']/1e6}
    gemma_rows.append(row); print('GEMMA26',i,name,f"PP={row['pp']:.3f}",f"TG={row['tg']:.3f}",flush=True)

def summarize_gemma(name):
    rows=[r for r in gemma_rows if r['policy']==name]
    return {k:statistics.mean(r[k] for r in rows) for k in ('pp','pp_ms','tg','tg_ms')}
gemma_summary={k:summarize_gemma(k) for k in ('upstream-normal','upstream-force','current-moe')}
(ROOT/'gemma26.json').write_text(json.dumps({'summary':gemma_summary,'rows':gemma_rows},indent=2)+'\n')
print('GEMMA_SUMMARY',json.dumps(gemma_summary,sort_keys=True),flush=True)
