#!/usr/bin/env python3
"""Fresh user-facing upstream vs v100-optimized long-context server benchmarks.

Each profile uses A-B-B-A process order. Every process discards one warm request,
then retains --reps requests. A request restores the same saved prefix, appends
1000 tokens, and generates 64 tokens, so PP/TTFT/TG are measured together.
"""
from __future__ import annotations
import argparse, hashlib, json, os, pathlib, signal, statistics, struct, subprocess, time
import urllib.error, urllib.request

ROOT = pathlib.Path('/workspace/muse-glimmer-v100/benches/upstream-sync-0912')
ROOT.mkdir(parents=True, exist_ok=True)
V100 = 'GPU-1d64d3a4-4aea-87ec-9048-33a5217efd79'
RTX  = 'GPU-30acbf2b-a41a-8b88-e298-fe66c4138b29'
QWEN_MODEL = '/models/Qwen3.8-27B/Qwen3.8-27B-UD-Q5_K_XL.gguf'
ORNITH_Q6Q5 = '/models/Ornith-1.5-35B-A3B/Ornith-1.5-35B-A3B-AD-Q6_K-Q5_K.gguf'
ORNITH_Q5Q4 = '/models/Ornith-1.5-35B-A3B/Ornith-1.5-35B-A3B-AD-Q5_K-Q4_K.gguf'
MUSE_MODEL = '/models/Muse-Glimmer-30B/Muse-Glimmer-30B-UD-Q4_K_XL.gguf'
QWEN_SEQ = pathlib.Path('/workspace/oai-qwen38-pp-lab/results/cache100k-sharedstate-ab/seq.json')
ORNITH_SEQ = pathlib.Path('/workspace/oai-qwen38-pp-lab/results/ornith15-seq101k.json')
MUSE_SEQ = pathlib.Path('/models/.bench-muse-glimmer-v100/seq101k.json')

BINS = {
    'upstream': pathlib.Path('/workspace/llama-upstream-3057bb66/build-sm70-75/bin/llama-server'),
    'upstream-force': pathlib.Path('/workspace/llama-upstream-3057bb66/build-sm70-75-force/bin/llama-server'),
    'current': pathlib.Path('/workspace/muse-glimmer-v100/build-sync-sm70-75/bin/llama-server'),
    'current-force': pathlib.Path('/workspace/muse-glimmer-v100/build-sync-sm70-75-force/bin/llama-server'),
}

PROFILES = {
    'qwen-v100': dict(model=QWEN_MODEL, seq=QWEN_SEQ, state=pathlib.Path('/workspace/oai-qwen38-pp-lab/results/sync-master-20260828/cache-v3-regenerated'), slot='cache100k.bin', prefix=100000, ctx=131072, batch=4096, ubatch=4096, cuda=V100, build='normal', placement=['--device','CUDA0','--split-mode','none'], extra=['--ctx-checkpoints','32']),
    'qwen-rtx': dict(model=QWEN_MODEL, seq=QWEN_SEQ, state=pathlib.Path('/models/.bench-qwen2080-sync'), slot='cache65536.bin', prefix=65536, ctx=67584, batch=4096, ubatch=2048, cuda=RTX, build='normal', placement=['--device','CUDA0','--split-mode','none'], extra=['--ctx-checkpoints','0']),
    'qwen-dual': dict(model=QWEN_MODEL, seq=QWEN_SEQ, state=pathlib.Path('/workspace/llama-q8attn-validate/benches/q8attn-validate/cache-yarn400'), slot='cache100k.bin', prefix=100000, ctx=409600, batch=4096, ubatch=2048, cuda=f'{V100},{RTX}', build='normal', placement=['--device','CUDA1,CUDA0','--split-mode','tensor','--tensor-split','4,5'], extra=['--override-kv','qwen35.context_length=int:409600','--rope-scaling','yarn','--rope-scale','1.5625','--yarn-orig-ctx','262144','--ctx-checkpoints','32','--checkpoint-min-step','8192'], allreduce=True),
    'ornith-v100': dict(model=ORNITH_Q6Q5, seq=ORNITH_SEQ, state=pathlib.Path('/models/.bench-ornith-sync'), slot='ornith100k.bin', prefix=100000, ctx=131072, batch=2048, ubatch=512, cuda=V100, build='normal', placement=['--device','CUDA0','--split-mode','none'], extra=['--ctx-checkpoints','0','--reasoning','off','--reasoning-format','none'], current_moe=True),
    'ornith-rtx': dict(model=ORNITH_Q5Q4, seq=ORNITH_SEQ, state=pathlib.Path('/models/.bench-ornith2080'), slot='ornith65536-q5q4-force-mmq.bin', prefix=65536, ctx=67584, batch=4096, ubatch=512, cuda=RTX, build='force', placement=['--device','CUDA0','--split-mode','none'], extra=['--ctx-checkpoints','0','--reasoning','off','--reasoning-format','none']),
    'ornith-dual': dict(model=ORNITH_Q6Q5, seq=ORNITH_SEQ, state=pathlib.Path('/models/.bench-ornith-sync'), slot='ornith100k.bin', prefix=100000, ctx=131072, batch=2048, ubatch=1024, cuda=f'{V100},{RTX}', build='force', placement=['--device','CUDA1,CUDA0','--split-mode','tensor','--tensor-split','1,1'], extra=['--ctx-checkpoints','0','--reasoning','off','--reasoning-format','none'], allreduce=True),
    'muse-v100': dict(model=MUSE_MODEL, seq=MUSE_SEQ, state=pathlib.Path('/models/.bench-muse-glimmer-v100'), slot='cache100k.bin', prefix=100000, ctx=106496, batch=4096, ubatch=1024, cuda=V100, build='normal', placement=['--device','CUDA0','--split-mode','none'], extra=[]),
}

def saved_tokens(path: pathlib.Path) -> list[int]:
    with path.open('rb') as f:
        magic, version, n = struct.unpack('<III', f.read(12))
        if magic != 0x67677371 or version != 3 or n > 2_000_000:
            raise ValueError(f'unsupported state header {magic:x}/{version}/{n}: {path}')
        toks = list(struct.unpack(f'<{n}i', f.read(4*n)))
    if toks and toks[0] == -1:
        n2 = toks[2]
        return toks[3:3+n2]
    return toks

def req(port, path, obj=None, timeout=1800):
    data=json.dumps(obj or {},separators=(',',':')).encode()
    r=urllib.request.Request(f'http://127.0.0.1:{port}{path}',data=data,headers={'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(r,timeout=timeout) as f: return json.load(f)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f'HTTP {e.code}: '+e.read().decode(errors='ignore')[:4000])

def health(port):
    with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=.5) as f: return json.load(f)

def stream(port,prompt,npred=128):
    obj={'prompt':prompt,'n_predict':npred,'ignore_eos':True,'temperature':0.0,'seed':1234,'id_slot':0,'cache_prompt':True,'stream':True,'return_tokens':True}
    data=json.dumps(obj,separators=(',',':')).encode(); r=urllib.request.Request(f'http://127.0.0.1:{port}/completion',data=data,headers={'Content-Type':'application/json'})
    t0=time.perf_counter(); first=None; final=None; toks=[]
    with urllib.request.urlopen(r,timeout=2400) as f:
        for raw in f:
            line=raw.strip()
            if not line.startswith(b'data:'): continue
            d=line[5:].strip()
            if not d or d==b'[DONE]': continue
            try: z=json.loads(d)
            except Exception: continue
            ts=z.get('tokens') or []
            if first is None and (ts or z.get('content')): first=time.perf_counter()
            toks.extend(ts)
            if z.get('stop'): final=z
    t1=time.perf_counter()
    if final is None: raise RuntimeError('no final event')
    return final,(first or t1),t0,t1,toks

def binary_for(cfg, arm):
    suffix='-force' if cfg['build']=='force' else ''
    return BINS[arm+suffix]

def env_for(cfg, arm, binary):
    e=os.environ.copy()
    for k in list(e):
        if k.startswith(('GGML_CUDA_','LLAMA_EXPERIMENT_','LLAMA_MTP_')): e.pop(k,None)
    if cfg.get('cuda'): e['CUDA_VISIBLE_DEVICES']=cfg['cuda']
    e['LD_LIBRARY_PATH']=str(binary.parent)
    if cfg.get('allreduce'):
        e['GGML_CUDA_ALLREDUCE']='internal'; e['GGML_CUDA_AR_COPY_THRESHOLD']='131072'
    if cfg.get('current_moe') and arm=='current': e['GGML_CUDA_VOLTA_FORCE_MMQ']='moe'
    return e

def run_arm(profile, cfg, arm, label, port, reps):
    binary=binary_for(cfg,arm); assert binary.exists(), binary
    seq=json.loads(cfg['seq'].read_text())[:cfg['prefix']+1000]
    state=cfg['state']/cfg['slot']; assert state.exists(),state
    saved=saved_tokens(state)
    if saved != seq[:cfg['prefix']]: raise RuntimeError(f'{profile}: saved prefix != prompt prefix')
    args=[str(binary),'--model',cfg['model'],'--host','127.0.0.1','--port',str(port),'--ctx-size',str(cfg['ctx']),'--parallel','1','--fit','off','--gpu-layers','all','--flash-attn','on','--batch-size',str(cfg['batch']),'--ubatch-size',str(cfg['ubatch']),'--cache-type-k','q8_0','--cache-type-v','q8_0','--cache-ram','0','--slot-save-path',str(cfg['state']),'--no-warmup','--perf','--skip-chat-parsing']+cfg['placement']+cfg['extra']
    env=env_for(cfg,arm,binary)
    logp=ROOT/f'{profile}-{label}.server.log'; log=logp.open('wb')
    p=subprocess.Popen(args,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    try:
        for _ in range(3600):
            if p.poll() is not None: raise RuntimeError(logp.read_text(errors='ignore')[-12000:])
            try:
                if health(port).get('status')=='ok': break
            except Exception: pass
            time.sleep(.1)
        else: raise RuntimeError('health timeout')
        rows=[]
        for rep in range(reps+1):
            rr=req(port,'/slots/0?action=restore',{'filename':cfg['slot']},1800)
            if rr.get('n_restored') != cfg['prefix']: raise RuntimeError(rr)
            z,first,t0,t1,toks=stream(port,seq,128); tm=z['timings']
            if tm.get('cache_n') != cfg['prefix'] or tm.get('prompt_n') != 1000 or tm.get('predicted_n') != 128: raise RuntimeError(tm)
            row={'rep':rep,'warmup':rep==0,'pp':tm['prompt_per_second'],'prompt_ms':tm['prompt_ms'],'tg':tm['predicted_per_second'],'predicted_ms':tm['predicted_ms'],'ttft_wall_ms':(first-t0)*1000,'e2e_wall_ms':(t1-t0)*1000,'token_sha':hashlib.sha256(json.dumps(toks).encode()).hexdigest()}
            print(profile,label,rep,f"PP={row['pp']:.2f}",f"TG={row['tg']:.2f}",f"TTFT={row['ttft_wall_ms']:.1f}",flush=True)
            if rep: rows.append(row)
        return {'profile':profile,'arm':arm,'label':label,'binary':str(binary),'env':{k:v for k,v in env.items() if k.startswith(('CUDA_VISIBLE','GGML_CUDA_'))},'argv':args,'rows':rows}
    finally:
        if p.poll() is None:
            os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(30)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        log.close()

def summarize(profile,cfg,arms):
    out={}
    for arm in ('upstream','current'):
        rows=[r for a in arms if a['arm']==arm for r in a['rows']]
        out[arm]={k:statistics.mean(r[k] for r in rows) for k in ('pp','prompt_ms','tg','predicted_ms','ttft_wall_ms','e2e_wall_ms')}
        out[arm]['pp_sd']=statistics.stdev(r['pp'] for r in rows) if len(rows)>1 else 0
        out[arm]['tg_sd']=statistics.stdev(r['tg'] for r in rows) if len(rows)>1 else 0
        out[arm]['hashes']=sorted({r['token_sha'] for r in rows})
    out['pp_gain_pct']=100*(out['current']['pp']/out['upstream']['pp']-1)
    out['tg_gain_pct']=100*(out['current']['tg']/out['upstream']['tg']-1)
    out['ttft_reduction_pct']=100*(1-out['current']['ttft_wall_ms']/out['upstream']['ttft_wall_ms'])
    return {'profile':profile,'prefix':cfg['prefix'],'append':1000,'generated':128,'summary':out,'arms':arms}

def main():
    p=argparse.ArgumentParser(); p.add_argument('profiles',nargs='+',choices=PROFILES); p.add_argument('--reps',type=int,default=2); p.add_argument('--port',type=int,default=42400); a=p.parse_args()
    all_summary={}
    for pi,profile in enumerate(a.profiles):
        cfg=PROFILES[profile]; arms=[]
        order=[('upstream','A0'),('current','B0'),('current','B1'),('upstream','A1')]
        for i,(arm,label) in enumerate(order): arms.append(run_arm(profile,cfg,arm,label,a.port+pi*10+i,a.reps))
        report=summarize(profile,cfg,arms); (ROOT/f'{profile}.json').write_text(json.dumps(report,indent=2)+'\n'); all_summary[profile]=report['summary']; print('SUMMARY',profile,json.dumps(report['summary'],sort_keys=True),flush=True)
    (ROOT/'headline-server-summary.json').write_text(json.dumps(all_summary,indent=2)+'\n')

if __name__=='__main__': main()
