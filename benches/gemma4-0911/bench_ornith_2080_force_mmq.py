#!/usr/bin/env python3
import hashlib, json, os, pathlib, signal, statistics, subprocess, time, urllib.request, urllib.error

ROOT = pathlib.Path('/workspace/llama-v100-optimized/benches/gemma4-0911/ornith-2080-force-mmq')
ROOT.mkdir(parents=True, exist_ok=True)
CACHE = pathlib.Path('/models/.bench-ornith2080')
CACHE.mkdir(parents=True, exist_ok=True)
SLOT = 'ornith65536-q5q4-force-mmq.bin'
MODEL = '/models/Ornith-1.5-35B-A3B/Ornith-1.5-35B-A3B-AD-Q5_K-Q4_K.gguf'
SEQ = json.load(open('/workspace/oai-qwen38-pp-lab/results/ornith15-seq101k.json'))
PREFIX = 65536
APPEND = 1000
RTX = 'GPU-30acbf2b-a41a-8b88-e298-fe66c4138b29'
BINS = {
    'upstream': pathlib.Path('/workspace/llama-upstream-head-0911/build-sm70-75-force-mmq/bin/llama-server'),
    'fork': pathlib.Path('/workspace/llama-v100-optimized/build-compare-sm70-75-force-mmq/bin/llama-server'),
}
COMMON = [
    '--ctx-size','67584','--parallel','1','--fit','off','--gpu-layers','all','--split-mode','none',
    '--flash-attn','on','--batch-size','4096','--ubatch-size','512',
    '--cache-type-k','q8_0','--cache-type-v','q8_0','--cache-ram','0','--ctx-checkpoints','0',
    '--slot-save-path',str(CACHE),'--no-warmup','--perf','--skip-chat-parsing',
    '--reasoning','off','--reasoning-format','none',
]

def post(port, path, obj=None, timeout=1200):
    data = json.dumps(obj or {}, separators=(',',':')).encode()
    req = urllib.request.Request(f'http://127.0.0.1:{port}{path}', data=data, headers={'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as f:
            return json.load(f)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f'HTTP {e.code}: ' + e.read().decode(errors='ignore')[:4000])

def health(port):
    with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=.5) as f:
        return json.load(f)

def stream(port, prompt, n_predict=1):
    obj = {'prompt':prompt,'n_predict':n_predict,'ignore_eos':True,'seed':1234,'id_slot':0,
           'cache_prompt':True,'stream':True,'return_tokens':True}
    data = json.dumps(obj,separators=(',',':')).encode()
    req = urllib.request.Request(f'http://127.0.0.1:{port}/completion', data=data, headers={'Content-Type':'application/json'})
    t0=time.perf_counter(); first=None; final=None; toks=[]
    with urllib.request.urlopen(req,timeout=1200) as f:
        for raw in f:
            line=raw.strip()
            if not line.startswith(b'data:'): continue
            d=line[5:].strip()
            if not d or d==b'[DONE]': continue
            try: z=json.loads(d)
            except Exception: continue
            ts=z.get('tokens') or []
            if first is None and (ts or z.get('content') is not None): first=time.perf_counter()
            toks.extend(ts)
            if z.get('stop'): final=z
    t1=time.perf_counter()
    if final is None: raise RuntimeError('no final event')
    return final,(first or t1),t0,t1,toks

def env_for(binpath):
    e=os.environ.copy()
    for k in list(e):
        if k.startswith(('GGML_CUDA_VOLTA_','GGML_CUDA_TURING_','GGML_CUDA_QWEN35_','LLAMA_EXPERIMENT_','LLAMA_MTP_')) or k=='GGML_CUDA_FORCE_MMQ':
            e.pop(k,None)
    e['CUDA_VISIBLE_DEVICES']=RTX
    e['LD_LIBRARY_PATH']=str(binpath.parent)
    return e

def with_server(label, key, port, fn):
    b=BINS[key]
    logp=ROOT/f'{label}.server.log'
    log=logp.open('wb')
    args=[str(b),'--model',MODEL,'--host','127.0.0.1','--port',str(port)]+COMMON
    p=subprocess.Popen(args,env=env_for(b),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    try:
        for _ in range(2400):
            if p.poll() is not None: raise RuntimeError(logp.read_text(errors='ignore')[-10000:])
            try:
                if health(port).get('status')=='ok': break
            except Exception: pass
            time.sleep(.1)
        else: raise RuntimeError('health timeout')
        return fn(port,args)
    finally:
        if p.poll() is None:
            os.killpg(p.pid,signal.SIGTERM)
            try: p.wait(30)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid,signal.SIGKILL); p.wait()
        log.close()

def prime():
    def body(port,args):
        z,first,t0,t1,toks=stream(port,SEQ[:PREFIX],1)
        sv=post(port,'/slots/0?action=save',{'filename':SLOT},600)
        if sv.get('n_saved') != PREFIX: raise RuntimeError(sv)
        out={'timings':z['timings'],'save':sv,'wall_ms':(t1-t0)*1000,
             'slot':str(CACHE/SLOT),'slot_bytes':(CACHE/SLOT).stat().st_size,'argv':args}
        (ROOT/'prime.json').write_text(json.dumps(out,indent=2)+'\n')
        print('PRIME',json.dumps(out,sort_keys=True),flush=True)
        return out
    return with_server('prime-upstream','upstream',36320,body)

def run_arm(label,key,port,reps=4):
    def body(port,args):
        rows=[]
        for rep in range(reps):
            rr=post(port,'/slots/0?action=restore',{'filename':SLOT},600)
            if rr.get('n_restored') != PREFIX: raise RuntimeError(rr)
            z,first,t0,t1,toks=stream(port,SEQ[:PREFIX+APPEND],1)
            tm=z['timings']
            if tm.get('cache_n') != PREFIX or tm.get('prompt_n') != APPEND: raise RuntimeError(tm)
            row={'rep':rep,'warmup':rep==0,'pp':tm['prompt_per_second'],'prompt_ms':tm['prompt_ms'],
                 'ttft_wall_ms':(first-t0)*1000,'e2e_wall_ms':(t1-t0)*1000,
                 'cache_n':tm.get('cache_n'),'prompt_n':tm.get('prompt_n'),
                 'token_sha':hashlib.sha256(json.dumps(toks).encode()).hexdigest()}
            rows.append(row)
            print(label,rep,round(row['pp'],2),round(row['ttft_wall_ms'],2),flush=True)
        keep=rows[1:]
        return {'label':label,'key':key,'binary':str(BINS[key]),'argv':args,'rows':rows,
                'mean_pp':statistics.mean(r['pp'] for r in keep),
                'sd_pp':statistics.stdev(r['pp'] for r in keep),
                'mean_prompt_ms':statistics.mean(r['prompt_ms'] for r in keep),
                'mean_ttft_wall_ms':statistics.mean(r['ttft_wall_ms'] for r in keep)}
    return with_server(label,key,port,body)

if not (CACHE/SLOT).exists(): prime()
arms=[]
for i,(label,key) in enumerate([('upstream-a','upstream'),('fork-a','fork'),('fork-b','fork'),('upstream-b','upstream')]):
    arms.append(run_arm(label,key,36330+i))
summary={k:{} for k in ['upstream','fork']}
for key in ['upstream','fork']:
    xs=[a for a in arms if a['key']==key]
    summary[key]['pp']=statistics.mean(a['mean_pp'] for a in xs)
    summary[key]['ttft_wall_ms']=statistics.mean(a['mean_ttft_wall_ms'] for a in xs)
    summary[key]['prompt_ms']=statistics.mean(a['mean_prompt_ms'] for a in xs)
out={'model':MODEL,'prefix':PREFIX,'append':APPEND,'ctx':67584,'ubatch':512,'arms':arms,'summary':summary}
(ROOT/'abba.json').write_text(json.dumps(out,indent=2)+'\n')
print('SUMMARY',json.dumps(summary,indent=2),flush=True)
