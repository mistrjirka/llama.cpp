#!/usr/bin/env python3
import concurrent.futures as cf, hashlib, json, os, pathlib, signal, statistics, subprocess, threading, time, urllib.request, urllib.error
R=pathlib.Path(__file__).resolve().parent
F=R.parent/'parallel-refined-0907'
fx=json.loads((F/'real-fixture.json').read_text()); H=fx['histories']; SUF=[fx['prompts'][i][100000:] for i in range(4)]
assert [len(x) for x in SUF]==[38,42,44,41]
MODEL='/models/Ornith-1.5-35B-A3B/Ornith-1.5-35B-A3B-AD-Q6_K-Q5_K.gguf'
DRAFT='/workspace/models/Ornith-1.5-35B-A3B/shisa-mtp/mtp-shisa-ornith15-all-Q5_0.gguf'
MMP='/models/local-llm-setup/ornith15/mmproj-Ornith-1.5-35B-BF16.gguf'
UP=pathlib.Path('/workspace/upstream-fair-0907/build/bin/llama-server')
FORK=pathlib.Path('/workspace/llama-v100-sync-0907/build-sm70-75/bin/llama-server')
PORT=32851
for i in range(4):
    for ext in ('','.draft','.spec'):
        p=R/f'real100k-{i}.bin{ext}'
        if not p.exists(): p.symlink_to(F/p.name)
BASE_ENV=os.environ.copy()
for k in list(BASE_ENV):
    if k.startswith(('LLAMA_EXPERIMENT_','LLAMA_MTP_','QWEN4EXP_')) or k in ('LLAMA_KV_INDEXED_RM','LLAMA_SAMPLING_VIEW','LD_PRELOAD'):
        BASE_ENV.pop(k,None)
BASE_ENV.update({'GGML_CUDA_ALLREDUCE':'internal','GGML_CUDA_AR_COPY_THRESHOLD':'131072','GGML_CUDA_TURING_CUBLAS_MIN_BATCH':'256','GGML_CUDA_VOLTA_Q8_FATTN_TC':'1','GGML_CUDA_VOLTA_Q5_X4':'1','GGML_CUDA_VOLTA_Q6_W4R4':'1','GGML_CUDA_VOLTA_FORCE_MMQ':'moe','GGML_CUDA_VOLTA_GQA8_NCOLS2':'2'})

def req(path,body=None,timeout=900):
    data=None if body is None else json.dumps(body,separators=(',',':')).encode(); q=urllib.request.Request(f'http://127.0.0.1:{PORT}{path}',data=data,headers={} if data is None else {'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(q,timeout=timeout) as f:return json.load(f)
    except urllib.error.HTTPError as e: raise RuntimeError(f'HTTP {e.code}: {e.read().decode()[:800]}') from e

def argv(binary,mode):
    a=[str(binary),'-m',MODEL,'--mmproj',MMP,'--no-mmproj-offload','--host','127.0.0.1','--port',str(PORT),'--ctx-size','1400000','--parallel','4','--kv-unified','--kv-unified-per-slot','400000','--cache-ram','0','--no-cache-idle-slots','--slot-prompt-similarity','0','--override-kv','qwen35moe.context_length=int:400000','--rope-scaling','yarn','--rope-scale','1.52587890625','--yarn-orig-ctx','262144','--split-mode','layer','--fit','off','--gpu-layers','all','--device','CUDA1,CUDA0','--tensor-split','14,35','--flash-attn','on','--batch-size','2048','--ubatch-size','256','--cache-type-k','q8_0','--cache-type-v','q8_0','--slot-save-path',str(R),'--no-warmup','--perf','--slots','--spec-type','draft-mtp','--spec-draft-model',DRAFT,'--spec-draft-device','CUDA1','--spec-draft-ngl','all','--spec-draft-type-k','q8_0','--spec-draft-type-v','q8_0','--spec-draft-n-max','3']
    if mode=='strict-fork': a += ['--no-slot-fork-prefix','--no-spec-mtp-defer-prompt','--spec-draft-ubatch','0','--pipeline-copies','0']
    elif mode=='production-fork': a += ['--slot-fork-prefix','--spec-mtp-defer-prompt','--spec-draft-ubatch','128','--pipeline-copies','1']
    return a

def start(binary,mode,label):
    env=BASE_ENV.copy(); env['LD_LIBRARY_PATH']=str(binary.parent)
    log=(R/f'ornith-{label}.log').open('wb'); p=subprocess.Popen(argv(binary,mode),env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    for _ in range(700):
        if p.poll() is not None:
            log.close(); raise RuntimeError((R/f'ornith-{label}.log').read_text(errors='ignore')[-2500:])
        try:
            if req('/health',timeout=.5).get('status')=='ok':return p,log
        except:pass
        time.sleep(.1)
    raise TimeoutError(label)

def stop(p,log):
    if p.poll() is None:
        os.killpg(p.pid,signal.SIGTERM)
        try:p.wait(25)
        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
    log.close()

def restore_all():
    for i in range(4):
        z=req(f'/slots/{i}?action=restore',{'filename':f'real100k-{i}.bin'}); assert z['n_restored']==100000,z

def turn():
    barrier=threading.Barrier(4)
    def one(i):
        barrier.wait(); z=req('/completion',{'prompt':H[i]+SUF[i],'id_slot':i,'cache_prompt':True,'n_predict':128,'ignore_eos':True,'temperature':0.0,'seed':1234,'return_tokens':True})
        tm=z['timings']; assert tm['cache_n']==100000 and tm['prompt_n']==len(SUF[i]) and tm['predicted_n']==128,(i,tm)
        return {'slot':i,'timings':tm,'sha':hashlib.sha256(','.join(map(str,z['tokens'])).encode()).hexdigest()}
    t=time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=4) as ex: out=list(ex.map(one,range(4)))
    wall=time.perf_counter()-t
    return {'wall_s':wall,'aggregate_pp_tg':677/wall,'aggregate_generated':512/wall,'mean_agent_tg':statistics.mean(x['timings']['predicted_per_second'] for x in out),'mean_agent_pp':statistics.mean(x['timings']['prompt_per_second'] for x in out),'acceptance':[[x['timings'].get('draft_n_accepted',0),x['timings'].get('draft_n',0)] for x in out],'sha':[x['sha'] for x in out]}

def run(label,binary,mode,reps=5):
    p,log=start(binary,mode,label)
    try:
        rows=[]
        for rep in range(reps):
            restore_all(); z=turn(); z['rep']=rep; z['warmup']=rep==0; rows.append(z)
            print(label,rep,round(z['wall_s'],4),round(z['aggregate_pp_tg'],2),round(z['aggregate_generated'],2),round(z['mean_agent_tg'],2),z['acceptance'],flush=True)
        keep=rows[1:]
        s={'label':label,'mode':mode,'binary':str(binary),'argv':argv(binary,mode),'rows':rows,'n_retained':len(keep)}
        for k in ('wall_s','aggregate_pp_tg','aggregate_generated','mean_agent_tg','mean_agent_pp'):
            s[k]={'mean':statistics.mean(x[k] for x in keep),'min':min(x[k] for x in keep),'max':max(x[k] for x in keep)}
        return s
    finally:stop(p,log)

arms=[]
for label,binary,mode in [('upstream-a',UP,'upstream'),('fork-strict-a',FORK,'strict-fork'),('fork-strict-b',FORK,'strict-fork'),('upstream-b',UP,'upstream')]:
    arms.append(run(label,binary,mode)); (R/'ornith-fair-raw.json').write_text(json.dumps(arms,indent=2))
prod=run('fork-production',FORK,'production-fork',5); arms.append(prod); (R/'ornith-fair-raw.json').write_text(json.dumps(arms,indent=2))

def combine(prefix):
    rr=[r for a in arms if a['label'].startswith(prefix) for r in a['rows'][1:]]
    o={'n':len(rr)}
    for k in ('wall_s','aggregate_pp_tg','aggregate_generated','mean_agent_tg','mean_agent_pp'): o[k]=statistics.mean(x[k] for x in rr)
    return o
summary={'upstream':combine('upstream-'),'fork_strict':combine('fork-strict-'),'fork_production':combine('fork-production')}
summary['strict_gain_pct']=100*(summary['fork_strict']['aggregate_pp_tg']/summary['upstream']['aggregate_pp_tg']-1)
summary['strict_wall_reduction_pct']=100*(1-summary['fork_strict']['wall_s']/summary['upstream']['wall_s'])
summary['production_gain_vs_upstream_pct']=100*(summary['fork_production']['aggregate_pp_tg']/summary['upstream']['aggregate_pp_tg']-1)
(R/'ornith-fair-summary.json').write_text(json.dumps(summary,indent=2)+'\n'); print('SUMMARY',json.dumps(summary,indent=2),flush=True)
