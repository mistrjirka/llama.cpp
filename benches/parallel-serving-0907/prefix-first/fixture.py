#!/usr/bin/env python3
"""Controlled concurrent Ornith MTP screen. Never changes production settings."""
import concurrent.futures as cf
import hashlib, json, os, pathlib, signal, subprocess, threading, time, urllib.request, urllib.error
OUT=pathlib.Path(os.environ.get('PARALLEL_FIXTURE_DIR', '/workspace/oai-qwen38-pp-lab/results/parallel-research-0907'))
BIN='/workspace/llama-multiagent-cache/build-sm70-75/bin/llama-server'
MODEL='/models/Ornith-1.5-35B-A3B/Ornith-1.5-35B-A3B-AD-Q6_K-Q5_K.gguf'
DRAFT='/workspace/models/Ornith-1.5-35B-A3B/shisa-mtp/mtp-shisa-ornith15-all-Q5_0.gguf'
MMP='/models/local-llm-setup/ornith15/mmproj-Ornith-1.5-35B-BF16.gguf'
SEQ=json.load(open('/workspace/oai-qwen38-pp-lab/results/ornith15-seq101k.json'))
PARENT=SEQ[:98000]
H=[PARENT+[110+i]+SEQ[i*19000:i*19000+1999] for i in range(4)]
PROMPTS=[h+SEQ[81000+i*1000:82000+i*1000] for i,h in enumerate(H)]
PORT=32465
ENV=os.environ.copy(); ENV.update({'GGML_CUDA_ALLREDUCE':'internal','GGML_CUDA_AR_COPY_THRESHOLD':'131072','GGML_CUDA_TURING_CUBLAS_MIN_BATCH':'256','GGML_CUDA_VOLTA_Q8_FATTN_TC':'1','GGML_CUDA_VOLTA_Q5_X4':'1','GGML_CUDA_VOLTA_Q6_W4R4':'1','GGML_CUDA_VOLTA_FORCE_MMQ':'moe','GGML_CUDA_VOLTA_GQA8_NCOLS2':'2'})
def req(path,body=None,timeout=900):
    data=None if body is None else json.dumps(body,separators=(',',':')).encode()
    q=urllib.request.Request(f'http://127.0.0.1:{PORT}{path}',data=data,headers={} if data is None else {'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(q,timeout=timeout) as r:return json.load(r)
    except urllib.error.HTTPError as e:raise RuntimeError(f'HTTP {e.code}: {e.read().decode()[:1000]}') from e
def comp(slot,prompt,n=0,barrier=None):
    if barrier:barrier.wait()
    t=time.perf_counter()
    r=req('/completion',{'prompt':prompt,'id_slot':slot,'cache_prompt':True,'n_predict':n,'ignore_eos':True,'temperature':0.0,'seed':1234,'return_tokens':True})
    toks=r.get('tokens',[])
    return {'slot':slot,'wall_s':time.perf_counter()-t,'timings':r.get('timings',{}),'sha':hashlib.sha256(','.join(map(str,toks)).encode()).hexdigest(),'tokens':toks}
def args(n):
    a=[BIN,'-m',MODEL,'--mmproj',MMP,'--no-mmproj-offload','--host','127.0.0.1','--port',str(PORT),'--ctx-size','1400000','--parallel','4','--kv-unified','--kv-unified-per-slot','400000','--slot-fork-prefix','--cache-ram','0','--no-cache-idle-slots','--slot-prompt-similarity','0','--override-kv','qwen35moe.context_length=int:400000','--rope-scaling','yarn','--rope-scale','1.52587890625','--yarn-orig-ctx','262144','--split-mode','layer','--fit','off','--gpu-layers','all','--device','CUDA1,CUDA0','--tensor-split','14,35','--flash-attn','on','--batch-size','2048','--ubatch-size','256','--pipeline-copies','1','--cache-type-k','q8_0','--cache-type-v','q8_0','--slot-save-path',str(OUT),'--no-warmup','--perf','--slots']
    if n:a+=['--spec-type','draft-mtp','--spec-draft-model',DRAFT,'--spec-draft-device','CUDA1','--spec-draft-ngl','all','--spec-draft-type-k','q8_0','--spec-draft-type-v','q8_0','--spec-draft-ubatch','128','--spec-draft-n-max',str(n),'--spec-mtp-defer-prompt']
    return a
def start(n,label):
    log=open(OUT/f'{label}.log','wb'); p=subprocess.Popen(args(n),env=ENV,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    for _ in range(600):
        if p.poll() is not None:log.close();raise RuntimeError(f'{label} server died {p.returncode}')
        try:
            if req('/health',timeout=.5).get('status')=='ok':return p,log
        except Exception:pass
        time.sleep(.1)
    stop(p,log);raise TimeoutError('server startup')
def stop(p,log):
    if p.poll() is None:
        os.killpg(p.pid,signal.SIGTERM)
        try:p.wait(20)
        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
    log.close()
def restore():
    for i in range(4):req(f'/slots/{i}?action=erase',{})
    for i in range(4):
        r=req(f'/slots/{i}?action=restore',{'filename':f'base{i}.bin'})
        assert r['n_restored']==100000,r

def main():
    rows=[]
    metadata={'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'weights':MODEL,'draft':DRAFT,'target_kv':'q8_0/q8_0','draft_kv':'q8_0/q8_0','shared_prefix':98000,'cached_per_slot':100000,'append_per_slot':1000,'generated_per_request':256,'physical_pool':1400000,'logical_per_slot':400000,'env':{k:v for k,v in ENV.items() if k.startswith('GGML_')},'corpus':'deterministic text tokens, not a representative quality workload'}
    (OUT/'metadata.json').write_text(json.dumps(metadata,indent=2))
    for index,n in enumerate([3,1,2,0,3]):
        label=f'n{n}-r{index}';p,log=start(n,label)
        try:
            if index==0:
                print('restoring saved shared 98000-token parent',flush=True)
                restored=req('/slots/0?action=restore',{'filename':'parent98k.bin'})
                assert restored['n_restored']==98000
                prime={'timings':restored}
                print('parent',json.dumps({k:v for k,v in prime.items() if k!='tokens'}),flush=True)
                for i in [1,2,3,0]:
                    z=comp(i,H[i]);print('branch',i,z['timings'],flush=True)
                    assert z['timings']['cache_n']==98000,z['timings']
                for i in range(4):req(f'/slots/{i}?action=save',{'filename':f'base{i}.bin'})
            for parallel in [1,4]:
                restore()
                barrier=threading.Barrier(parallel)
                t=time.perf_counter()
                with cf.ThreadPoolExecutor(max_workers=parallel) as ex:
                    futures=[ex.submit(comp,i,PROMPTS[i],256,barrier) for i in range(parallel)]
                    measurements=[f.result() for f in futures]
                wall=time.perf_counter()-t
                for z in measurements:
                    assert z['timings']['cache_n']==100000,z['timings']
                    assert z['timings']['prompt_n']==1000,z['timings']
                    assert z['timings']['predicted_n']==256,z['timings']
                row={'nmax':n,'repetition':index,'concurrent':parallel,'wall_s':wall,'aggregate_output_tps_including_pp':parallel*256/wall,'rows':measurements}
                rows.append(row);(OUT/'results.json').write_text(json.dumps(rows,indent=2))
                print('RESULT',json.dumps({**{k:v for k,v in row.items() if k!='rows'},'pp_each':[z['timings']['prompt_per_second'] for z in measurements],'tg_each':[z['timings']['predicted_per_second'] for z in measurements],'accepted_each':[(z['timings'].get('draft_n_accepted'),z['timings'].get('draft_n')) for z in measurements],'sha_each':[z['sha'] for z in measurements]}),flush=True)
        finally:stop(p,log)
    print('DONE',flush=True)
if __name__=='__main__':main()
