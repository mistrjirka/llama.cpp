#!/usr/bin/env python3
import json, os, pathlib, signal, subprocess, time, urllib.request, urllib.error
ROOT=pathlib.Path('/models/.bench-ornith-mtp4'); ROOT.mkdir(parents=True,exist_ok=True)
BIN=pathlib.Path('/workspace/llama-v100-optimized/build-compare-sm70-75-force-mmq/bin/llama-server')
MODEL='/models/Ornith-1.5-35B-A3B/Ornith-1.5-35B-A3B-AD-Q6_K-Q5_K.gguf'
DRAFT='/workspace/models/Ornith-1.5-35B-A3B/shisa-mtp/mtp-shisa-ornith15-all-Q5_0.gguf'
SLOTDIR=pathlib.Path('/workspace/oai-qwen38-pp-lab/results/parallel-refined-0907')
CONFIGS=[('layer-14-35-u256','layer','14,35',256),('tensor-1-1-u512','tensor','1,1',512),('tensor-1-1-u1024','tensor','1,1',1024),('tensor-4-5-u1024','tensor','4,5',1024),('tensor-5-4-u1024','tensor','5,4',1024)]
def req(port,path,body=None,timeout=600):
    data=None if body is None else json.dumps(body,separators=(',',':')).encode()
    q=urllib.request.Request(f'http://127.0.0.1:{port}{path}',data=data,headers={} if data is None else {'Content-Type':'application/json'})
    with urllib.request.urlopen(q,timeout=timeout) as r:return json.load(r)
def mem():
    return subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,memory.free','--format=csv,noheader,nounits'],text=True).strip().splitlines()
def args(port,mode,split,ub):
    return [str(BIN),'-m',MODEL,'--host','127.0.0.1','--port',str(port),'--ctx-size','1400000','--parallel','4','--kv-unified','--kv-unified-per-slot','400000','--slot-fork-prefix','--cache-ram','0','--no-cache-idle-slots','--slot-prompt-similarity','0','--override-kv','qwen35moe.context_length=int:400000','--rope-scaling','yarn','--rope-scale','1.52587890625','--yarn-orig-ctx','262144','--split-mode',mode,'--fit','off','--gpu-layers','all','--device','CUDA1,CUDA0','--tensor-split',split,'--flash-attn','on','--batch-size','2048','--ubatch-size',str(ub),'--pipeline-copies','1','--cache-type-k','q8_0','--cache-type-v','q8_0','--slot-save-path',str(SLOTDIR),'--no-warmup','--perf','--slots','--spec-type','draft-mtp','--spec-draft-model',DRAFT,'--spec-draft-device','CUDA1','--spec-draft-ngl','all','--spec-draft-type-k','q8_0','--spec-draft-type-v','q8_0','--spec-draft-ubatch','128','--spec-draft-n-max','3','--spec-mtp-defer-prompt','--reasoning','off','--reasoning-format','none']
def clean_env():
    e=os.environ.copy()
    for k in list(e):
        if k.startswith(('GGML_CUDA_VOLTA_','GGML_CUDA_TURING_','GGML_CUDA_MOE_MMQ_J_','LLAMA_EXPERIMENT_')) or k=='GGML_CUDA_FORCE_MMQ': e.pop(k,None)
    e['LD_LIBRARY_PATH']=str(BIN.parent);e['GGML_CUDA_ALLREDUCE']='internal';e['GGML_CUDA_AR_COPY_THRESHOLD']='131072'
    return e
out=[]
for i,(label,mode,split,ub) in enumerate(CONFIGS):
    port=39100+i; logp=ROOT/f'{label}.log'; log=logp.open('wb'); a=args(port,mode,split,ub)
    p=subprocess.Popen(a,env=clean_env(),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    row={'label':label,'mode':mode,'split':split,'ubatch':ub,'argv':a}
    try:
        ok=False
        for _ in range(1800):
            if p.poll() is not None:break
            try:
                if req(port,'/health',timeout=.5).get('status')=='ok':ok=True;break
            except Exception:pass
            time.sleep(.1)
        if not ok:
            row['error']='startup';row['returncode']=p.poll();row['tail']=logp.read_text(errors='ignore')[-6000:];print(label,'FAILED startup',row['returncode'],flush=True);out.append(row);continue
        row['memory_startup']=mem(); restored=[]
        for slot in range(4):
            r=req(port,f'/slots/{slot}?action=restore',{'filename':f'real100k-{slot}.bin'},600); restored.append({'slot':slot,'n_restored':r.get('n_restored'),'timings':r.get('timings')})
        row['restored']=restored;row['memory_restored']=mem();print(label,'OK',row['memory_startup'],'->',row['memory_restored'],flush=True)
        out.append(row)
    except Exception as e:
        row['error']=repr(e);row['tail']=logp.read_text(errors='ignore')[-6000:];out.append(row);print(label,'ERROR',e,flush=True)
    finally:
        if p.poll() is None:
            os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(20)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        log.close(); (ROOT/'topology-screen.json').write_text(json.dumps(out,indent=2)+'\n')
print('SUMMARY',json.dumps(out,indent=2))
