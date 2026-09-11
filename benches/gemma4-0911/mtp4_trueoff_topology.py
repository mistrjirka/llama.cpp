#!/usr/bin/env python3
import json,os,pathlib,signal,subprocess,time,urllib.request,re
ROOT=pathlib.Path('/models/.bench-ornith-mtp4/trueoff-topology');ROOT.mkdir(parents=True,exist_ok=True)
BIN=pathlib.Path('/workspace/llama-v100-optimized/build-compare-sm70-75/bin/llama-server')
MODEL='/models/Ornith-1.5-35B-A3B/Ornith-1.5-35B-A3B-AD-Q6_K-Q5_K.gguf'; SLOTDIR=pathlib.Path('/workspace/oai-qwen38-pp-lab/results/parallel-refined-0907')
CONFIGS=[('layer-14-35-b2048-u256','layer','14,35',2048,256),('layer-16-33-b2048-u256','layer','16,33',2048,256),('layer-18-31-b2048-u256','layer','18,31',2048,256),('tensor-1-1-b512-u128','tensor','1,1',512,128),('tensor-1-1-b512-u256','tensor','1,1',512,256),('tensor-4-5-b512-u256','tensor','4,5',512,256),('tensor-3-5-b512-u256','tensor','3,5',512,256)]
def req(port,path,body=None,timeout=600):
 data=None if body is None else json.dumps(body,separators=(',',':')).encode();q=urllib.request.Request(f'http://127.0.0.1:{port}{path}',data=data,headers={} if data is None else {'Content-Type':'application/json'});return json.load(urllib.request.urlopen(q,timeout=timeout))
def env(bin):
 e=os.environ.copy()
 for k in list(e):
  if k.startswith(('GGML_CUDA_TURING_','GGML_CUDA_MOE_MMQ_J_','LLAMA_EXPERIMENT_')) or k=='GGML_CUDA_FORCE_MMQ':e.pop(k,None)
 e.update({'LD_LIBRARY_PATH':str(bin.parent),'GGML_CUDA_VOLTA_FORCE_MMQ':'moe','GGML_CUDA_ALLREDUCE':'internal','GGML_CUDA_AR_COPY_THRESHOLD':'131072'});return e
def args(port,mode,split,batch,ub): return [str(BIN),'-m',MODEL,'--host','127.0.0.1','--port',str(port),'--ctx-size','1400000','--parallel','4','--kv-unified','--kv-unified-per-slot','400000','--slot-fork-prefix','--cache-ram','0','--no-cache-idle-slots','--slot-prompt-similarity','0','--override-kv','qwen35moe.context_length=int:400000','--rope-scaling','yarn','--rope-scale','1.52587890625','--yarn-orig-ctx','262144','--split-mode',mode,'--fit','off','--gpu-layers','all','--device','CUDA1,CUDA0','--tensor-split',split,'--flash-attn','on','--batch-size',str(batch),'--ubatch-size',str(ub),'--pipeline-copies','1','--cache-type-k','q8_0','--cache-type-v','q8_0','--slot-save-path',str(SLOTDIR),'--no-warmup','--perf','--slots','--reasoning','off','--reasoning-format','none']
rows=[]
for i,c in enumerate(CONFIGS):
 label,mode,split,batch,ub=c;port=39300+i;logp=ROOT/f'{label}.log';log=logp.open('wb');a=args(port,mode,split,batch,ub);p=subprocess.Popen(a,env=env(BIN),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 try:
  ok=False
  for _ in range(1800):
   if p.poll() is not None: break
   try:
    if req(port,'/health',timeout=.5).get('status')=='ok':ok=True;break
   except:pass
   time.sleep(.1)
  if not ok:
   tail=logp.read_text(errors='ignore')[-7000:];r={'label':label,'mode':mode,'split':split,'batch':batch,'ubatch':ub,'error':'startup','returncode':p.poll(),'tail':tail};print(label,'FAIL',flush=True);rows.append(r);continue
  mem0=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,memory.free','--format=csv,noheader,nounits'],text=True).strip().splitlines()
  rest=[]
  for s in range(4):
   z=req(port,f'/slots/{s}?action=restore',{'filename':f'real100k-{s}.bin'});rest.append(z.get('n_restored'))
  mem1=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,memory.free','--format=csv,noheader,nounits'],text=True).strip().splitlines()
  r={'label':label,'mode':mode,'split':split,'batch':batch,'ubatch':ub,'memory_startup':mem0,'memory_restored':mem1,'restored':rest,'argv':a};print(label,'OK',mem1,flush=True);rows.append(r)
 finally:
  if p.poll() is None:
   os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(20)
   except:os.killpg(p.pid,signal.SIGKILL);p.wait()
  log.close();(ROOT/'screen.json').write_text(json.dumps(rows,indent=2)+'\n')
print('SUMMARY',json.dumps(rows,indent=2))
