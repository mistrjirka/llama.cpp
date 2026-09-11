#!/usr/bin/env python3
import json,os,pathlib,signal,subprocess,time,urllib.request,re
ROOT=pathlib.Path('/models/.bench-ornith-mtp4/layer-memory');ROOT.mkdir(parents=True,exist_ok=True)
BIN=pathlib.Path('/workspace/llama-v100-optimized/build-compare-sm70-75/bin/llama-server')
MODEL='/models/Ornith-1.5-35B-A3B/Ornith-1.5-35B-A3B-AD-Q6_K-Q5_K.gguf';DRAFT='/workspace/models/Ornith-1.5-35B-A3B/shisa-mtp/mtp-shisa-ornith15-all-Q5_0.gguf';SLOTDIR=pathlib.Path('/workspace/oai-qwen38-pp-lab/results/parallel-refined-0907')
CONFIGS=[]
for split in ['14,35','15,34','16,33','17,32']:
 for dub in [32,64,128]: CONFIGS.append((split,dub))
def req(port,path,body=None,timeout=600):
 data=None if body is None else json.dumps(body,separators=(',',':')).encode();q=urllib.request.Request(f'http://127.0.0.1:{port}{path}',data=data,headers={} if data is None else {'Content-Type':'application/json'});return json.load(urllib.request.urlopen(q,timeout=timeout))
def env():
 e=os.environ.copy()
 for k in list(e):
  if k.startswith(('GGML_CUDA_TURING_','GGML_CUDA_MOE_MMQ_J_','LLAMA_EXPERIMENT_')) or k=='GGML_CUDA_FORCE_MMQ':e.pop(k,None)
 e.update({'LD_LIBRARY_PATH':str(BIN.parent),'GGML_CUDA_VOLTA_FORCE_MMQ':'moe','GGML_CUDA_ALLREDUCE':'internal','GGML_CUDA_AR_COPY_THRESHOLD':'131072'});return e
def args(port,split,dub):return [str(BIN),'-m',MODEL,'--host','127.0.0.1','--port',str(port),'--ctx-size','1400000','--parallel','4','--kv-unified','--kv-unified-per-slot','400000','--slot-fork-prefix','--cache-ram','0','--no-cache-idle-slots','--slot-prompt-similarity','0','--override-kv','qwen35moe.context_length=int:400000','--rope-scaling','yarn','--rope-scale','1.52587890625','--yarn-orig-ctx','262144','--split-mode','layer','--fit','off','--gpu-layers','all','--device','CUDA1,CUDA0','--tensor-split',split,'--flash-attn','on','--batch-size','2048','--ubatch-size','256','--pipeline-copies','1','--cache-type-k','q8_0','--cache-type-v','q8_0','--slot-save-path',str(SLOTDIR),'--no-warmup','--perf','--slots','--spec-type','draft-mtp','--spec-draft-model',DRAFT,'--spec-draft-device','CUDA1','--spec-draft-ngl','all','--spec-draft-type-k','q8_0','--spec-draft-type-v','q8_0','--spec-draft-ubatch',str(dub),'--spec-draft-n-max','1','--spec-mtp-defer-prompt','--reasoning','off','--reasoning-format','none']
rows=[]
for i,(split,dub) in enumerate(CONFIGS):
 label=f's{split.replace(",","-")}-du{dub}';port=39400+i;logp=ROOT/f'{label}.log';log=logp.open('wb');a=args(port,split,dub);p=subprocess.Popen(a,env=env(),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 try:
  ok=False
  for _ in range(1200):
   if p.poll() is not None:break
   try:
    if req(port,'/health',timeout=.5).get('status')=='ok':ok=True;break
   except:pass
   time.sleep(.1)
  if not ok:
   tail=logp.read_text(errors='ignore')[-5000:];r={'split':split,'draft_ubatch':dub,'error':'startup','returncode':p.poll(),'tail':tail};rows.append(r);print(label,'FAIL',flush=True);continue
  mem=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,memory.free','--format=csv,noheader,nounits'],text=True).strip().splitlines()
  for s in range(4):
   z=req(port,f'/slots/{s}?action=restore',{'filename':f'real100k-{s}.bin'},600);assert z.get('n_restored')==100000,z
  mem2=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,memory.free','--format=csv,noheader,nounits'],text=True).strip().splitlines()
  r={'split':split,'draft_ubatch':dub,'memory_startup':mem,'memory_restored':mem2};rows.append(r);print(label,'OK',mem2,flush=True)
 finally:
  if p.poll() is None:
   os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(20)
   except:os.killpg(p.pid,signal.SIGKILL);p.wait()
  log.close();(ROOT/'screen.json').write_text(json.dumps(rows,indent=2)+'\n')
print('SUMMARY',json.dumps(rows,indent=2))
