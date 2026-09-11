#!/usr/bin/env python3
import json,os,pathlib,signal,subprocess,time,urllib.request
ROOT=pathlib.Path('/models/.bench-ornith-mtp4/best-capacity');ROOT.mkdir(parents=True,exist_ok=True)
BIN='/workspace/llama-v100-optimized/build-compare-sm70-75/bin/llama-server'
MODEL='/models/Ornith-1.5-35B-A3B/Ornith-1.5-35B-A3B-AD-Q6_K-Q5_K.gguf'
DRAFT='/workspace/models/Ornith-1.5-35B-A3B/shisa-mtp/mtp-shisa-ornith15-all-q4.gguf'
SLOTS='/workspace/oai-qwen38-pp-lab/results/parallel-refined-0907'
CONFIGS=[]
for sp in ['14,35','15,34','16,33','17,32','18,31']:
 for dub in [64,32]: CONFIGS.append(('mtp',sp,dub))
for sp in ['14,35','16,33','18,31','19,30','20,29','21,28','22,27','23,26']:
 CONFIGS.append(('off',sp,0))
def req(port,path,body=None,timeout=600):
 d=None if body is None else json.dumps(body,separators=(',',':')).encode();q=urllib.request.Request(f'http://127.0.0.1:{port}{path}',data=d,headers={} if d is None else {'Content-Type':'application/json'});return json.load(urllib.request.urlopen(q,timeout=timeout))
def env(mtp):
 e=os.environ.copy()
 for k in list(e):
  if k.startswith(('GGML_CUDA_TURING_','GGML_CUDA_MOE_MMQ_J_','LLAMA_EXPERIMENT_')) or k in ('GGML_CUDA_FORCE_MMQ','LLAMA_MTP_SHARE_TARGET_IO'):e.pop(k,None)
 e.update({'LD_LIBRARY_PATH':str(pathlib.Path(BIN).parent),'GGML_CUDA_VOLTA_FORCE_MMQ':'moe','GGML_CUDA_ALLREDUCE':'internal','GGML_CUDA_AR_COPY_THRESHOLD':'131072'})
 if mtp:e['LLAMA_MTP_SHARE_TARGET_IO']='head'
 return e
def args(port,mode,sp,dub):
 a=[BIN,'-m',MODEL,'--host','127.0.0.1','--port',str(port),'--ctx-size','1400000','--parallel','4','--kv-unified','--kv-unified-per-slot','400000','--slot-fork-prefix','--cache-ram','0','--no-cache-idle-slots','--slot-prompt-similarity','0','--override-kv','qwen35moe.context_length=int:400000','--rope-scaling','yarn','--rope-scale','1.52587890625','--yarn-orig-ctx','262144','--split-mode','layer','--fit','off','--gpu-layers','all','--device','CUDA1,CUDA0','--tensor-split',sp,'--flash-attn','on','--batch-size','512','--ubatch-size','128','--pipeline-copies','1','--cache-type-k','q8_0','--cache-type-v','q8_0','--slot-save-path',SLOTS,'--no-warmup','--perf','--slots','--reasoning','off','--reasoning-format','none']
 if mode=='mtp':a += ['--spec-type','draft-mtp','--spec-draft-model',DRAFT,'--spec-draft-device','CUDA1,CUDA0','--spec-draft-ngl','all','--spec-draft-type-k','q8_0','--spec-draft-type-v','q8_0','--spec-draft-ubatch',str(dub),'--spec-draft-n-max','1','--spec-mtp-defer-prompt']
 return a
rows=[]
for ix,(mode,sp,dub) in enumerate(CONFIGS):
 port=39600+ix;label=f'{mode}-{sp.replace(",","-")}-d{dub}';lp=ROOT/f'{label}.log';log=lp.open('wb');p=subprocess.Popen(args(port,mode,sp,dub),env=env(mode=='mtp'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 try:
  ok=False
  for _ in range(900):
   if p.poll() is not None:break
   try:
    if req(port,'/health',timeout=.3).get('status')=='ok':ok=True;break
   except:pass
   time.sleep(.1)
  if not ok:
   tail=lp.read_text(errors='ignore')[-5000:];rows.append({'label':label,'mode':mode,'split':sp,'draft_ubatch':dub,'ok':False,'tail':tail});print(label,'FAIL',flush=True);continue
  mem=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,memory.free','--format=csv,noheader,nounits'],text=True).strip().splitlines()
  restored=[]
  for i in range(4): restored.append(req(port,f'/slots/{i}?action=restore',{'filename':f'real100k-{i}.bin'},600).get('n_restored'))
  mem2=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,memory.free','--format=csv,noheader,nounits'],text=True).strip().splitlines()
  r={'label':label,'mode':mode,'split':sp,'draft_ubatch':dub,'ok':True,'memory_startup':mem,'memory_restored':mem2,'restored':restored};rows.append(r);print(label,'OK',mem2,flush=True)
 finally:
  if p.poll() is None:
   os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(20)
   except:os.killpg(p.pid,signal.SIGKILL);p.wait()
  log.close();(ROOT/'capacity.json').write_text(json.dumps(rows,indent=2)+'\n')
print('DONE')
