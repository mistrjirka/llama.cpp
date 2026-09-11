#!/usr/bin/env python3
import json,os,pathlib,signal,subprocess,time,urllib.request,sys
label,kt,vt=sys.argv[1:4]
BIN='./build-compare-sm70-75/bin/llama-server';MODEL='/models/Ornith-1.5-35B-A3B/Ornith-1.5-35B-A3B-AD-Q6_K-Q5_K.gguf';DRAFT='/workspace/models/Ornith-1.5-35B-A3B/shisa-mtp/mtp-shisa-ornith15-all-q4.gguf'
SRC=pathlib.Path('/workspace/oai-qwen38-pp-lab/results/parallel-refined-0907'); F=json.load(open(SRC/'real-fixture.json')); H=F['histories']
ROOT=pathlib.Path('/models/.bench-ornith-mtp4/draft-kv-states')/label; ROOT.mkdir(parents=True,exist_ok=True)
for x in ROOT.glob('real100k-*'): x.unlink()
port=40640
e=os.environ.copy()
for k in list(e):
 if k.startswith(('GGML_CUDA_TURING_','GGML_CUDA_MOE_MMQ_J_','LLAMA_EXPERIMENT_')) or k in ('GGML_CUDA_FORCE_MMQ','LLAMA_MTP_SHARE_TARGET_IO'): e.pop(k,None)
e.update({'LD_LIBRARY_PATH':'./build-compare-sm70-75/bin','GGML_CUDA_VOLTA_FORCE_MMQ':'moe','LLAMA_MTP_SHARE_TARGET_IO':'head'})
a=[BIN,'-m',MODEL,'--host','127.0.0.1','--port',str(port),'--ctx-size','131072','--parallel','1','--fit','off','--gpu-layers','all','--device','CUDA0','--split-mode','none','--flash-attn','on','--batch-size','4096','--ubatch-size','4096','--pipeline-copies','1','--cache-type-k','q8_0','--cache-type-v','q8_0','--slot-save-path',str(ROOT),'--no-warmup','--perf','--slots','--reasoning','off','--reasoning-format','none','--spec-type','draft-mtp','--spec-draft-model',DRAFT,'--spec-draft-device','CUDA1,CUDA0','--spec-draft-ngl','all','--spec-draft-type-k',kt,'--spec-draft-type-v',vt,'--spec-draft-ubatch','512','--spec-draft-n-max','1']
logp=ROOT/'generate.log';log=logp.open('wb');p=subprocess.Popen(a,env=e,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
def req(path,body=None,timeout=1200):
 d=None if body is None else json.dumps(body,separators=(',',':')).encode(); q=urllib.request.Request(f'http://127.0.0.1:{port}{path}',data=d,headers={} if d is None else {'Content-Type':'application/json'}); return json.load(urllib.request.urlopen(q,timeout=timeout))
try:
 for _ in range(1800):
  if p.poll() is not None: raise RuntimeError(logp.read_text(errors='ignore')[-16000:])
  try:
   if req('/health',timeout=.5).get('status')=='ok':break
  except: pass
  time.sleep(.1)
 print('MEM_START',subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,memory.free','--format=csv,noheader,nounits'],text=True).strip().splitlines(),flush=True)
 for i,h in enumerate(H):
  t=time.perf_counter();z=req('/completion',{'prompt':h,'id_slot':0,'cache_prompt':True,'n_predict':0,'temperature':0.0,'seed':1234,'return_tokens':True},1800);print('PREFILL',i,'wall',time.perf_counter()-t,'timings',z.get('timings'),flush=True)
  # Verify draft really reached the exact prompt end before saving.
  t=time.perf_counter();sv=req('/slots/0?action=save',{'filename':f'real100k-{i}.bin'},1800);print('SAVE',i,'wall',time.perf_counter()-t,sv,flush=True)
  dst=ROOT/f'real100k-{i}.bin';dst.unlink();dst.symlink_to(SRC/f'real100k-{i}.bin')
 print('FILES',flush=True)
 for x in sorted(ROOT.glob('real100k-*')):print(x.name,('-> '+str(x.resolve())) if x.is_symlink() else x.stat().st_size,flush=True)
finally:
 if p.poll() is None:
  os.killpg(p.pid,signal.SIGTERM)
  try:p.wait(20)
  except:os.killpg(p.pid,signal.SIGKILL);p.wait()
 log.close()
