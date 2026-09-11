#!/usr/bin/env python3
import concurrent.futures as cf, hashlib, json, os, pathlib, signal, statistics, subprocess, threading, time, urllib.request, urllib.error
ROOT=pathlib.Path('/models/.bench-ornith-mtp4/depth'); ROOT.mkdir(parents=True,exist_ok=True)
BIN=pathlib.Path('/workspace/llama-v100-optimized/build-compare-sm70-75/bin/llama-server')
MODEL='/models/Ornith-1.5-35B-A3B/Ornith-1.5-35B-A3B-AD-Q6_K-Q5_K.gguf'
DRAFT='/workspace/models/Ornith-1.5-35B-A3B/shisa-mtp/mtp-shisa-ornith15-all-Q5_0.gguf'
SLOTDIR=pathlib.Path('/workspace/oai-qwen38-pp-lab/results/parallel-refined-0907')
FIX=json.load(open(SLOTDIR/'real-fixture.json')); PROMPTS=FIX['prompts']; PORT=39200
ORDER=[0,1,2,3,3,2,1,0]
def req(path,body=None,timeout=900):
 data=None if body is None else json.dumps(body,separators=(',',':')).encode(); q=urllib.request.Request(f'http://127.0.0.1:{PORT}{path}',data=data,headers={} if data is None else {'Content-Type':'application/json'})
 with urllib.request.urlopen(q,timeout=timeout) as r:return json.load(r)
def comp(slot,prompt,nmax,barrier):
 barrier.wait();t=time.perf_counter();body={'prompt':prompt,'id_slot':slot,'cache_prompt':True,'n_predict':128,'ignore_eos':True,'temperature':0.0,'seed':1234,'return_tokens':True,'speculative_n_max':nmax};r=req('/completion',body);wall=time.perf_counter()-t;tok=r.get('tokens',[]);return {'slot':slot,'wall_s':wall,'timings':r.get('timings',{}),'sha':hashlib.sha256(','.join(map(str,tok)).encode()).hexdigest()}
def restore():
 for i in range(4):req(f'/slots/{i}?action=erase',{})
 for i in range(4):
  z=req(f'/slots/{i}?action=restore',{'filename':f'real100k-{i}.bin'},600);assert z.get('n_restored')==100000,z
def env():
 e=os.environ.copy()
 for k in list(e):
  if k.startswith(('GGML_CUDA_TURING_','GGML_CUDA_MOE_MMQ_J_','LLAMA_EXPERIMENT_')) or k=='GGML_CUDA_FORCE_MMQ':e.pop(k,None)
 e.update({'LD_LIBRARY_PATH':str(BIN.parent),'GGML_CUDA_VOLTA_FORCE_MMQ':'moe','GGML_CUDA_ALLREDUCE':'internal','GGML_CUDA_AR_COPY_THRESHOLD':'131072'});return e
a=[str(BIN),'-m',MODEL,'--host','127.0.0.1','--port',str(PORT),'--ctx-size','1400000','--parallel','4','--kv-unified','--kv-unified-per-slot','400000','--slot-fork-prefix','--cache-ram','0','--no-cache-idle-slots','--slot-prompt-similarity','0','--override-kv','qwen35moe.context_length=int:400000','--rope-scaling','yarn','--rope-scale','1.52587890625','--yarn-orig-ctx','262144','--split-mode','layer','--fit','off','--gpu-layers','all','--device','CUDA1,CUDA0','--tensor-split','14,35','--flash-attn','on','--batch-size','2048','--ubatch-size','256','--pipeline-copies','1','--cache-type-k','q8_0','--cache-type-v','q8_0','--slot-save-path',str(SLOTDIR),'--no-warmup','--perf','--slots','--spec-type','draft-mtp','--spec-draft-model',DRAFT,'--spec-draft-device','CUDA1','--spec-draft-ngl','all','--spec-draft-type-k','q8_0','--spec-draft-type-v','q8_0','--spec-draft-ubatch','128','--spec-draft-n-max','3','--spec-mtp-defer-prompt','--reasoning','off','--reasoning-format','none']
log=open(ROOT/'server.log','wb');p=subprocess.Popen(a,env=env(),stdout=log,stderr=subprocess.STDOUT,start_new_session=True);rows=[]
try:
 for _ in range(1800):
  if p.poll() is not None:raise RuntimeError((ROOT/'server.log').read_text(errors='ignore')[-6000:])
  try:
   if req('/health',timeout=.5).get('status')=='ok':break
  except:pass
  time.sleep(.1)
 else:raise RuntimeError('health timeout')
 print('MEM',subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,memory.free','--format=csv,noheader,nounits'],text=True).strip().splitlines(),flush=True)
 for block,n in enumerate(ORDER):
  restore();bar=threading.Barrier(4);t=time.perf_counter()
  with cf.ThreadPoolExecutor(max_workers=4) as ex:out=list(ex.map(lambda i:comp(i,PROMPTS[i],n,bar),range(4)))
  wall=time.perf_counter()-t
  for z in out:
   tm=z['timings'];assert tm.get('cache_n')==100000 and tm.get('predicted_n')==128,(n,tm)
  row={'block':block,'nmax':n,'wall_s':wall,'aggregate_output_tps':512/wall,'mean_tg':statistics.mean(z['timings']['predicted_per_second'] for z in out),'mean_pp':statistics.mean(z['timings']['prompt_per_second'] for z in out),'acceptance':[(z['timings'].get('draft_n_accepted',0),z['timings'].get('draft_n',0)) for z in out],'sha':[z['sha'] for z in out],'requests':out};rows.append(row);(ROOT/'depth-screen.json').write_text(json.dumps({'argv':a,'rows':rows},indent=2)+'\n');print('RESULT',json.dumps({k:v for k,v in row.items() if k!='requests'}),flush=True)
finally:
 if p.poll() is None:
  os.killpg(p.pid,signal.SIGTERM)
  try:p.wait(20)
  except:os.killpg(p.pid,signal.SIGKILL);p.wait()
 log.close()
summary={}
for n in range(4):
 rs=[r for r in rows if r['nmax']==n]
 summary[n]={k:statistics.mean(r[k] for r in rs) for k in ['wall_s','aggregate_output_tps','mean_tg','mean_pp']}
 acc=sum(a for r in rs for a,d in r['acceptance']);draft=sum(d for r in rs for a,d in r['acceptance']);summary[n]['acceptance_ratio']=acc/draft if draft else 0;summary[n]['accepted']=acc;summary[n]['drafted']=draft
print('SUMMARY',json.dumps(summary,indent=2));(ROOT/'depth-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
