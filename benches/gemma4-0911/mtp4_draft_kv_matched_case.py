#!/usr/bin/env python3
import concurrent.futures as cf,hashlib,json,os,pathlib,signal,statistics,subprocess,threading,time,urllib.request,sys
label,kt,vt=sys.argv[1:4]; split=sys.argv[4] if len(sys.argv)>4 else '14,35'
ROOT=pathlib.Path('/models/.bench-ornith-mtp4/draft-kv-matched')/label;ROOT.mkdir(parents=True,exist_ok=True)
BIN='./build-compare-sm70-75/bin/llama-server';MODEL='/models/Ornith-1.5-35B-A3B/Ornith-1.5-35B-A3B-AD-Q6_K-Q5_K.gguf';DRAFT='/workspace/models/Ornith-1.5-35B-A3B/shisa-mtp/mtp-shisa-ornith15-all-q4.gguf'
STATE=pathlib.Path('/models/.bench-ornith-mtp4/draft-kv-states')/label
FIX=pathlib.Path('/workspace/oai-qwen38-pp-lab/results/parallel-refined-0907');PROMPTS=json.load(open(FIX/'real-fixture.json'))['prompts'];port=40700

def req(path,body=None,timeout=900):
 d=None if body is None else json.dumps(body,separators=(',',':')).encode();q=urllib.request.Request(f'http://127.0.0.1:{port}{path}',data=d,headers={} if d is None else {'Content-Type':'application/json'});return json.load(urllib.request.urlopen(q,timeout=timeout))
def mem():return subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,memory.free','--format=csv,noheader,nounits'],text=True).strip().splitlines()
e=os.environ.copy()
for k in list(e):
 if k.startswith(('GGML_CUDA_TURING_','GGML_CUDA_MOE_MMQ_J_','LLAMA_EXPERIMENT_')) or k in ('GGML_CUDA_FORCE_MMQ','LLAMA_MTP_SHARE_TARGET_IO'):e.pop(k,None)
e.update({'LD_LIBRARY_PATH':'./build-compare-sm70-75/bin','GGML_CUDA_VOLTA_FORCE_MMQ':'moe','GGML_CUDA_ALLREDUCE':'internal','GGML_CUDA_AR_COPY_THRESHOLD':'131072','LLAMA_MTP_SHARE_TARGET_IO':'head'})
a=[BIN,'-m',MODEL,'--host','127.0.0.1','--port',str(port),'--ctx-size','1400000','--parallel','4','--kv-unified','--kv-unified-per-slot','400000','--slot-fork-prefix','--cache-ram','0','--no-cache-idle-slots','--slot-prompt-similarity','0','--override-kv','qwen35moe.context_length=int:400000','--rope-scaling','yarn','--rope-scale','1.52587890625','--yarn-orig-ctx','262144','--split-mode','layer','--fit','off','--gpu-layers','all','--device','CUDA1,CUDA0','--tensor-split',split,'--flash-attn','on','--batch-size','512','--ubatch-size','128','--pipeline-copies','1','--cache-type-k','q8_0','--cache-type-v','q8_0','--slot-save-path',str(STATE),'--no-warmup','--perf','--slots','--reasoning','off','--reasoning-format','none','--spec-type','draft-mtp','--spec-draft-model',DRAFT,'--spec-draft-device','CUDA1,CUDA0','--spec-draft-ngl','all','--spec-draft-type-k',kt,'--spec-draft-type-v',vt,'--spec-draft-ubatch','64','--spec-draft-n-max','1','--spec-mtp-defer-prompt']
logp=ROOT/'server.log';log=logp.open('wb');p=subprocess.Popen(a,env=e,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
try:
 for _ in range(1800):
  if p.poll() is not None:raise RuntimeError(logp.read_text(errors='ignore')[-12000:])
  try:
   if req('/health',timeout=.5).get('status')=='ok':break
  except:pass
  time.sleep(.1)
 ms=mem(); print('MEM_START',ms,flush=True)
 rows=[]; mr=None
 for rep in range(5):
  for i in range(4):req(f'/slots/{i}?action=erase',{})
  for i in range(4):
   z=req(f'/slots/{i}?action=restore',{'filename':f'real100k-{i}.bin'},600);assert z.get('n_restored')==100000,z
  if rep==0: mr=mem();print('MEM_REST',mr,flush=True)
  bar=threading.Barrier(4)
  def one(i):
   bar.wait();t=time.perf_counter();r=req('/completion',{'prompt':PROMPTS[i],'id_slot':i,'cache_prompt':True,'n_predict':128,'ignore_eos':True,'temperature':0.0,'seed':1234,'return_tokens':True});w=time.perf_counter()-t;tm=r['timings'];tok=r.get('tokens',[]);return {'wall':w,'tg':tm['predicted_per_second'],'pp':tm['prompt_per_second'],'a':tm.get('draft_n_accepted',0),'d':tm.get('draft_n',0),'sha':hashlib.sha256(','.join(map(str,tok)).encode()).hexdigest()}
  t=time.perf_counter()
  with cf.ThreadPoolExecutor(max_workers=4) as ex:rs=list(ex.map(one,range(4)))
  wall=time.perf_counter()-t;row={'rep':rep,'warmup':rep==0,'out_tps':512/wall,'wall_s':wall,'tg':statistics.mean(x['tg'] for x in rs),'pp':statistics.mean(x['pp'] for x in rs),'accepted':sum(x['a'] for x in rs),'drafted':sum(x['d'] for x in rs),'sha':[x['sha'] for x in rs]};rows.append(row);print('ROW',json.dumps(row),flush=True)
 keep=rows[1:];aa=sum(r['accepted'] for r in keep);dd=sum(r['drafted'] for r in keep)
 out={'label':label,'k':kt,'v':vt,'memory_startup':ms,'memory_restored':mr,'out_tps':statistics.mean(r['out_tps'] for r in keep),'stdev':statistics.stdev(r['out_tps'] for r in keep),'tg':statistics.mean(r['tg'] for r in keep),'pp':statistics.mean(r['pp'] for r in keep),'acceptance':aa/dd,'accepted':aa,'drafted':dd,'rows':rows,'argv':a};(ROOT/'result.json').write_text(json.dumps(out,indent=2)+'\n');print('RESULT',json.dumps({k:v for k,v in out.items() if k not in ('rows','argv')},indent=2),flush=True)
finally:
 if p.poll() is None:
  os.killpg(p.pid,signal.SIGTERM)
  try:p.wait(20)
  except:os.killpg(p.pid,signal.SIGKILL);p.wait()
 log.close()
