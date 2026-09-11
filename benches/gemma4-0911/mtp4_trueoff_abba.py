#!/usr/bin/env python3
import concurrent.futures as cf,hashlib,json,os,pathlib,signal,statistics,subprocess,threading,time,urllib.request
ROOT=pathlib.Path('/models/.bench-ornith-mtp4/trueoff');ROOT.mkdir(parents=True,exist_ok=True)
BIN=pathlib.Path('/workspace/llama-v100-optimized/build-compare-sm70-75/bin/llama-server');MODEL='/models/Ornith-1.5-35B-A3B/Ornith-1.5-35B-A3B-AD-Q6_K-Q5_K.gguf';DRAFT='/workspace/models/Ornith-1.5-35B-A3B/shisa-mtp/mtp-shisa-ornith15-all-Q5_0.gguf';SLOTDIR=pathlib.Path('/workspace/oai-qwen38-pp-lab/results/parallel-refined-0907');PROMPTS=json.load(open(SLOTDIR/'real-fixture.json'))['prompts']
def env():
 e=os.environ.copy()
 for k in list(e):
  if k.startswith(('GGML_CUDA_TURING_','GGML_CUDA_MOE_MMQ_J_','LLAMA_EXPERIMENT_')) or k=='GGML_CUDA_FORCE_MMQ':e.pop(k,None)
 e.update({'LD_LIBRARY_PATH':str(BIN.parent),'GGML_CUDA_VOLTA_FORCE_MMQ':'moe','GGML_CUDA_ALLREDUCE':'internal','GGML_CUDA_AR_COPY_THRESHOLD':'131072'});return e
def args(port,mtp):
 a=[str(BIN),'-m',MODEL,'--host','127.0.0.1','--port',str(port),'--ctx-size','1400000','--parallel','4','--kv-unified','--kv-unified-per-slot','400000','--slot-fork-prefix','--cache-ram','0','--no-cache-idle-slots','--slot-prompt-similarity','0','--override-kv','qwen35moe.context_length=int:400000','--rope-scaling','yarn','--rope-scale','1.52587890625','--yarn-orig-ctx','262144','--split-mode','layer','--fit','off','--gpu-layers','all','--device','CUDA1,CUDA0','--tensor-split','14,35','--flash-attn','on','--batch-size','2048','--ubatch-size','256','--pipeline-copies','1','--cache-type-k','q8_0','--cache-type-v','q8_0','--slot-save-path',str(SLOTDIR),'--no-warmup','--perf','--slots','--reasoning','off','--reasoning-format','none']
 if mtp:a+=['--spec-type','draft-mtp','--spec-draft-model',DRAFT,'--spec-draft-device','CUDA1','--spec-draft-ngl','all','--spec-draft-type-k','q8_0','--spec-draft-type-v','q8_0','--spec-draft-ubatch','128','--spec-draft-n-max','1','--spec-mtp-defer-prompt']
 return a
def run(label,mtp,port):
 a=args(port,mtp);logp=ROOT/f'{label}.log';log=logp.open('wb');p=subprocess.Popen(a,env=env(),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 def req(path,body=None,timeout=900):
  data=None if body is None else json.dumps(body,separators=(',',':')).encode();q=urllib.request.Request(f'http://127.0.0.1:{port}{path}',data=data,headers={} if data is None else {'Content-Type':'application/json'});return json.load(urllib.request.urlopen(q,timeout=timeout))
 try:
  for _ in range(1800):
   if p.poll() is not None:raise RuntimeError(logp.read_text(errors='ignore')[-5000:])
   try:
    if req('/health',timeout=.5).get('status')=='ok':break
   except:pass
   time.sleep(.1)
  rows=[]
  for rep in range(3):
   for i in range(4):req(f'/slots/{i}?action=erase',{})
   for i in range(4):assert req(f'/slots/{i}?action=restore',{'filename':f'real100k-{i}.bin'},600).get('n_restored')==100000
   bar=threading.Barrier(4)
   def one(i):
    bar.wait();t=time.perf_counter();r=req('/completion',{'prompt':PROMPTS[i],'id_slot':i,'cache_prompt':True,'n_predict':128,'ignore_eos':True,'temperature':0.0,'seed':1234,'return_tokens':True});w=time.perf_counter()-t;tok=r.get('tokens',[]);return {'wall_s':w,'timings':r['timings'],'sha':hashlib.sha256(','.join(map(str,tok)).encode()).hexdigest()}
   t=time.perf_counter()
   with cf.ThreadPoolExecutor(max_workers=4) as ex:o=list(ex.map(one,range(4)))
   wall=time.perf_counter()-t;row={'rep':rep,'warmup':rep==0,'wall_s':wall,'aggregate_output_tps':512/wall,'mean_tg':statistics.mean(x['timings']['predicted_per_second'] for x in o),'mean_pp':statistics.mean(x['timings']['prompt_per_second'] for x in o),'acceptance':[(x['timings'].get('draft_n_accepted',0),x['timings'].get('draft_n',0)) for x in o],'sha':[x['sha'] for x in o]};rows.append(row);print(label,row,flush=True)
  keep=rows[1:];return {'label':label,'mtp':mtp,'pp':statistics.mean(r['mean_pp'] for r in keep),'tg':statistics.mean(r['mean_tg'] for r in keep),'aggregate_output_tps':statistics.mean(r['aggregate_output_tps'] for r in keep),'wall_s':statistics.mean(r['wall_s'] for r in keep),'rows':rows,'argv':a}
 finally:
  if p.poll() is None:
   os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(20)
   except:os.killpg(p.pid,signal.SIGKILL);p.wait()
  log.close()
arms=[]
for i,(lab,mtp) in enumerate([('off-a',False),('mtp1-a',True),('mtp1-b',True),('off-b',False)]):arms.append(run(lab,mtp,39300+i))
def avg(mtp,key):return statistics.mean(a[key] for a in arms if a['mtp']==mtp)
out={'arms':arms,'off_output_tps':avg(False,'aggregate_output_tps'),'mtp1_output_tps':avg(True,'aggregate_output_tps'),'gain_pct':100*(avg(True,'aggregate_output_tps')/avg(False,'aggregate_output_tps')-1),'off_tg':avg(False,'tg'),'mtp1_tg':avg(True,'tg')};(ROOT/'abba.json').write_text(json.dumps(out,indent=2)+'\n');print('SUMMARY',json.dumps({k:v for k,v in out.items() if k!='arms'},indent=2))
