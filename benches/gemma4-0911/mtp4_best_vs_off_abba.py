#!/usr/bin/env python3
import concurrent.futures as cf, hashlib, json, os, pathlib, signal, statistics, subprocess, threading, time, urllib.request
ROOT=pathlib.Path('/models/.bench-ornith-mtp4/best-vs-off-abba'); ROOT.mkdir(parents=True,exist_ok=True)
BIN='/workspace/llama-v100-optimized/build-compare-sm70-75/bin/llama-server'; MODEL='/models/Ornith-1.5-35B-A3B/Ornith-1.5-35B-A3B-AD-Q6_K-Q5_K.gguf'; DRAFT='/workspace/models/Ornith-1.5-35B-A3B/shisa-mtp/mtp-shisa-ornith15-all-q4.gguf'; S=pathlib.Path('/workspace/oai-qwen38-pp-lab/results/parallel-refined-0907'); PROMPTS=json.load(open(S/'real-fixture.json'))['prompts']
ARMS=[('off-a',False),('mtp-a',True),('mtp-b',True),('off-b',False)]
def env(mtp):
 e=os.environ.copy()
 for k in list(e):
  if k.startswith(('GGML_CUDA_TURING_','GGML_CUDA_MOE_MMQ_J_','LLAMA_EXPERIMENT_')) or k in ('GGML_CUDA_FORCE_MMQ','LLAMA_MTP_SHARE_TARGET_IO'):e.pop(k,None)
 e.update({'LD_LIBRARY_PATH':str(pathlib.Path(BIN).parent),'GGML_CUDA_VOLTA_FORCE_MMQ':'moe','GGML_CUDA_ALLREDUCE':'internal','GGML_CUDA_AR_COPY_THRESHOLD':'131072'})
 if mtp:e['LLAMA_MTP_SHARE_TARGET_IO']='head'
 return e
def run(label,mtp,port):
 split='14,35' if mtp else '18,31'; batch=512 if mtp else 2048; ub=128 if mtp else 256
 a=[BIN,'-m',MODEL,'--host','127.0.0.1','--port',str(port),'--ctx-size','1400000','--parallel','4','--kv-unified','--kv-unified-per-slot','400000','--slot-fork-prefix','--cache-ram','0','--no-cache-idle-slots','--slot-prompt-similarity','0','--override-kv','qwen35moe.context_length=int:400000','--rope-scaling','yarn','--rope-scale','1.52587890625','--yarn-orig-ctx','262144','--split-mode','layer','--fit','off','--gpu-layers','all','--device','CUDA1,CUDA0','--tensor-split',split,'--flash-attn','on','--batch-size',str(batch),'--ubatch-size',str(ub),'--pipeline-copies','1','--cache-type-k','q8_0','--cache-type-v','q8_0','--slot-save-path',str(S),'--no-warmup','--perf','--slots','--reasoning','off','--reasoning-format','none']
 if mtp:a += ['--spec-type','draft-mtp','--spec-draft-model',DRAFT,'--spec-draft-device','CUDA1,CUDA0','--spec-draft-ngl','all','--spec-draft-type-k','q8_0','--spec-draft-type-v','q8_0','--spec-draft-ubatch','64','--spec-draft-n-max','1','--spec-mtp-defer-prompt']
 logp=ROOT/f'{label}.log';log=logp.open('wb');p=subprocess.Popen(a,env=env(mtp),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 def req(path,body=None,timeout=900):
  d=None if body is None else json.dumps(body,separators=(',',':')).encode();q=urllib.request.Request(f'http://127.0.0.1:{port}{path}',data=d,headers={} if d is None else {'Content-Type':'application/json'});return json.load(urllib.request.urlopen(q,timeout=timeout))
 try:
  for _ in range(1800):
   if p.poll() is not None:raise RuntimeError(logp.read_text(errors='ignore')[-6000:])
   try:
    if req('/health',timeout=.5).get('status')=='ok':break
   except:pass
   time.sleep(.1)
  mem=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,memory.free','--format=csv,noheader,nounits'],text=True).strip().splitlines();rows=[]
  for rep in range(3):
   for i in range(4):req(f'/slots/{i}?action=erase',{})
   for i in range(4): assert req(f'/slots/{i}?action=restore',{'filename':f'real100k-{i}.bin'},600).get('n_restored')==100000
   bar=threading.Barrier(4)
   def one(i):
    bar.wait();t=time.perf_counter();body={'prompt':PROMPTS[i],'id_slot':i,'cache_prompt':True,'n_predict':128,'ignore_eos':True,'temperature':0.0,'seed':1234,'return_tokens':True}
    if mtp:body['speculative_n_max']=1
    r=req('/completion',body); tok=r.get('tokens',[]);return {'wall':time.perf_counter()-t,'tm':r['timings'],'sha':hashlib.sha256(','.join(map(str,tok)).encode()).hexdigest()}
   t=time.perf_counter();
   with cf.ThreadPoolExecutor(max_workers=4) as ex:o=list(ex.map(one,range(4)))
   wall=time.perf_counter()-t; row={'rep':rep,'warmup':rep==0,'wall_s':wall,'out_tps':512/wall,'tg':statistics.mean(x['tm']['predicted_per_second'] for x in o),'pp':statistics.mean(x['tm']['prompt_per_second'] for x in o),'acc':[(x['tm'].get('draft_n_accepted',0),x['tm'].get('draft_n',0)) for x in o],'sha':[x['sha'] for x in o]};rows.append(row);print(label,row,flush=True)
  keep=rows[1:];acc=sum(x for r in keep for x,y in r['acc']);draft=sum(y for r in keep for x,y in r['acc']);return {'label':label,'mtp':mtp,'memory':mem,'out_tps':statistics.mean(r['out_tps'] for r in keep),'tg':statistics.mean(r['tg'] for r in keep),'pp':statistics.mean(r['pp'] for r in keep),'wall_s':statistics.mean(r['wall_s'] for r in keep),'acceptance':acc/draft if draft else 0,'rows':rows,'argv':a}
 finally:
  if p.poll() is None:
   os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(20)
   except:os.killpg(p.pid,signal.SIGKILL);p.wait()
  log.close()
arms=[]
for i,(lab,mtp) in enumerate(ARMS):
 r=run(lab,mtp,40000+i);arms.append(r);(ROOT/'results.json').write_text(json.dumps(arms,indent=2)+'\n');print('ARM',lab,r['out_tps'],r['tg'],r['acceptance'],flush=True)
def avg(m,key):return statistics.mean(r[key] for r in arms if r['mtp']==m)
off=avg(False,'out_tps');mtp=avg(True,'out_tps');out={'off_out_tps':off,'mtp_out_tps':mtp,'gain_pct':100*(mtp/off-1),'off_tg':avg(False,'tg'),'mtp_tg':avg(True,'tg'),'tg_gain_pct':100*(avg(True,'tg')/avg(False,'tg')-1),'mtp_acceptance':statistics.mean(r['acceptance'] for r in arms if r['mtp'])};(ROOT/'summary.json').write_text(json.dumps(out,indent=2)+'\n');print('SUMMARY',json.dumps(out,indent=2),flush=True)
