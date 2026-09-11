#!/usr/bin/env python3
import concurrent.futures as cf
import hashlib, json, os, pathlib, signal, statistics, subprocess, threading, time, urllib.request

ROOT=pathlib.Path('/models/.bench-ornith-mtp4/mtp14-sweep'); ROOT.mkdir(parents=True,exist_ok=True)
BIN='/workspace/llama-v100-optimized/build-compare-sm70-75/bin/llama-server'
MODEL='/models/Ornith-1.5-35B-A3B/Ornith-1.5-35B-A3B-AD-Q6_K-Q5_K.gguf'
DRAFT='/workspace/models/Ornith-1.5-35B-A3B/shisa-mtp/mtp-shisa-ornith15-all-q4.gguf'
SLOTDIR=pathlib.Path('/workspace/oai-qwen38-pp-lab/results/parallel-refined-0907')
PROMPTS=json.load(open(SLOTDIR/'real-fixture.json'))['prompts']
CONFIGS=[
 ('mtp14-big-d128',True,'14,35',2048,256,128),
 ('mtp14-big-d64',True,'14,35',2048,256,64),
 ('mtp14-big-d32',True,'14,35',2048,256,32),
 ('mtp14-small-d64',True,'14,35',512,128,64),
 ('mtp14-small-d32',True,'14,35',512,128,32),
]
def env(mtp):
 e=os.environ.copy()
 for k in list(e):
  if k.startswith(('GGML_CUDA_TURING_','GGML_CUDA_MOE_MMQ_J_','LLAMA_EXPERIMENT_')) or k in ('GGML_CUDA_FORCE_MMQ','LLAMA_MTP_SHARE_TARGET_IO'):
   e.pop(k,None)
 e.update({'LD_LIBRARY_PATH':str(pathlib.Path(BIN).parent),'GGML_CUDA_VOLTA_FORCE_MMQ':'moe','GGML_CUDA_ALLREDUCE':'internal','GGML_CUDA_AR_COPY_THRESHOLD':'131072'})
 if mtp: e['LLAMA_MTP_SHARE_TARGET_IO']='head'
 return e

def run(cfg,idx):
 label,mtp,split,batch,ubatch,dub=cfg; port=39900+idx
 argv=[BIN,'-m',MODEL,'--host','127.0.0.1','--port',str(port),'--ctx-size','1400000','--parallel','4','--kv-unified','--kv-unified-per-slot','400000','--slot-fork-prefix','--cache-ram','0','--no-cache-idle-slots','--slot-prompt-similarity','0','--override-kv','qwen35moe.context_length=int:400000','--rope-scaling','yarn','--rope-scale','1.52587890625','--yarn-orig-ctx','262144','--split-mode','layer','--fit','off','--gpu-layers','all','--device','CUDA1,CUDA0','--tensor-split',split,'--flash-attn','on','--batch-size',str(batch),'--ubatch-size',str(ubatch),'--pipeline-copies','1','--cache-type-k','q8_0','--cache-type-v','q8_0','--slot-save-path',str(SLOTDIR),'--no-warmup','--perf','--slots','--reasoning','off','--reasoning-format','none']
 if mtp:
  argv += ['--spec-type','draft-mtp','--spec-draft-model',DRAFT,'--spec-draft-device','CUDA1,CUDA0','--spec-draft-ngl','all','--spec-draft-type-k','q8_0','--spec-draft-type-v','q8_0','--spec-draft-ubatch',str(dub),'--spec-draft-n-max','1','--spec-mtp-defer-prompt']
 logp=ROOT/f'{label}.log'; log=logp.open('wb'); p=subprocess.Popen(argv,env=env(mtp),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 def req(path,body=None,timeout=900):
  data=None if body is None else json.dumps(body,separators=(',',':')).encode();q=urllib.request.Request(f'http://127.0.0.1:{port}{path}',data=data,headers={} if data is None else {'Content-Type':'application/json'});return json.load(urllib.request.urlopen(q,timeout=timeout))
 try:
  for _ in range(1800):
   if p.poll() is not None: raise RuntimeError(logp.read_text(errors='ignore')[-6000:])
   try:
    if req('/health',timeout=.5).get('status')=='ok': break
   except: pass
   time.sleep(.1)
  else: raise RuntimeError('health timeout')
  mem=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,memory.free','--format=csv,noheader,nounits'],text=True).strip().splitlines()
  rows=[]
  for rep in range(3):
   for i in range(4): req(f'/slots/{i}?action=erase',{})
   for i in range(4):
    z=req(f'/slots/{i}?action=restore',{'filename':f'real100k-{i}.bin'},600); assert z.get('n_restored')==100000,z
   bar=threading.Barrier(4)
   def one(i):
    bar.wait();t=time.perf_counter();body={'prompt':PROMPTS[i],'id_slot':i,'cache_prompt':True,'n_predict':128,'ignore_eos':True,'temperature':0.0,'seed':1234,'return_tokens':True}
    if mtp: body['speculative_n_max']=1
    r=req('/completion',body); wall=time.perf_counter()-t; tok=r.get('tokens',[])
    return {'slot':i,'wall_s':wall,'timings':r['timings'],'sha':hashlib.sha256(','.join(map(str,tok)).encode()).hexdigest()}
   t=time.perf_counter()
   with cf.ThreadPoolExecutor(max_workers=4) as ex: out=list(ex.map(one,range(4)))
   wall=time.perf_counter()-t
   for z in out: assert z['timings'].get('cache_n')==100000 and z['timings'].get('predicted_n')==128,(label,z['timings'])
   row={'rep':rep,'warmup':rep==0,'wall_s':wall,'out_tps':512/wall,'tg':statistics.mean(z['timings']['predicted_per_second'] for z in out),'pp':statistics.mean(z['timings']['prompt_per_second'] for z in out),'acc':[(z['timings'].get('draft_n_accepted',0),z['timings'].get('draft_n',0)) for z in out],'sha':[z['sha'] for z in out]}; rows.append(row); print(label,row,flush=True)
  keep=rows[1:]; acc=sum(a for r in keep for a,d in r['acc']); drafted=sum(d for r in keep for a,d in r['acc'])
  return {'label':label,'mtp':mtp,'split':split,'batch':batch,'ubatch':ubatch,'draft_ubatch':dub,'memory':mem,'out_tps':statistics.mean(r['out_tps'] for r in keep),'tg':statistics.mean(r['tg'] for r in keep),'pp':statistics.mean(r['pp'] for r in keep),'wall_s':statistics.mean(r['wall_s'] for r in keep),'acceptance':acc/drafted if drafted else 0,'accepted':acc,'drafted':drafted,'rows':rows,'argv':argv}
 finally:
  if p.poll() is None:
   os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(20)
   except: os.killpg(p.pid,signal.SIGKILL);p.wait()
  log.close()

results=[]
for i,c in enumerate(CONFIGS):
 try:r=run(c,i)
 except Exception as e:r={'label':c[0],'error':repr(e)}
 results.append(r); (ROOT/'results.json').write_text(json.dumps(results,indent=2)+'\n'); print('RESULT',json.dumps({k:v for k,v in r.items() if k not in ('rows','argv')}),flush=True)
print('SUMMARY')
for r in results:
 if 'error' in r: print(r['label'],'ERR',r['error'])
 else: print(f"{r['label']:12s} {r['out_tps']:8.3f} out/s  tg={r['tg']:6.2f} pp={r['pp']:7.2f} acc={r['acceptance']:.4f} mem={r['memory'][0]}")
