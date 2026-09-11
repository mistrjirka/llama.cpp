#!/usr/bin/env python3
import concurrent.futures as cf, hashlib, json, os, pathlib, signal, statistics, subprocess, threading, time, urllib.request
ROOT=pathlib.Path('/models/.bench-ornith-mtp4/draft-kv-sweep'); ROOT.mkdir(parents=True,exist_ok=True)
BIN='./build-compare-sm70-75/bin/llama-server'
MODEL='/models/Ornith-1.5-35B-A3B/Ornith-1.5-35B-A3B-AD-Q6_K-Q5_K.gguf'
DRAFT='/workspace/models/Ornith-1.5-35B-A3B/shisa-mtp/mtp-shisa-ornith15-all-q4.gguf'
S=pathlib.Path('/workspace/oai-qwen38-pp-lab/results/parallel-refined-0907'); PROMPTS=json.load(open(S/'real-fixture.json'))['prompts']
CASES=[
 ('q8-q8','q8_0','q8_0'),
 ('q5_1-q5_1','q5_1','q5_1'),
 ('q5_0-q5_0','q5_0','q5_0'),
 ('q4_1-q4_1','q4_1','q4_1'),
 ('q4_0-q4_0','q4_0','q4_0'),
 ('iq4_nl-iq4_nl','iq4_nl','iq4_nl'),
 ('q4_0-q8','q4_0','q8_0'),
 ('q8-q4_0','q8_0','q4_0'),
]
def req(port,path,body=None,timeout=900):
 d=None if body is None else json.dumps(body,separators=(',',':')).encode(); q=urllib.request.Request(f'http://127.0.0.1:{port}{path}',data=d,headers={} if d is None else {'Content-Type':'application/json'}); return json.load(urllib.request.urlopen(q,timeout=timeout))
def env():
 e=os.environ.copy()
 for k in list(e):
  if k.startswith(('GGML_CUDA_TURING_','GGML_CUDA_MOE_MMQ_J_','LLAMA_EXPERIMENT_')) or k in ('GGML_CUDA_FORCE_MMQ','LLAMA_MTP_SHARE_TARGET_IO'): e.pop(k,None)
 e.update({'LD_LIBRARY_PATH':'./build-compare-sm70-75/bin','GGML_CUDA_VOLTA_FORCE_MMQ':'moe','GGML_CUDA_ALLREDUCE':'internal','GGML_CUDA_AR_COPY_THRESHOLD':'131072','LLAMA_MTP_SHARE_TARGET_IO':'head'})
 return e
def gpu_mem():
 return subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,memory.free','--format=csv,noheader,nounits'],text=True).strip().splitlines()
def run(idx,label,kt,vt):
 port=40500+idx; logp=ROOT/f'{label}.log'
 a=[BIN,'-m',MODEL,'--host','127.0.0.1','--port',str(port),'--ctx-size','1400000','--parallel','4','--kv-unified','--kv-unified-per-slot','400000','--slot-fork-prefix','--cache-ram','0','--no-cache-idle-slots','--slot-prompt-similarity','0','--override-kv','qwen35moe.context_length=int:400000','--rope-scaling','yarn','--rope-scale','1.52587890625','--yarn-orig-ctx','262144','--split-mode','layer','--fit','off','--gpu-layers','all','--device','CUDA1,CUDA0','--tensor-split','14,35','--flash-attn','on','--batch-size','512','--ubatch-size','128','--pipeline-copies','1','--cache-type-k','q8_0','--cache-type-v','q8_0','--slot-save-path',str(S),'--no-warmup','--perf','--slots','--reasoning','off','--reasoning-format','none','--spec-type','draft-mtp','--spec-draft-model',DRAFT,'--spec-draft-device','CUDA1,CUDA0','--spec-draft-ngl','all','--spec-draft-type-k',kt,'--spec-draft-type-v',vt,'--spec-draft-ubatch','64','--spec-draft-n-max','1','--spec-mtp-defer-prompt']
 log=logp.open('wb'); p=subprocess.Popen(a,env=env(),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 try:
  for _ in range(1800):
   if p.poll() is not None: raise RuntimeError(logp.read_text(errors='ignore')[-10000:])
   try:
    if req(port,'/health',timeout=.5).get('status')=='ok': break
   except: pass
   time.sleep(.1)
  else: raise RuntimeError('health timeout')
  mem_start=gpu_mem()
  # restore once before memory sample, then keep this state for first generation
  for i in range(4):
   z=req(port,f'/slots/{i}?action=restore',{'filename':f'real100k-{i}.bin'},600); assert z.get('n_restored')==100000,z
  mem_rest=gpu_mem()
  rows=[]
  for rep in range(4):
   if rep:
    for i in range(4): req(port,f'/slots/{i}?action=erase',{})
    for i in range(4):
     z=req(port,f'/slots/{i}?action=restore',{'filename':f'real100k-{i}.bin'},600); assert z.get('n_restored')==100000,z
   bar=threading.Barrier(4)
   def one(i):
    bar.wait(); t=time.perf_counter(); r=req(port,'/completion',{'prompt':PROMPTS[i],'id_slot':i,'cache_prompt':True,'n_predict':128,'ignore_eos':True,'temperature':0.0,'seed':1234,'return_tokens':True}); w=time.perf_counter()-t; tm=r['timings']; tok=r.get('tokens',[])
    return {'wall':w,'tg':tm['predicted_per_second'],'pp':tm['prompt_per_second'],'a':tm.get('draft_n_accepted',0),'d':tm.get('draft_n',0),'sha':hashlib.sha256(','.join(map(str,tok)).encode()).hexdigest()}
   t=time.perf_counter()
   with cf.ThreadPoolExecutor(max_workers=4) as ex: rs=list(ex.map(one,range(4)))
   wall=time.perf_counter()-t; row={'rep':rep,'warmup':rep==0,'out_tps':512/wall,'wall_s':wall,'tg':statistics.mean(x['tg'] for x in rs),'pp':statistics.mean(x['pp'] for x in rs),'accepted':sum(x['a'] for x in rs),'drafted':sum(x['d'] for x in rs),'sha':[x['sha'] for x in rs]}; rows.append(row); print(label,row,flush=True)
  keep=rows[1:]; acc=sum(r['accepted'] for r in keep); drafted=sum(r['drafted'] for r in keep)
  out={'label':label,'k':kt,'v':vt,'memory_startup':mem_start,'memory_restored':mem_rest,'out_tps':statistics.mean(r['out_tps'] for r in keep),'out_stdev':statistics.stdev(r['out_tps'] for r in keep),'tg':statistics.mean(r['tg'] for r in keep),'pp':statistics.mean(r['pp'] for r in keep),'acceptance':acc/drafted if drafted else 0,'accepted':acc,'drafted':drafted,'rows':rows}
  return out
 except Exception as ex:
  return {'label':label,'k':kt,'v':vt,'error':repr(ex),'tail':logp.read_text(errors='ignore')[-10000:] if logp.exists() else ''}
 finally:
  if p.poll() is None:
   os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(20)
   except: os.killpg(p.pid,signal.SIGKILL); p.wait()
  log.close()
results=[]
for i,c in enumerate(CASES):
 r=run(i,*c); results.append(r); (ROOT/'results.json').write_text(json.dumps(results,indent=2)+'\n'); print('RESULT',json.dumps({k:v for k,v in r.items() if k not in ('rows','tail')},indent=2),flush=True)
print('SUMMARY')
for r in results:
 print(r['label'], 'ERROR '+r['error'] if 'error' in r else f"out={r['out_tps']:.3f} tg={r['tg']:.3f} acc={100*r['acceptance']:.2f}% mem={r['memory_restored'][0]}")
