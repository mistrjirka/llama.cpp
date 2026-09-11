#!/usr/bin/env python3
import concurrent.futures as cf, json, os, pathlib, signal, statistics, subprocess, threading, time, urllib.request, traceback
BIN='./build-compare-sm70-75/bin/llama-server'; MODEL='/models/Ornith-1.5-35B-A3B/Ornith-1.5-35B-A3B-AD-Q6_K-Q5_K.gguf'; DRAFT='/workspace/models/Ornith-1.5-35B-A3B/shisa-mtp/mtp-shisa-ornith15-all-q4.gguf'; S=pathlib.Path('/workspace/oai-qwen38-pp-lab/results/parallel-refined-0907'); PROMPTS=json.load(open(S/'real-fixture.json'))['prompts']; ROOT=pathlib.Path('/models/.bench-ornith-mtp4/stability-matrix');ROOT.mkdir(parents=True,exist_ok=True)
CONFIGS=[('head-share',True,True),('plain-mtp',True,False),('no-mtp',False,False)]
def base_env(head):
 e=os.environ.copy()
 for k in list(e):
  if k.startswith(('GGML_CUDA_TURING_','GGML_CUDA_MOE_MMQ_J_','LLAMA_EXPERIMENT_')) or k in ('GGML_CUDA_FORCE_MMQ','LLAMA_MTP_SHARE_TARGET_IO'):e.pop(k,None)
 e.update({'LD_LIBRARY_PATH':'./build-compare-sm70-75/bin','GGML_CUDA_VOLTA_FORCE_MMQ':'moe','GGML_CUDA_ALLREDUCE':'internal','GGML_CUDA_AR_COPY_THRESHOLD':'131072'})
 if head:e['LLAMA_MTP_SHARE_TARGET_IO']='head'
 return e
def run(label,mtp,head,port):
 a=[BIN,'-m',MODEL,'--host','127.0.0.1','--port',str(port),'--ctx-size','1400000','--parallel','4','--kv-unified','--kv-unified-per-slot','400000','--slot-fork-prefix','--cache-ram','0','--no-cache-idle-slots','--slot-prompt-similarity','0','--override-kv','qwen35moe.context_length=int:400000','--rope-scaling','yarn','--rope-scale','1.52587890625','--yarn-orig-ctx','262144','--split-mode','layer','--fit','off','--gpu-layers','all','--device','CUDA1,CUDA0','--tensor-split','14,35','--flash-attn','on','--batch-size','512','--ubatch-size','128','--pipeline-copies','1','--cache-type-k','q8_0','--cache-type-v','q8_0','--slot-save-path',str(S),'--no-warmup','--perf','--slots','--reasoning','off','--reasoning-format','none']
 if mtp:
  a += ['--spec-type','draft-mtp','--spec-draft-model',DRAFT,'--spec-draft-device',('CUDA1,CUDA0' if head else 'CUDA1'),'--spec-draft-ngl','all','--spec-draft-type-k','q8_0','--spec-draft-type-v','q8_0','--spec-draft-ubatch','64','--spec-draft-n-max','1','--spec-mtp-defer-prompt']
 logp=ROOT/f'{label}.log'; log=logp.open('wb'); p=subprocess.Popen(a,env=base_env(head),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 def req(path,body=None,timeout=900):
  d=None if body is None else json.dumps(body,separators=(',',':')).encode();q=urllib.request.Request(f'http://127.0.0.1:{port}{path}',data=d,headers={} if d is None else {'Content-Type':'application/json'});return json.load(urllib.request.urlopen(q,timeout=timeout))
 out={'label':label,'rounds':[],'crashed':False}
 try:
  for _ in range(1800):
   if p.poll() is not None: raise RuntimeError('startup exit')
   try:
    if req('/health',timeout=.5).get('status')=='ok':break
   except:pass
   time.sleep(.1)
  for rep in range(10):
   if p.poll() is not None: raise RuntimeError(f'process exited before round {rep}')
   for i in range(4):req(f'/slots/{i}?action=erase',{})
   for i in range(4):
    z=req(f'/slots/{i}?action=restore',{'filename':f'real100k-{i}.bin'},600); assert z.get('n_restored')==100000,(rep,i,z)
   bar=threading.Barrier(4)
   def one(i):
    bar.wait();t=time.perf_counter();body={'prompt':PROMPTS[i],'id_slot':i,'cache_prompt':True,'n_predict':128,'ignore_eos':True,'temperature':0.0,'seed':1234}
    if mtp:body['speculative_n_max']=1
    r=req('/completion',body);return time.perf_counter()-t,r['timings']
   t=time.perf_counter()
   with cf.ThreadPoolExecutor(max_workers=4) as ex: rs=list(ex.map(one,range(4)))
   wall=time.perf_counter()-t; acc=sum(x[1].get('draft_n_accepted',0) for x in rs); draft=sum(x[1].get('draft_n',0) for x in rs); row={'rep':rep,'out_tps':512/wall,'tg':statistics.mean(x[1]['predicted_per_second'] for x in rs),'acc':acc/draft if draft else 0};out['rounds'].append(row);print(label,row,flush=True)
 except Exception as e:
  out['crashed']=True;out['error']=repr(e);out['round_failed']=len(out['rounds']);print(label,'CRASH',out['round_failed'],repr(e),flush=True)
 finally:
  if p.poll() is None:
   os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(20)
   except:os.killpg(p.pid,signal.SIGKILL);p.wait()
  log.close();out['exit_code']=p.returncode;out['log_tail']=logp.read_text(errors='ignore')[-5000:];(ROOT/f'{label}.json').write_text(json.dumps(out,indent=2)+'\n')
 return out
allr=[]
for i,c in enumerate(CONFIGS):
 r=run(*c,40100+i);allr.append(r);(ROOT/'summary.json').write_text(json.dumps(allr,indent=2)+'\n')
print('FINAL')
for r in allr:print(r['label'],'crashed=',r['crashed'],'rounds=',len(r['rounds']),'mean=',statistics.mean(x['out_tps'] for x in r['rounds']) if r['rounds'] else None,'exit=',r['exit_code'])
