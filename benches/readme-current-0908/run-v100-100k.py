#!/usr/bin/env python3
import hashlib,json,os,pathlib,signal,statistics,subprocess,time,urllib.request,urllib.error
R=pathlib.Path(__file__).resolve().parent
SEQ=json.load(open('/workspace/oai-qwen38-pp-lab/results/cache100k-sharedstate-ab/seq.json'))
SLOT=pathlib.Path('/workspace/oai-qwen38-pp-lab/results/sync-master-20260828/cache-v3-regenerated')
MODEL='/models/Qwen3.8-27B/Qwen3.8-27B-UD-Q5_K_XL.gguf'
GPU='GPU-1d64d3a4-4aea-87ec-9048-33a5217efd79'
BINS={
 'upstream':pathlib.Path('/workspace/upstream-fair-0907/build/bin/llama-server'),
 'fork':pathlib.Path('/workspace/llama-v100-sync-0907/build-sm70-75/bin/llama-server'),
}
WIDTHS=[128,256,1000]; ARMS=['upstream','fork','fork','upstream']; REPS=3

def post(port,path,obj=None,timeout=900):
 d=None if obj is None else json.dumps(obj,separators=(',',':')).encode(); q=urllib.request.Request(f'http://127.0.0.1:{port}{path}',data=d,headers={} if d is None else {'Content-Type':'application/json'})
 try:
  with urllib.request.urlopen(q,timeout=timeout) as f:return json.load(f)
 except urllib.error.HTTPError as e:raise RuntimeError(f'HTTP {e.code}: {e.read().decode()[:1000]}')

def env_for(b):
 e=os.environ.copy(); e['CUDA_VISIBLE_DEVICES']=GPU; e['LD_LIBRARY_PATH']=str(b.parent)
 for k in list(e):
  if k.startswith(('GGML_CUDA_VOLTA_','GGML_CUDA_QWEN35_','QWEN4EXP_','LLAMA_EXPERIMENT_','LLAMA_MTP_')) or k in ('GGML_CUDA_GRAPH_OPT','GGML_CUDA_TURING_CUBLAS_MIN_BATCH','GGML_CUDA_PREFETCH_WEIGHTS'):
   e.pop(k,None)
 return e

def run_req(port,width):
 rr=post(port,'/slots/0?action=restore',{'filename':'cache100k.bin'}); assert rr.get('n_restored')==100000,rr
 z=post(port,'/completion',{'prompt':SEQ[:100000+width],'n_predict':1,'ignore_eos':True,'temperature':0.0,'seed':1234,'id_slot':0,'cache_prompt':True,'stream':False,'return_tokens':True})
 t=z['timings']; assert t.get('cache_n')==100000 and t.get('prompt_n')==width,t
 toks=z.get('tokens') or []
 return {'width':width,'pp':t['prompt_per_second'],'prompt_ms':t['prompt_ms'],'cache_n':t['cache_n'],'prompt_n':t['prompt_n'],'token_sha':hashlib.sha256(','.join(map(str,toks)).encode()).hexdigest()}

def arm(name,idx):
 b=BINS[name];port=32820+idx; log=R/f'v100-100k-{idx}-{name}.log'
 args=[str(b),'--model',MODEL,'--host','127.0.0.1','--port',str(port),'--ctx-size','131072','--parallel','1','--fit','off','--gpu-layers','all','--split-mode','none','--flash-attn','on','--batch-size','2048','--ubatch-size','1024','--cache-type-k','q8_0','--cache-type-v','q8_0','--cache-ram','0','--ctx-checkpoints','32','--slot-save-path',str(SLOT),'--no-warmup','--perf']
 with log.open('wb') as lf:p=subprocess.Popen(args,env=env_for(b),stdout=lf,stderr=subprocess.STDOUT,start_new_session=True)
 try:
  for _ in range(1200):
   if p.poll() is not None:raise RuntimeError(log.read_text(errors='ignore')[-3000:])
   try:
    if post(port,'/health',timeout=.3).get('status')=='ok':break
   except:pass
   time.sleep(.1)
  out=[]
  for w in WIDTHS:
   warm=run_req(port,w)
   for rep in range(REPS):
    x=run_req(port,w); x.update(arm=idx,name=name,rep=rep);out.append(x);print(name,idx,w,rep,round(x['pp'],2),flush=True)
  return {'name':name,'arm':idx,'argv':args,'rows':out}
 finally:
  if p.poll() is None:
   os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(20)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()

allarms=[]
for i,n in enumerate(ARMS):
 x=arm(n,i);allarms.append(x);(R/'v100-100k-raw.json').write_text(json.dumps(allarms,indent=2))
summary={}
for w in WIDTHS:
 u=[r for a in allarms if a['name']=='upstream' for r in a['rows'] if r['width']==w]; f=[r for a in allarms if a['name']=='fork' for r in a['rows'] if r['width']==w]
 upp=statistics.mean(r['pp'] for r in u);fpp=statistics.mean(r['pp'] for r in f);ums=statistics.mean(r['prompt_ms'] for r in u);fms=statistics.mean(r['prompt_ms'] for r in f)
 summary[str(w)]={'upstream_pp':upp,'fork_pp':fpp,'gain_pct':100*(fpp/upp-1),'upstream_ms':ums,'fork_ms':fms,'n_per_side':len(u),'output_exact':len({r['token_sha'] for r in u+f})==1}
(R/'v100-100k-summary.json').write_text(json.dumps(summary,indent=2)+'\n');print('SUMMARY',json.dumps(summary,indent=2),flush=True)
