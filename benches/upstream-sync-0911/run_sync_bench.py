#!/usr/bin/env python3
import argparse, hashlib, json, os, pathlib, signal, statistics, subprocess, time, urllib.request, urllib.error

P=argparse.ArgumentParser()
P.add_argument('--case',choices=['qwen-v100','qwen-2080','qwen-dual','ornith-v100'],required=True)
P.add_argument('--matrix',choices=['sync','qwen-mmq','qwen-moe-abba','ornith-mmq','upstream'],required=True)
P.add_argument('--reps',type=int,default=3)
P.add_argument('--port',type=int,default=36100)
a=P.parse_args()

ROOT=pathlib.Path('/workspace/llama-v100-optimized/benches/upstream-sync-0911')
BINS={
 'pre': pathlib.Path('/workspace/llama-v100-optimized/build-presync-sm70-75/bin/llama-server'),
 'post': pathlib.Path('/workspace/llama-v100-optimized/build-compare-sm70-75/bin/llama-server'),
 'force': pathlib.Path('/workspace/llama-v100-optimized/build-compare-sm70-75-force-mmq/bin/llama-server'),
 'upstream': pathlib.Path('/workspace/llama-upstream-head-0911/build-sm70-75/bin/llama-server'),
}
V100='GPU-1d64d3a4-4aea-87ec-9048-33a5217efd79'
QSEQ=json.load(open('/workspace/oai-qwen38-pp-lab/results/cache100k-sharedstate-ab/seq.json'))
OSEQ=json.load(open('/workspace/oai-qwen38-pp-lab/results/ornith15-seq101k.json'))

if a.case=='qwen-v100':
 model='/models/Qwen3.8-27B/Qwen3.8-27B-UD-Q5_K_XL.gguf'; seq=QSEQ; prefix=100000; append=1000; npred=64
 slot_dir=pathlib.Path('/workspace/oai-qwen38-pp-lab/results/sync-master-20260828/cache-v3-regenerated'); slot='cache100k.bin'
 common=['--ctx-size','131072','--parallel','1','--fit','off','--gpu-layers','all','--split-mode','none','--flash-attn','on','--batch-size','4096','--ubatch-size','4096','--cache-type-k','q8_0','--cache-type-v','q8_0','--cache-ram','0','--ctx-checkpoints','32','--slot-save-path',str(slot_dir),'--no-warmup','--perf','--skip-chat-parsing']
 case_env={'CUDA_VISIBLE_DEVICES':V100}
elif a.case=='qwen-2080':
 model='/models/Qwen3.8-27B/Qwen3.8-27B-UD-Q5_K_XL.gguf'; seq=QSEQ; prefix=65536; append=1000; npred=64
 slot_dir=pathlib.Path('/models/.bench-qwen2080-sync'); slot='cache65536.bin'
 common=['--ctx-size','67584','--parallel','1','--fit','off','--gpu-layers','all','--split-mode','none','--flash-attn','on','--batch-size','4096','--ubatch-size','2048','--cache-type-k','q8_0','--cache-type-v','q8_0','--cache-ram','0','--ctx-checkpoints','0','--slot-save-path',str(slot_dir),'--no-warmup','--perf','--skip-chat-parsing']
 case_env={'CUDA_VISIBLE_DEVICES':'GPU-30acbf2b-a41a-8b88-e298-fe66c4138b29'}
elif a.case=='qwen-dual':
 model='/models/Qwen3.8-27B/Qwen3.8-27B-UD-Q5_K_XL.gguf'; seq=QSEQ; prefix=100000; append=1000; npred=64
 slot_dir=pathlib.Path('/workspace/llama-q8attn-validate/benches/q8attn-validate/cache-yarn400'); slot='cache100k.bin'
 common=['--ctx-size','409600','--override-kv','qwen35.context_length=int:409600','--rope-scaling','yarn','--rope-scale','1.5625','--yarn-orig-ctx','262144','--parallel','1','--fit','off','--gpu-layers','all','--split-mode','tensor','--device','CUDA1,CUDA0','--tensor-split','4,5','--flash-attn','on','--batch-size','4096','--ubatch-size','2048','--cache-type-k','q8_0','--cache-type-v','q8_0','--cache-ram','0','--ctx-checkpoints','32','--checkpoint-min-step','8192','--slot-save-path',str(slot_dir),'--no-warmup','--perf','--skip-chat-parsing']
 case_env={'GGML_CUDA_ALLREDUCE':'internal','GGML_CUDA_AR_COPY_THRESHOLD':'131072'}
else:
 model='/models/Ornith-1.5-35B-A3B/Ornith-1.5-35B-A3B-AD-Q6_K-Q5_K.gguf'; seq=OSEQ; prefix=100000; append=1000; npred=64
 slot_dir=pathlib.Path('/models/.bench-ornith-sync'); slot='ornith100k.bin'
 common=['--ctx-size','131072','--parallel','1','--fit','off','--gpu-layers','all','--split-mode','none','--flash-attn','on','--batch-size','2048','--ubatch-size','512','--cache-type-k','q8_0','--cache-type-v','q8_0','--cache-ram','0','--ctx-checkpoints','0','--slot-save-path',str(slot_dir),'--no-warmup','--perf','--skip-chat-parsing','--reasoning','off','--reasoning-format','none']
 case_env={'CUDA_VISIBLE_DEVICES':V100}

assert (slot_dir/slot).exists(), slot_dir/slot

def post(port,path,obj=None,timeout=1200):
 data=json.dumps(obj or {},separators=(',',':')).encode(); req=urllib.request.Request(f'http://127.0.0.1:{port}{path}',data=data,headers={'Content-Type':'application/json'})
 try:
  with urllib.request.urlopen(req,timeout=timeout) as f:return json.load(f)
 except urllib.error.HTTPError as e: raise RuntimeError(f'HTTP {e.code}: '+e.read().decode(errors='ignore')[:3000])

def health(port):
 with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=.5) as f:return json.load(f)

def stream(port,prompt,npred):
 obj={'prompt':prompt,'n_predict':npred,'ignore_eos':True,'temperature':0.0,'seed':1234,'id_slot':0,'cache_prompt':True,'stream':True,'return_tokens':True}
 data=json.dumps(obj,separators=(',',':')).encode(); req=urllib.request.Request(f'http://127.0.0.1:{port}/completion',data=data,headers={'Content-Type':'application/json'})
 t0=time.perf_counter(); first=None; final=None; toks=[]
 with urllib.request.urlopen(req,timeout=1200) as f:
  for raw in f:
   line=raw.strip()
   if not line.startswith(b'data:'): continue
   d=line[5:].strip()
   if not d or d==b'[DONE]': continue
   try:z=json.loads(d)
   except Exception:continue
   ts=z.get('tokens') or []
   if first is None and (ts or z.get('content') is not None): first=time.perf_counter()
   toks.extend(ts)
   if z.get('stop'): final=z
 t1=time.perf_counter()
 if final is None: raise RuntimeError('no final event')
 return final,(first or t1),t0,t1,toks

def arm_env(label,binpath):
 e=os.environ.copy()
 for k in list(e):
  if k.startswith(('GGML_CUDA_VOLTA_','GGML_CUDA_TURING_','GGML_CUDA_QWEN35_','LLAMA_EXPERIMENT_','LLAMA_MTP_')) or k=='GGML_CUDA_FORCE_MMQ': e.pop(k,None)
 e.update(case_env); e['LD_LIBRARY_PATH']=str(binpath.parent)
 if ('moe' in label and binpath == BINS['post']) or (a.case=='ornith-v100' and label not in ('post-force','force')):
  e['GGML_CUDA_VOLTA_FORCE_MMQ']='moe'
 return e

def run(label,bin_key,port):
 b=BINS[bin_key]; assert b.exists(),b
 env=arm_env(label,b)
 logp=ROOT/f'{a.case}-{a.matrix}-{label}.server.log'; log=logp.open('wb')
 args=[str(b),'--model',model,'--host','127.0.0.1','--port',str(port)]+common
 p=subprocess.Popen(args,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 try:
  for _ in range(2400):
   if p.poll() is not None: raise RuntimeError(logp.read_text(errors='ignore')[-10000:])
   try:
    if health(port).get('status')=='ok': break
   except Exception: pass
   time.sleep(.1)
  else: raise RuntimeError('health timeout')
  rows=[]
  for rep in range(a.reps+1):
   rr=post(port,'/slots/0?action=restore',{'filename':slot},600); assert rr.get('n_restored')==prefix,rr
   z,first,t0,t1,toks=stream(port,seq[:prefix+append],npred); tm=z['timings']; assert tm.get('cache_n')==prefix and tm.get('prompt_n')==append,(label,tm)
   row={'rep':rep,'warmup':rep==0,'pp':tm['prompt_per_second'],'prompt_ms':tm['prompt_ms'],'tg':tm.get('predicted_per_second'),'predicted_ms':tm.get('predicted_ms'),'predicted_n':tm.get('predicted_n'),'ttft_wall_ms':(first-t0)*1000,'e2e_wall_ms':(t1-t0)*1000,'token_sha':hashlib.sha256(json.dumps(toks).encode()).hexdigest()}
   print(label,rep,round(row['pp'],2),round(row['tg'] or 0,2),round(row['ttft_wall_ms'],1),round(row['e2e_wall_ms'],1),flush=True)
   if rep: rows.append(row)
  return {'label':label,'bin_key':bin_key,'binary':str(b),'env':{k:v for k,v in env.items() if k.startswith('GGML_CUDA_') or k=='CUDA_VISIBLE_DEVICES'},'argv':args,'rows':rows}
 finally:
  if p.poll() is None:
   os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(30)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
  log.close()

if a.matrix=='sync': arms=[('pre','pre'),('post','post'),('post-r2','post'),('pre-r2','pre')]
elif a.matrix=='upstream': arms=[('upstream-a','upstream'),('post-a','post'),('post-b','post'),('upstream-b','upstream')]
elif a.matrix=='qwen-mmq': arms=[('post-normal','post'),('post-moe','post'),('post-force','force')]
elif a.matrix=='qwen-moe-abba': arms=[('normal-a','post'),('moe-a','post'),('moe-b','post'),('normal-b','post')]
else: arms=[('post-moe','post'),('post-force','force')]
allrows=[]
for i,(label,key) in enumerate(arms):
 allrows.append(run(label,key,a.port+i))
summary={}
for x in allrows:
 summary[x['label']]={k:statistics.mean(r[k] for r in x['rows']) for k in ['pp','prompt_ms','tg','predicted_ms','ttft_wall_ms','e2e_wall_ms']}
out={'case':a.case,'matrix':a.matrix,'prefix':prefix,'append':append,'npred':npred,'reps_per_arm':a.reps,'arms':allrows,'summary':summary}
outp=ROOT/f'{a.case}-{a.matrix}.json';outp.write_text(json.dumps(out,indent=2)+'\n');print('SUMMARY',json.dumps(summary,indent=2),flush=True)
