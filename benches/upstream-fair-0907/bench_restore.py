#!/usr/bin/env python3
import concurrent.futures as cf,hashlib,json,os,pathlib,signal,statistics,subprocess,threading,time,urllib.request,urllib.error
R=pathlib.Path(__file__).resolve().parent;F=R.parent/'parallel-refined-0907';fx=json.loads((F/'real-fixture.json').read_text())
H=fx['histories'];SUF=[fx['prompts'][i][100000:] for i in range(4)];assert [len(x) for x in SUF]==[38,42,44,41]
MODEL='/models/Ornith-1.5-35B-A3B/Ornith-1.5-35B-A3B-AD-Q6_K-Q5_K.gguf';DRAFT='/workspace/models/Ornith-1.5-35B-A3B/shisa-mtp/mtp-shisa-ornith15-all-Q5_0.gguf';MMP='/models/local-llm-setup/ornith15/mmproj-Ornith-1.5-35B-BF16.gguf';PORT=32721
BASE_ENV=os.environ.copy();BASE_ENV.update({'GGML_CUDA_ALLREDUCE':'internal','GGML_CUDA_AR_COPY_THRESHOLD':'131072','GGML_CUDA_TURING_CUBLAS_MIN_BATCH':'256','GGML_CUDA_VOLTA_Q8_FATTN_TC':'1','GGML_CUDA_VOLTA_Q5_X4':'1','GGML_CUDA_VOLTA_Q6_W4R4':'1','GGML_CUDA_VOLTA_FORCE_MMQ':'moe','GGML_CUDA_VOLTA_GQA8_NCOLS2':'2'})
for k in list(BASE_ENV):
 if k.startswith('LLAMA_EXPERIMENT_') or k.startswith('LLAMA_MTP_') or k in ['LLAMA_KV_INDEXED_RM','LLAMA_SAMPLING_VIEW','LD_PRELOAD']:BASE_ENV.pop(k,None)
def req(path,body=None,timeout=900):
 data=None if body is None else json.dumps(body,separators=(',',':')).encode();q=urllib.request.Request(f'http://127.0.0.1:{PORT}{path}',data=data,headers={} if data is None else {'Content-Type':'application/json'})
 try:
  with urllib.request.urlopen(q,timeout=timeout) as f:return json.load(f)
 except urllib.error.HTTPError as e:raise RuntimeError(f'HTTP {e.code}: {e.read().decode()[:700]}') from e
def args(binary,is_fork):
 a=[str(binary),'-m',MODEL,'--mmproj',MMP,'--no-mmproj-offload','--host','127.0.0.1','--port',str(PORT),'--ctx-size','1400000','--parallel','4','--kv-unified','--kv-unified-per-slot','400000','--cache-ram','0','--no-cache-idle-slots','--slot-prompt-similarity','0','--override-kv','qwen35moe.context_length=int:400000','--rope-scaling','yarn','--rope-scale','1.52587890625','--yarn-orig-ctx','262144','--split-mode','layer','--fit','off','--gpu-layers','all','--device','CUDA1,CUDA0','--tensor-split','14,35','--flash-attn','on','--batch-size','2048','--ubatch-size','256','--cache-type-k','q8_0','--cache-type-v','q8_0','--slot-save-path',str(R),'--no-warmup','--perf','--slots','--spec-type','draft-mtp','--spec-draft-model',DRAFT,'--spec-draft-device','CUDA1','--spec-draft-ngl','all','--spec-draft-type-k','q8_0','--spec-draft-type-v','q8_0','--spec-draft-n-max','3']
 if is_fork:a += ['--no-slot-fork-prefix','--no-spec-mtp-defer-prompt','--spec-draft-ubatch','0','--pipeline-copies','0']
 return a
def start(binary,is_fork,label):
 env=BASE_ENV.copy();log=open(R/(label+'.log'),'wb');p=subprocess.Popen(args(binary,is_fork),env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 for _ in range(700):
  if p.poll() is not None:log.close();raise RuntimeError(f'{label} died {p.returncode}: '+(R/(label+'.log')).read_text(errors='ignore')[-1800:])
  try:
   if req('/health',timeout=.5).get('status')=='ok':return p,log
  except:pass
  time.sleep(.1)
 raise TimeoutError(label)
def stop(p,log):
 if p.poll() is None:
  os.killpg(p.pid,signal.SIGTERM)
  try:p.wait(25)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
 log.close()
def restore_all():
 rows=[]
 for i in range(4):
  z=req(f'/slots/{i}?action=restore',{'filename':f'real100k-{i}.bin'});assert z['n_restored']==100000,z;rows.append(z)
 return rows
def timed():
 barrier=threading.Barrier(4)
 def one(i):
  barrier.wait();t=time.perf_counter();z=req('/completion',{'prompt':H[i]+SUF[i],'id_slot':i,'cache_prompt':True,'n_predict':128,'ignore_eos':True,'temperature':0.0,'seed':1234,'return_tokens':True});wall=time.perf_counter()-t
  tm=z['timings'];assert tm['cache_n']==100000 and tm['prompt_n']==len(SUF[i]) and tm['predicted_n']==128,(i,tm)
  toks=z['tokens'];return {'slot':i,'wall_s':wall,'timings':tm,'sha':hashlib.sha256(','.join(map(str,toks)).encode()).hexdigest()}
 t=time.perf_counter()
 with cf.ThreadPoolExecutor(max_workers=4) as ex:out=list(ex.map(one,range(4)))
 wall=time.perf_counter()-t
 return {'wall_s':wall,'aggregate_generated_tps_whole_turn':512/wall,'aggregate_pp_plus_tg_tps':677/wall,'mean_agent_tg':statistics.mean(x['timings']['predicted_per_second'] for x in out),'mean_agent_pp':statistics.mean(x['timings']['prompt_per_second'] for x in out),'acceptance':[(x['timings'].get('draft_n_accepted',0),x['timings'].get('draft_n',0)) for x in out],'sha':[x['sha'] for x in out],'requests':out}
def run(label,binary,is_fork):
 p,log=start(binary,is_fork,label)
 try:
  startup_restore=restore_all();rows=[]
  for rep in range(7):
   restore_all();z=timed();z['rep']=rep;z['warmup']=rep==0;rows.append(z);print(label,rep,round(z['wall_s'],4),round(z['aggregate_generated_tps_whole_turn'],2),round(z['aggregate_pp_plus_tg_tps'],2),round(z['mean_agent_tg'],2),z['acceptance'],flush=True)
  keep=rows[1:];s={'label':label,'binary':str(binary),'is_fork':is_fork,'n':len(keep),'startup_restore':startup_restore,'rows':rows,'argv':args(binary,is_fork)}
  for k in ['wall_s','aggregate_generated_tps_whole_turn','aggregate_pp_plus_tg_tps','mean_agent_tg','mean_agent_pp']:
   s[k]={'mean':statistics.mean(x[k] for x in keep),'min':min(x[k] for x in keep),'max':max(x[k] for x in keep)}
  (R/(label+'-restore-summary.json')).write_text(json.dumps(s,indent=2));return s
 finally:stop(p,log)
if __name__=='__main__':
 arms=[('samebase-upstream',pathlib.Path('/workspace/llama-upstream-vanilla-0906/build-vanilla/bin/llama-server'),False),('fork-current',pathlib.Path('/workspace/oai-qwen38-pp-lab/results/mtp-final-integration-0907/candidate-bin/llama-server'),True),('latest-upstream',pathlib.Path('/workspace/upstream-fair-0907/build/bin/llama-server'),False),('fork-current-r2',pathlib.Path('/workspace/oai-qwen38-pp-lab/results/mtp-final-integration-0907/candidate-bin/llama-server'),True),('samebase-upstream-r2',pathlib.Path('/workspace/llama-upstream-vanilla-0906/build-vanilla/bin/llama-server'),False)]
 all=[]
 for arm in arms:
  x=run(*arm);all.append(x);(R/'restore-all-summary.json').write_text(json.dumps(all,indent=2))
