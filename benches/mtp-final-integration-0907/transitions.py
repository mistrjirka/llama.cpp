#!/usr/bin/env python3
"""Budget pause/resume, mixed-agent isolation and parked-bank process restart.
Correctness tests only; these timings are not performance claims.
"""
import concurrent.futures as cf,hashlib,importlib.util,json,pathlib,threading,time
R=pathlib.Path('/workspace/oai-qwen38-pp-lab/results/mtp-final-integration-0907'); F=R.parent/'parallel-refined-0907';B=R/'candidate-bin'
z=importlib.util.spec_from_file_location('fixture',R.parent/'parallel-research-0907/sweep.py');s=importlib.util.module_from_spec(z);z.loader.exec_module(s)
s.OUT=R;s.PORT=32704;s.BIN=str(B/'llama-server');s.ENV['LD_LIBRARY_PATH']=str(B)
for k in list(s.ENV):
 if k.startswith('LLAMA_EXPERIMENT_') or k.startswith('LLAMA_MTP_') or k=='LLAMA_KV_INDEXED_RM' or k.startswith('GGML_CUDA_VOLTA_Q8_MULTI') or k.startswith('GGML_CUDA_VOLTA_Q8_REF') or k in ['GGML_CUDA_VOLTA_Q8_COOP','GGML_CUDA_VOLTA_Q8_PV_F32','LD_PRELOAD','LLAMA_ARG_BACKEND_SAMPLING']:s.ENV.pop(k)
# Per-request ceilings are now supported by default.
f=json.loads((F/'real-fixture.json').read_text());base_args=s.args
for i in range(4):
 for ext in ['', '.draft','.spec']:
  link=R/f'real100k-{i}.bin{ext}'
  if not link.exists():link.symlink_to(F/link.name)
report={}
def sha(t):return hashlib.sha256(','.join(map(str,t)).encode()).hexdigest()
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as inp:
  for x in iter(lambda:inp.read(4*1024*1024),b''):h.update(x)
 return h.hexdigest()
def comp(q,n=32,cap=None,slot=None):
 body={'prompt':q,'n_predict':n,'return_tokens':True,'cache_prompt':True,'ignore_eos':True,'temperature':0.0,'seed':1234}
 if cap is not None:body['speculative_n_max']=cap
 if slot is not None:body['id_slot']=slot
 r=s.req('/completion',body)
 assert len(r.get('tokens',[]))==n, r
 if cap is not None:assert r['generation_settings']['speculative_n_max']==cap,r.get('generation_settings')
 if cap==0:assert r['timings'].get('draft_n',0)==0,r['timings']
 return r
# Ordinary vs K/V-only refresh with the identical 3 -> 0 -> 3 sequence of requests.
records=[]
for enabled in [False,True]:
 s.ENV['LLAMA_MTP_KV_ONLY']='1' if enabled else '0'
 p,l=s.start(3,'transition-'+str(int(enabled)))
 try:
  assert str(B/'libllama-server-impl.so') in pathlib.Path(f'/proc/{p.pid}/maps').read_text()
  for i in range(4):s.req(f'/slots/{i}?action=restore',{'filename':f'real100k-{i}.bin'})
  q=list(f['prompts'][0]);out=[]
  for cap in [3,0,3]:
   r=comp(q,48,cap,0);out.append(r)
   if len(out)==1:assert r['timings']['prompt_n']==len(f['prompts'][0])-100000,r['timings']
   else:assert r['timings']['prompt_n']<=2,r['timings']
   q+=r['tokens']
  assert out[0]['timings']['draft_n']>0 and out[2]['timings']['draft_n']>0
  s.req('/slots/0?action=save',{'filename':'transition.bin'})
  hashes={ext:digest(R/('transition.bin'+ext)) for ext in ['', '.draft','.spec']}
  s.req('/slots/0?action=erase',{});s.req('/slots/0?action=restore',{'filename':'transition.bin'})
  continuation=comp(q,32,3,0)
  records.append({'cache_only':enabled,'outputs':out,'snapshot':hashes,'continuation':continuation})
  if enabled:
   errors=[]
   for cap in [-1,4,'bad',True,1.5,2**63]:
    try:s.req('/completion',{'prompt':[100,101,102],'n_predict':0,'speculative_n_max':cap})
    except RuntimeError as e:errors.append({'value':cap,'error':str(e)});continue
    errors.append({'value':cap,'accepted':True})
   assert all('error' in x for x in errors),errors
   report['validation_errors']=errors
   for i in range(4):s.req(f'/slots/{i}?action=erase',{})
   for i in range(4):s.req(f'/slots/{i}?action=restore',{'filename':f'real100k-{i}.bin'})
   barrier=threading.Barrier(4);caps=[0,1,3,0]
   def work(i):barrier.wait();return comp(f['prompts'][i],48,caps[i],i)
   with cf.ThreadPoolExecutor(max_workers=4) as ex:mixed=list(ex.map(work,range(4)))
   assert all(x['timings']['cache_n']==100000 for x in mixed)
   assert all(mixed[i]['timings'].get('draft_n',0)==0 for i in [0,3])
   assert all(mixed[i]['timings'].get('draft_n',0)>0 for i in [1,2])
   # An omitted ceiling must recover the process default, not another request's zero.
   resumed=comp(f['prompts'][3]+mixed[3]['tokens'],32,None,3)
   assert resumed['generation_settings']['speculative_n_max']==3 and resumed['timings']['draft_n']>0
   report['mixed']={'caps':caps,'timings':[x['timings'] for x in mixed],'default_resume':resumed['timings']}
 finally:s.stop(p,l)
 report['transitions']=records;(R/'validation-raw.json').write_text(json.dumps(report,indent=2))
a,b=records
report['transition_checks']={'output_tokens_equal':all(x['tokens']==y['tokens'] for x,y in zip(a['outputs'],b['outputs'])),'snapshot_bytes_equal':a['snapshot']==b['snapshot'],'continuation_equal':a['continuation']['tokens']==b['continuation']['tokens'],'acceptance_equal':all((x['timings'].get('draft_n'),x['timings'].get('draft_n_accepted'))==(y['timings'].get('draft_n'),y['timings'].get('draft_n_accepted')) for x,y in zip(a['outputs'],b['outputs']))}
assert all(report['transition_checks'].values()),report['transition_checks']
(R/'validation-raw.json').write_text(json.dumps(report,indent=2))
print('TRANSITION',report['transition_checks'],flush=True)
# One physical slot serves five histories, four parked in RAM, then the process exits.
def small_args(n):
 a=base_args(n)
 for flag,value in [('--ctx-size','32768'),('--parallel','1'),('--kv-unified-per-slot','32768'),('--cache-ram','4096')]:a[a.index(flag)+1]=value
 return a
s.args=small_args;histories=[f['histories'][0][i*12000:i*12000+4096] for i in range(5)]
suffix=f['prompts'][0][100000:]
p,l=s.start(3,'bank-control')
try:
 comp(histories[0],1,0);reference=comp(histories[0]+suffix,64,3)
finally:s.stop(p,l)
p,l=s.start(3,'bank-before-restart')
try:
 for h in histories:comp(h,1,0)
 saved=s.req('/prompt-cache?action=save',{'filename':'parked-budget.bin'})
 assert saved['n_saved']>=4,saved
finally:s.stop(p,l)
p,l=s.start(3,'bank-after-restart')
try:
 loaded=s.req('/prompt-cache?action=restore',{'filename':'parked-budget.bin'})
 assert loaded['n_restored']==saved['n_saved'],loaded
 restored=comp(histories[0]+suffix,64,3)
 assert restored['timings']['cache_n']==4096,restored['timings']
 assert restored['timings']['prompt_n']==len(suffix),restored['timings']
 assert reference['tokens']==restored['tokens'],'bank continuation mismatch'
 assert (reference['timings']['draft_n'],reference['timings']['draft_n_accepted'])==(restored['timings']['draft_n'],restored['timings']['draft_n_accepted'])
 report['bank']={'saved':saved,'loaded':loaded,'reference_timing':reference['timings'],'restored_timing':restored['timings'],'sha_equal':sha(reference['tokens'])==sha(restored['tokens']),'sha':sha(restored['tokens'])}
finally:s.stop(p,l)
(R/'validation-raw.json').write_text(json.dumps(report,indent=2))
print('BANK',json.dumps(report['bank']),flush=True)
