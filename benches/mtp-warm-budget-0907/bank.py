#!/usr/bin/env python3
"""Budget pause/resume, mixed-agent isolation and parked-bank process restart.
Correctness tests only; these timings are not performance claims.
"""
import os
import concurrent.futures as cf,hashlib,importlib.util,json,pathlib,threading,time
R=pathlib.Path(os.environ.get('MTP_BUDGET_RESULTS_DIR', '/workspace/oai-qwen38-pp-lab/results/mtp-warm-budget-0907')); F=R.parent/'parallel-refined-0907';B=R/'candidate-bin'
z=importlib.util.spec_from_file_location('fixture',R.parent/'parallel-research-0907/sweep.py');s=importlib.util.module_from_spec(z);z.loader.exec_module(s)
s.OUT=R;s.PORT=32591;s.BIN=str(B/'llama-server');s.ENV['LD_LIBRARY_PATH']=str(B)
for k in list(s.ENV):
 if k.startswith('LLAMA_EXPERIMENT_') or k.startswith('GGML_CUDA_VOLTA_Q8_MULTI') or k.startswith('GGML_CUDA_VOLTA_Q8_REF') or k in ['GGML_CUDA_VOLTA_Q8_COOP','GGML_CUDA_VOLTA_Q8_PV_F32','LD_PRELOAD','LLAMA_ARG_BACKEND_SAMPLING']:s.ENV.pop(k)
s.ENV['LLAMA_EXPERIMENT_MTP_REQUEST_BUDGET']='1'
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

report=json.loads((R/'validation-raw.json').read_text())
assert all(report['transition_checks'].values())
s.ENV['LLAMA_EXPERIMENT_MTP_KV_ONLY']='1'
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
