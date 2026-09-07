#!/usr/bin/env python3
"""Sequential fixed histories: assert output, counters and raw snapshot parity."""
import hashlib,importlib.util,json,pathlib
R=pathlib.Path('/workspace/oai-qwen38-pp-lab/results/mtp-gantt-0907');F=R.parent/'parallel-refined-0907'
z=importlib.util.spec_from_file_location('fixture',R.parent/'parallel-research-0907/sweep.py');s=importlib.util.module_from_spec(z);z.loader.exec_module(s)
s.PORT=32653;s.OUT=R
for k in list(s.ENV):
 if k.startswith('LLAMA_EXPERIMENT_') or k.startswith('LLAMA_MTP_') or k.startswith('GGML_CUDA_VOLTA_Q8_MULTI') or k.startswith('GGML_CUDA_VOLTA_Q8_REF') or k in ['GGML_CUDA_VOLTA_Q8_COOP','GGML_CUDA_VOLTA_Q8_PV_F32','LD_PRELOAD']:s.ENV.pop(k)
f=json.loads((F/'real-fixture.json').read_text());rows=[]
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as inp:
  for b in iter(lambda:inp.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
for label,bin,enabled in [('baseline','baseline-bin',False),('patched','candidate-bin',True),('disabled','candidate-bin',False)]:
 B=R/bin;s.BIN=str(B/'llama-server');s.ENV['LD_LIBRARY_PATH']=str(B)
 for k in ['LLAMA_EXPERIMENT_KV_INDEXED_RM','LLAMA_EXPERIMENT_MTP_SKIP_PROMPT_COPY']:s.ENV.pop(k,None)
 if enabled:s.ENV.update({'LLAMA_EXPERIMENT_KV_INDEXED_RM':'1','LLAMA_EXPERIMENT_MTP_SKIP_PROMPT_COPY':'1'})
 p,log=s.start(3,'equiv-'+label)
 try:
  for i in range(4):s.req(f'/slots/{i}?action=restore',{'filename':f'real100k-{i}.bin'})
  out=[s.comp(i,f['prompts'][i],64) for i in range(4)]
  s.req('/slots/0?action=save',{'filename':'gantt-parity.bin'})
  hashes={ext:digest(R/('gantt-parity.bin'+ext)) for ext in ['','.draft','.spec']}
  s.req('/slots/0?action=erase',{});s.req('/slots/0?action=restore',{'filename':'gantt-parity.bin'})
  cont=s.comp(0,f['prompts'][0]+out[0]['tokens'],32)
  rows.append({'arm':label,'requests':out,'hashes':hashes,'continuation':cont});(R/'equiv-raw.json').write_text(json.dumps(rows,indent=2))
  print(label,[x['sha'][:10] for x in out],hashes,flush=True)
 finally:s.stop(p,log)
ref=rows[0];summary={}
for r in rows[1:]:
 summary[r['arm']]={'outputs_equal':all(x['tokens']==y['tokens'] for x,y in zip(ref['requests'],r['requests'])),'acceptance_equal':all((x['timings'].get('draft_n'),x['timings'].get('draft_n_accepted'))==(y['timings'].get('draft_n'),y['timings'].get('draft_n_accepted')) for x,y in zip(ref['requests'],r['requests'])),'raw_target_draft_spec_hashes_equal':ref['hashes']==r['hashes'],'restored_continuation_equal':ref['continuation']['tokens']==r['continuation']['tokens']}
(R/'equiv-summary.json').write_text(json.dumps(summary,indent=2));print('SUMMARY',json.dumps(summary),flush=True)
assert all(all(r.values()) for r in summary.values()),summary
