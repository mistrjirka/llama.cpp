#!/usr/bin/env python3
import os
"""Sequential serving removes arrival-order changes; compare target, draft, carry,
accepted-token counts and an on-disk checkpoint continuation across fresh processes.
"""
import importlib.util,json,pathlib,hashlib
R=pathlib.Path(os.environ.get('MTP_REFRESH_RESULTS_DIR', '/workspace/oai-qwen38-pp-lab/results/mtp-kv-refresh-0907'));F=R.parent/'parallel-refined-0907'
z=importlib.util.spec_from_file_location('fixture',R.parent/'parallel-research-0907/sweep.py');s=importlib.util.module_from_spec(z);z.loader.exec_module(s)
s.PORT=32579;s.OUT=R
for k in list(s.ENV):
 if k.startswith('LLAMA_EXPERIMENT_') or k.startswith('GGML_CUDA_VOLTA_Q8_MULTI') or k.startswith('GGML_CUDA_VOLTA_Q8_REF') or k=='LD_PRELOAD':s.ENV.pop(k)
f=json.loads((F/'real-fixture.json').read_text());rows=[]
original_args=s.args
def zero_args(_):
 cmd=original_args(3);cmd[cmd.index('--spec-draft-n-max')+1]='0';return cmd
s.args=zero_args
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as inp:
  for chunk in iter(lambda:inp.read(4*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
for label,binary,enabled in [('baseline','candidate-bin',True),('final','final-bin',True)]:
 B=R/binary;s.BIN=str(B/'llama-server');s.ENV['LD_LIBRARY_PATH']=str(B)
 s.ENV.pop('LLAMA_EXPERIMENT_MTP_KV_ONLY',None)
 if enabled:s.ENV['LLAMA_EXPERIMENT_MTP_KV_ONLY']='1'
 p,log=s.start(3,'zero-equiv-'+label)
 try:
  for i in range(4):s.req(f'/slots/{i}?action=restore',{'filename':f'real100k-{i}.bin'})
  out=[s.comp(i,f['prompts'][i],64) for i in range(4)]
  # Flush pending target/draft processing through the same public snapshot operation.
  s.req('/slots/0?action=save',{'filename':'equiv-slot0.bin'})
  hashes={ext:digest(R/('equiv-slot0.bin'+ext)) for ext in ['','.draft','.spec']}
  s.req('/slots/0?action=erase',{})
  s.req('/slots/0?action=restore',{'filename':'equiv-slot0.bin'})
  cont=s.comp(0,f['prompts'][0]+out[0]['tokens'],32)
  row={'arm':label,'requests':out,'snapshot_sha256':hashes,'restored_continuation':cont}
  rows.append(row);(R/'zero-equivalence-raw.json').write_text(json.dumps(rows,indent=2))
  print(label,'outputs',[x['sha'][:10] for x in out],'snapshot',hashes,'continued',cont['sha'][:10],flush=True)
 finally:s.stop(p,log)
ref=rows[0];summary={}
for row in rows[1:]:
 summary[row['arm']]={
 'output_tokens_identical':all(x['tokens']==y['tokens'] for x,y in zip(ref['requests'],row['requests'])),
 'acceptance_counters_identical':all((x['timings'].get('draft_n'),x['timings'].get('draft_n_accepted'))==(y['timings'].get('draft_n'),y['timings'].get('draft_n_accepted')) for x,y in zip(ref['requests'],row['requests'])),
 'snapshot_components_identical':{ext:ref['snapshot_sha256'][ext]==row['snapshot_sha256'][ext] for ext in ref['snapshot_sha256']},
 'restored_continuation_identical':ref['restored_continuation']['tokens']==row['restored_continuation']['tokens']}
(R/'zero-equivalence-summary.json').write_text(json.dumps(summary,indent=2));print('SUMMARY',json.dumps(summary),flush=True)
assert all(v if isinstance(v,bool) else all(v.values()) for row in summary.values() for v in row.values()),summary
