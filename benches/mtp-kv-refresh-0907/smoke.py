#!/usr/bin/env python3
import os
import importlib.util,json,pathlib,concurrent.futures as cf,threading
R=pathlib.Path(os.environ.get('MTP_REFRESH_RESULTS_DIR', '/workspace/oai-qwen38-pp-lab/results/mtp-kv-refresh-0907'));F=R.parent/'parallel-refined-0907'
z=importlib.util.spec_from_file_location('fixture',R.parent/'parallel-research-0907/sweep.py');s=importlib.util.module_from_spec(z);z.loader.exec_module(s)
s.PORT=32570;s.OUT=R;s.BIN='/workspace/llama-parallel-all-0907/build-sm70-75/bin/llama-server';s.ENV['LD_LIBRARY_PATH']=str(pathlib.Path(s.BIN).parent)
for k in list(s.ENV):
 if k.startswith('LLAMA_EXPERIMENT_') or k.startswith('GGML_CUDA_VOLTA_Q8_MULTI') or k.startswith('GGML_CUDA_VOLTA_Q8_REF') or k=='LD_PRELOAD':s.ENV.pop(k)
s.ENV['KV_CHECK_DIR']=str(R);s.ENV['LLAMA_EXPERIMENT_MTP_KV_ONLY']='1';s.ENV['LD_PRELOAD']=str(R/'validate-kv.so')
f=json.loads((F/'real-fixture.json').read_text())
for i in range(4):
 for ext in ['','.draft','.spec']:
  p=R/f'real100k-{i}.bin{ext}'
  if not p.exists():p.symlink_to(F/p.name)
p,log=s.start(3,'smoke')
try:
 for i in range(4):assert s.req(f'/slots/{i}?action=restore',{'filename':f'real100k-{i}.bin'})['n_restored']==100000
 b=threading.Barrier(4)
 with cf.ThreadPoolExecutor(max_workers=4) as ex:rr=list(ex.map(lambda i:s.comp(i,f['prompts'][i],32,b),range(4)))
 for x in rr:assert x['timings']['predicted_n']==32,x
 for i in range(4):s.req(f'/slots/{i}?action=save',{'filename':f'smoke-{i}.bin'})
 for i in range(4):s.req(f'/slots/{i}?action=erase',{})
 for i in range(4):s.req(f'/slots/{i}?action=restore',{'filename':f'smoke-{i}.bin'})
 b=threading.Barrier(4)
 with cf.ThreadPoolExecutor(max_workers=4) as ex:rt=list(ex.map(lambda i:s.comp(i,f['prompts'][i]+rr[i]['tokens'],16,b),range(4)))
 (R/'smoke.json').write_text(json.dumps({'initial':rr,'restored':rt},indent=2));print('server continuation and snapshot restore passed',flush=True)
finally:s.stop(p,log)
