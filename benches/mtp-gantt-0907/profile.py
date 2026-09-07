#!/usr/bin/env python3
"""Separate trace capture from timing comparisons; only immutable binary paths."""
import argparse,concurrent.futures as cf,ctypes,importlib.util,json,pathlib,threading,time
R=pathlib.Path('/workspace/oai-qwen38-pp-lab/results/mtp-gantt-0907');F=R.parent/'parallel-refined-0907'
p=argparse.ArgumentParser();p.add_argument('--binary',default='baseline-bin');p.add_argument('--label',default='baseline');p.add_argument('--modes',default='mtp3x4,offx4,mtp3x1');p.add_argument('--cpu-fixes',action='store_true');a=p.parse_args()
B=R/a.binary
z=importlib.util.spec_from_file_location('fixture',R.parent/'parallel-research-0907/sweep.py');s=importlib.util.module_from_spec(z);z.loader.exec_module(s)
s.PORT=32651;s.OUT=R;s.BIN=str(B/'llama-server');s.ENV['LD_LIBRARY_PATH']=str(B)
for k in list(s.ENV):
 if k.startswith('LLAMA_EXPERIMENT_') or k.startswith('LLAMA_MTP_') or k.startswith('GGML_CUDA_VOLTA_Q8_MULTI') or k.startswith('GGML_CUDA_VOLTA_Q8_REF') or k in ['GGML_CUDA_VOLTA_Q8_COOP','GGML_CUDA_VOLTA_Q8_PV_F32','LD_PRELOAD']:s.ENV.pop(k)
s.ENV['LD_PRELOAD']=str(R/'markers.so')
if a.cpu_fixes:s.ENV.update({'LLAMA_EXPERIMENT_KV_INDEXED_RM':'1','LLAMA_EXPERIMENT_MTP_SKIP_PROMPT_COPY':'1'})
for f in F.glob('real100k-*.bin*'):
 dest=R/f.name
 if not dest.exists():dest.symlink_to(f)
f=json.loads((F/'real-fixture.json').read_text());mark=ctypes.CDLL(str(R.parent/'nsight-parallel-0907/markers.so'));mark.mark_push.argtypes=[ctypes.c_char_p];rows=[]
def batch(q,n,c):
 b=threading.Barrier(c)
 with cf.ThreadPoolExecutor(max_workers=c) as ex:return list(ex.map(lambda i:s.comp(i,q[i],n,b),range(c)))
for mode in a.modes.split(','):
 depth=0 if mode.startswith('off') else 3;c=int(mode[-1]);p,log=s.start(depth,a.label+'-'+mode)
 try:
  assert str(B/'libllama') in pathlib.Path(f'/proc/{p.pid}/maps').read_text()
  for i in range(4):assert s.req(f'/slots/{i}?action=restore',{'filename':f'real100k-{i}.bin'})['n_restored']==100000
  warm=batch(f['prompts'],24,c);q=[p+x['tokens'] for p,x in zip(f['prompts'],warm)]
  mark.mark_push((a.label+'/'+mode+'/TG').encode())
  try:
   t=time.perf_counter();out=batch(q,128,c);wall=time.perf_counter()-t
  finally:mark.mark_pop()
  assert all(r['timings']['predicted_n']==128 for r in out)
  rows.append({'mode':mode,'wall_s_profiled':wall,'requests':out,'argv':s.args(depth)})
  (R/(a.label+'-profile-raw.json')).write_text(json.dumps(rows,indent=2))
  print(a.label,mode,wall,flush=True)
 finally:s.stop(p,log)
