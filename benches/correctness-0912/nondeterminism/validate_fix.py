#!/usr/bin/env python3
"""Run model-level determinism, cached throughput and production-state checks."""
import copy,json,statistics,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import validate_streamk_e2e as v
import benchmark_streamk_regression as b
D=Path('/models/.bench-ornith-mtp4/nondeterminism-0912');OUT=D/'validation';OUT.mkdir(exist_ok=True)
v.ROOT=b.ROOT=OUT
FIX_F=D/'ssm-snapshot-fixed/bin/llama-server';FIX_N=D/'ssm-snapshot-fixed-normal/bin/llama-server'
BASE=Path('/models/.bench-ornith-mtp4/int8-streamk-fixed-src')

def save(name,x):
 (OUT/(name+'.json')).write_text(json.dumps(x,indent=2)+'\n')
 print('RESULT',name, json.dumps({k:z for k,z in x.items() if k not in ('rows','arms','manifest')}),flush=True)

mode=sys.argv[1] if len(sys.argv)>1 else 'perf'
if mode=='perf':
 b.PROFILES['ornith-v100']=copy.deepcopy(b.PROFILES['ornith-dual'])
 b.PROFILES['ornith-v100'].update(cuda=v.V100,placement=['--split-mode','none'])
 for pi,name in enumerate(['ornith-rtx','ornith-dual','ornith-v100','qwen-rtx','qwen-v100']):
  force=name in ('ornith-rtx','ornith-dual');old=BASE/('build-matched-force' if force else 'build-matched-normal')/'bin/llama-server';new=FIX_F if force else FIX_N
  rows=[]
  for i,(key,binary) in enumerate([('before',old),('fixed',new),('fixed',new),('before',old)]):
   z=b.arm(binary,f'ssm-{name}-{key}-{i}',41500+pi*10+i,b.PROFILES[name],3);z['key']=key;rows.append(z)
   save(f'perf-{name}-progress',{'arms':rows})
  summary={}
  for key in ('before','fixed'):
   rr=[r for r in rows if r['key']==key];pps=[x['timings']['prompt_per_second'] for r in rr for x in r['rows'][1:]]
   summary[key]={'pp':statistics.mean(pps),'pp_sd':statistics.stdev(pps),'tg':statistics.mean(r['tg'] for r in rr),'decode_hashes':sorted({r['decode']['sha256'] for r in rr})}
  summary.update(pp_delta_pct=100*(summary['fixed']['pp']/summary['before']['pp']-1),tg_delta_pct=100*(summary['fixed']['tg']/summary['before']['tg']-1))
  save(f'perf-{name}',{'profile':name,'summary':summary,'arms':rows})
elif mode=='stress':
 for i,(name,kv,share) in enumerate([('q4-head','q4_0',True),('off',None,False),('q8-plain','q8_0',False)]):
  r=v.stress(FIX_N,'ssm-stress-'+name,41600+i,kv,10,share)
  save('stress-'+name,{'passed':True,**r})
elif mode=='boundary-cold':
 for i,(label,fn) in enumerate([
 ('boundary',lambda:v.boundary(FIX_N,'ssm-boundary',41610,False)),
 ('cold100k-off',lambda:v.cold(FIX_N,'ssm-cold100k-off',41611,None)),
 ('cold100k-q4',lambda:v.cold(FIX_N,'ssm-cold100k-q4',41612,'q4_0'))]):
  save(label,{'passed':True,**fn()})
else:raise ValueError(mode)
