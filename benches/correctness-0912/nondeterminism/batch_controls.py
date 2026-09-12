#!/usr/bin/env python3
"""Separate concurrent batching variation from fixed-shape repeatability."""
import concurrent.futures as cf, hashlib,json,sys,threading,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import validate_streamk_e2e as v
D=Path('/models/.bench-ornith-mtp4/nondeterminism-0912');v.ROOT=D/'batch-controls';v.ROOT.mkdir(exist_ok=True)
B=D/'ssm-snapshot-fixed-normal/bin/llama-server'
results=[]
for ci,(name,serial,env) in enumerate([
 ('off-serial',True,{}),
 ('off-concurrent-int8off',False,{'GGML_CUDA_TURING_INT8_QK':'0'})]):
 s=v.Server(B,'batch-'+name,41800+ci,v.FIXTURE_DIR)
 s.env.update(env);s.manifest['env'].update(env);rows=[]
 with s:
  for rep in range(4):
   for i in range(4):s.request(f'/slots/{i}?action=erase',{})
   for i in range(4):s.restore(i,f'real100k-{i}.bin',100000)
   barrier=threading.Barrier(4)
   def one(i):
    if not serial:barrier.wait(30)
    return s.generate(v.FIXTURE['prompts'][i],slot=i,count=64,expected_cache=100000)
   if serial:out=[one(i) for i in range(4)]
   else:
    with cf.ThreadPoolExecutor(max_workers=4) as pool:out=list(pool.map(one,range(4)))
   rows.append({'rep':rep,'outputs':out});print(name,rep,[x['sha256'][:12] for x in out],flush=True)
 r={'name':name,'rows':rows,'unique_hashes_per_slot':[len({r['outputs'][i]['sha256'] for r in rows}) for i in range(4)],'manifest':s.manifest}
 results.append(r);(v.ROOT/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
 print('SUMMARY',name,r['unique_hashes_per_slot'],flush=True)
