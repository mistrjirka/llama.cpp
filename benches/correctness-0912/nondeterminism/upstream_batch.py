#!/usr/bin/env python3
"""Observe native upstream four-slot output repeatability (not a performance comparison)."""
import concurrent.futures as cf,json,sys,threading
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import validate_streamk_e2e as v
D=Path('/models/.bench-ornith-mtp4/nondeterminism-0912');v.ROOT=D/'upstream-batch';v.ROOT.mkdir(exist_ok=True)
B=Path('/workspace/llama-upstream-head-0911/build-sm70-75/bin/llama-server')
s=v.Server(B,'upstream-four-slot',41810,v.FIXTURE_DIR)
for flag,values in [('--slot-fork-prefix',0),('--kv-unified-per-slot',1),('--pipeline-copies',1)]:
 i=s.args.index(flag);del s.args[i:i+1+values]
s.manifest['argv']=s.args
rows=[]
with s:
 for rep in range(4):
  for i in range(4):s.request(f'/slots/{i}?action=erase',{})
  for i in range(4):s.restore(i,f'real100k-{i}.bin',100000)
  barrier=threading.Barrier(4)
  def one(i):
   barrier.wait(30)
   return s.generate(v.FIXTURE['prompts'][i],slot=i,count=64,expected_cache=100000)
  with cf.ThreadPoolExecutor(max_workers=4) as pool:out=list(pool.map(one,range(4)))
  rows.append({'rep':rep,'outputs':out});print('upstream',rep,[x['sha256'][:12] for x in out],flush=True)
r={'passed':True,'rows':rows,'unique_hashes_per_slot':[len({r['outputs'][i]['sha256'] for r in rows}) for i in range(4)],'manifest':s.manifest,
   'scope':'Native upstream has no prefix dedup/per-slot limit/pipeline-copies extensions; all 4 saved 100k target states fit in the same 1.4M physical pool.'}
(v.ROOT/'result.json').write_text(json.dumps(r,indent=2)+'\n')
print('SUMMARY',r['unique_hashes_per_slot'],flush=True)
