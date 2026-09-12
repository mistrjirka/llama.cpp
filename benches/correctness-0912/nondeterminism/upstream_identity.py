#!/usr/bin/env python3
"""Compare full token outputs against actual upstream, not the pre-fix fork."""
import json
from pathlib import Path
import probe
D=Path('/models/.bench-ornith-mtp4/nondeterminism-0912')
new=D/'ssm-snapshot-fixed-normal/bin/llama-server'
up=Path('/workspace/llama-upstream-head-0911/build-sm70-75/bin/llama-server')
results=[]
for i,profile in enumerate(['qwen-rtx','qwen-v100']):
 fixed=probe.run('base','cached',41700+2*i,2,profile,new,64)
 reference=probe.run('upstream','cached',41701+2*i,2,profile,up,64)
 a=[r['response']['tokens'] for r in fixed['rows']];b=[r['response']['tokens'] for r in reference['rows']]
 r={'profile':profile,'fixed_unique':fixed['unique_hashes'],'upstream_unique':reference['unique_hashes'],
    'identical_to_upstream':a==b,'first_difference':next((j for j,(x,y) in enumerate(zip(a[0],b[0])) if x!=y),None),
    'fixed':fixed,'upstream':reference}
 results.append(r);print('IDENTITY',profile,r['identical_to_upstream'],r['first_difference'],flush=True)
 (D/'qwen-upstream-identity.json').write_text(json.dumps(results,indent=2)+'\n')
