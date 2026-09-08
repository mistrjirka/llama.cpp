#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
from analyze_traces import load_tensor, compare
D=Path('/models/llama-pxq4-build/divergence-0908')
V=Path(__file__).resolve().parent
for left,right in [('pxa','port'),('fallback','port')]:
 result=compare(str(D/f'trace-{left}.trace.json'),str(D/f'trace-{right}.trace.json'))
 (V/f'trace-{left}-vs-{right}.json').write_text(json.dumps(result,indent=2))
 print(left,right,'first differences:')
 for r in [r for r in result['rows'] if not r['equal']][:12]:
  print(r['name'], r['op'], 'relative_rms',r['rms_relative'],'max_abs',r['max_abs'])
 print('Router IDs:',[(r['name'],r['different'],r['total']) for r in result['rows'] if 'topk' in r['name'] or 'ids' in r['name']])
for tag in ['pxa','port']:
 rows=json.loads((D/f'trace-{tag}.trace.json').read_text())
 print(tag,'layer0 names:')
 for r in rows:
  if r['name'].endswith('-0') and any(s in r['name'] for s in ['ffn','norm']):print(r['name'],r['op'],r['ne'],r['sources'])
a=json.loads((D/'trace-pxa.trace.json').read_text());b=json.loads((D/'trace-port.trace.json').read_text())
x=load_tensor(next(r for r in a if r['name']=='new_state-0'));y=load_tensor(next(r for r in b if r['name']=='new_state-0'))
print('STATE transpose relative RMS:',np.linalg.norm(x-y.swapaxes(-1,-2))/np.linalg.norm(x))
