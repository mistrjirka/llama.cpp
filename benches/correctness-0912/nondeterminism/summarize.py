#!/usr/bin/env python3
"""Copy completed evidence and produce a compact, auditable validation summary."""
import hashlib,json,shutil
from pathlib import Path
D=Path('/models/.bench-ornith-mtp4/nondeterminism-0912')
HERE=Path(__file__).resolve().parent;OUT=HERE/'results';OUT.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text())
summary={'base_commit':'a70ee26ae','fix_commit':'580168936','production_change':'SSM_CONV reads its materialized input rather than CONCAT ancestors',
 'performance':{},'state_tests':{},'identity_probes':{},'evidence':{}}
for name in ['snapshot-before.log','snapshot-fixed.log','snapshot-memcheck.log','main-ctest.log']:
 p=D/name
 if not p.exists():continue
 shutil.copy2(p,OUT/name);summary['evidence'][name]={'path':str(p),'sha256':sha(p)}
for p in D.glob('main-flash-*.log'):
 if '10/10 tests passed' not in p.read_text():continue
 shutil.copy2(p,OUT/p.name);summary['evidence'][p.name]={'path':str(p),'sha256':sha(p)}
for name in ['ornith-rtx','ornith-dual','ornith-v100','qwen-rtx','qwen-v100']:
 p=D/'validation'/f'perf-{name}.json'
 if not p.exists():continue
 d=load(p);shutil.copy2(p,OUT/p.name);summary['performance'][name]=d['summary']
 summary['evidence'][p.name]={'path':str(p),'sha256':sha(p)}
for name in ['boundary','cold100k-off','cold100k-q4','stress-q4-head','stress-off','stress-q8-plain']:
 p=D/'validation'/(name+'.json')
 if not p.exists():continue
 d=load(p);compact={k:v for k,v in d.items() if k!='rows'}
 compact['rows']=[]
 for row in d.get('rows',[]):
  r={k:v for k,v in row.items() if k not in ('outputs','tokens')}
  if 'outputs' in row:r['outputs']=[{k:v for k,v in out.items() if k!='tokens'} for out in row['outputs']]
  compact['rows'].append(r)
 compact['full_raw_evidence']={'path':str(p),'sha256':sha(p)}
 (OUT/(name+'.json')).write_text(json.dumps(compact,indent=2)+'\n')
 summary['state_tests'][name]={k:v for k,v in compact.items() if k not in ('rows','manifest')}
 summary['state_tests'][name]['rounds']=len(d.get('rows',[]))
for p in D.glob('det-*.json'):
 if '.manifest.' in p.name:continue
 d=load(p)
 r={'unique_hashes':d['unique_hashes'],'manifest':d['manifest'],'rows':[],'raw_evidence':{'path':str(p),'sha256':sha(p)}}
 for row in d['rows']:
  z=row['response'];r['rows'].append({'rep':row['rep'],'sha256':row['hash'],'first_diff':row['first_diff'],
   'tokens':z['tokens'],'timings':z['timings'],'temperature':z['generation_settings']['temperature']})
 (OUT/p.name).write_text(json.dumps(r,indent=2)+'\n')
 summary['identity_probes'][p.name]={'unique_hashes':r['unique_hashes'],'runs':len(r['rows'])}
for name in ['ssm-snapshot-fixed','ssm-snapshot-fixed-normal']:
 p=D/name/'manifest.json'
 if p.exists():shutil.copy2(p,OUT/(name+'-manifest.json'))
p=D/'qwen-upstream-identity.json'
if p.exists():
 d=load(p);summary['qwen_upstream_identity']=[{k:v for k,v in row.items() if k not in ('fixed','upstream')} for row in d]
for name,source in [('batch-controls',D/'batch-controls/summary.json'),('upstream-batch',D/'upstream-batch/result.json')]:
 if not source.exists():continue
 raw=load(source);cases=raw if isinstance(raw,list) else [raw];compact=[]
 for case in cases:
  c={k:v for k,v in case.items() if k!='rows'}
  c['rows']=[{'rep':r['rep'],'outputs':[{k:v for k,v in out.items() if k!='tokens'} for out in r['outputs']]} for r in case['rows']]
  c['full_raw_evidence']={'path':str(source),'sha256':sha(source)};compact.append(c)
 (OUT/(name+'.json')).write_text(json.dumps(compact,indent=2)+'\n')
 summary[name]=[{k:v for k,v in c.items() if k not in ('rows','manifest','full_raw_evidence')} for c in compact]
summary['not_covered']=['Remaining RTX Qwen final-token discrepancy versus upstream (unchanged with INT8 QK disabled)','Dense Qwen MTP target-head-reuse validation','Reducing the draft compute arena','Universal batch-size-independent determinism']
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ('evidence','identity_probes')},indent=2))
