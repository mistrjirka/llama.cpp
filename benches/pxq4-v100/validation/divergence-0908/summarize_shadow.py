#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
p=Path(__file__).resolve().parent
rows=json.loads((p/'shadow-results.json').read_text())['rows']
summary={}
for mode in ['restore-control','direct-f32','integer']:
 r=[x for x in rows if x['mode']==mode]
 out=dict(positions=len(r),top1_agreement=np.mean([x['top1_reference']==x['top1_candidate'] for x in r]),exact_positions=sum(x['exact'] for x in r),mean_kl=np.mean([x['kl'] for x in r]),max_kl=max(x['kl'] for x in r),mean_tv=np.mean([x['tv'] for x in r]),mean_logit_rmse=np.mean([x['logit_rmse'] for x in r]),max_logit_error=max(x['logit_maxabs'] for x in r),delta_nll=np.mean([x['nll_candidate']-x['nll_reference'] for x in r]))
 out['ppl_ratio']=float(np.exp(out['delta_nll']));summary[mode]=out
 print(mode,json.dumps(out,indent=2))
(p/'shadow-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
