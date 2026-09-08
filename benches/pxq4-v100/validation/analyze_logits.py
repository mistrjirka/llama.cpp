#!/usr/bin/env python3
"""Full-distribution teacher-forced comparison; use only aligned, finite logits."""
import argparse, json, pathlib, numpy as np
p=argparse.ArgumentParser();p.add_argument('reference');p.add_argument('candidate');p.add_argument('--output',required=True);a=p.parse_args()
def load(prefix):
    m=json.loads(pathlib.Path(prefix+'.json').read_text());V=m['n_vocab'];n=len(m['rows']);z=np.memmap(prefix+'.f32',dtype='<f4',mode='r',shape=(n,V));return m,z
r,x=load(a.reference);s,y=load(a.candidate)
assert r['n_vocab']==s['n_vocab'] and len(r['rows'])==len(s['rows'])
assert [(q['sample'],q['position'],q['target']) for q in r['rows']]==[(q['sample'],q['position'],q['target']) for q in s['rows']]
rows=[]
for i,(xr,yr) in enumerate(zip(x,y)):
    u=np.array(xr,dtype=np.float64);v=np.array(yr,dtype=np.float64)
    assert np.all(np.isfinite(u)) and np.all(np.isfinite(v))
    ul=u-(u.max()+np.log(np.exp(u-u.max()).sum()));vl=v-(v.max()+np.log(np.exp(v-v.max()).sum()))
    prob=np.exp(ul);q=np.exp(vl);diff=v-u;center=diff-diff.mean();target=r['rows'][i]['target']
    ref_top=int(u.argmax());cand_top=int(v.argmax());tops=np.partition(u,-2)[-2:]
    rows.append(dict(sample=r['rows'][i]['sample'],position=r['rows'][i]['position'],kl=float(prob@(ul-vl)),tv=float(np.abs(prob-q).sum()/2),logit_rmse=float(np.sqrt(np.mean(center**2))),logit_maxerr=float(np.abs(center).max()),top1_equal=ref_top==cand_top,ref_margin=float(tops.max()-tops.min()),nll_ref=float(-ul[target]),nll_candidate=float(-vl[target])))
def metrics(z):
    out={'positions':len(z),'top1_agreement':float(np.mean([q['top1_equal'] for q in z]))}
    for key in ('kl','tv','logit_rmse','logit_maxerr'):
        values=np.array([q[key] for q in z]);out[key]={'mean':float(values.mean()),'p99':float(np.quantile(values,.99)),'max':float(values.max())}
    out['nll_reference']=float(np.mean([q['nll_ref'] for q in z]));out['nll_candidate']=float(np.mean([q['nll_candidate'] for q in z]));out['delta_nll']=out['nll_candidate']-out['nll_reference'];out['ppl_ratio']=float(np.exp(out['delta_nll']))
    out['ppl_reference']=float(np.exp(out['nll_reference']));out['ppl_candidate']=float(np.exp(out['nll_candidate']))
    out['mismatch_reference_margin_max']=max((q['ref_margin'] for q in z if not q['top1_equal']),default=0.)
    return out
result={'reference':a.reference,'candidate':a.candidate,'overall':metrics(rows),'by_dataset':{n:metrics([q for q in rows if q['sample']==n]) for n in dict.fromkeys(q['sample'] for q in rows)},'rows':rows}
pathlib.Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['overall'],indent=2))
