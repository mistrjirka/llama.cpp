#!/usr/bin/env python3
"""Recreate the recorded matched benchmark summary from retained JSONs."""
import json,pathlib,statistics
P=pathlib.Path(__file__).resolve().parent
names=('wmma-long','pxa-long','fork-code-long','pxa-code-long')
summary=[]
for name in names:
    d=json.loads((P/'results'/f'{name}.json').read_text())
    r={'label':name,'fixture':d['fixture'],'engine':d['engine'],'token_ids_sha256':d['token_ids_sha256'],'distinct_tokens':d['distinct_tokens'],'cold_pp':d['cold']['timings']['prompt_per_second'],'summary':{}}
    for n in (64,512):
        xs=[x for x in d['rows'] if x['n']==n and not x['warmup']]
        assert len(xs)==2
        assert all(x['timings']['prompt_n']==1000 and x['timings']['predicted_n']==n for x in xs)
        if d['engine']=='fork':assert all(x['timings']['cache_n']==100000 for x in xs)
        r['summary'][str(n)]={k:statistics.mean(x[k] for x in xs) for k in ('pp','tg_steps_s','wall_s')}
        r['summary'][str(n)]['content_hashes']=sorted(set(x['sha256'] for x in xs))
    summary.append(r)
for fi in ('synthetic','code'):
    pair=[r for r in summary if r['fixture']==fi]
    assert pair[0]['token_ids_sha256']==pair[1]['token_ids_sha256']
(P/'results'/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
lines=['| Prefix | Engine | PP +64 | TG 64 | PP +512 | TG 512 |','|---|---|---:|---:|---:|---:|']
for r in summary:
    x,y=r['summary']['64'],r['summary']['512']
    lines.append(f"| {r['fixture']} | {r['engine']} | {x['pp']:.1f} | {x['tg_steps_s']:.2f} | {y['pp']:.1f} | {y['tg_steps_s']:.2f} |")
print('\n'.join(lines))
(P/'results'/'table.md').write_text('\n'.join(lines)+'\n')
