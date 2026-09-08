#!/usr/bin/env python3
"""Compare captured graph tensors without assuming contiguous views."""
import argparse
import collections
import json
from pathlib import Path
import numpy as np


def load_tensor(row):
    dtype = {'f32': '<f4', 'f16': '<f2', 'i32': '<i4'}[row['type']]
    buf = Path(row['file']).read_bytes()
    return np.ndarray(shape=tuple(row['ne'][::-1]), dtype=dtype,
                      buffer=buf, strides=tuple(row['nb'][::-1])).copy().astype(np.float64)


def keyed(rows):
    count = collections.Counter()
    result = {}
    for row in rows:
        name = row['name']; k = (name, count[name]); count[name] += 1
        result[k] = row
    return result


def compare(ref, cand):
    a = keyed(json.loads(Path(ref).read_text()))
    b = keyed(json.loads(Path(cand).read_text()))
    out = []
    for k, r in a.items():
        if k not in b or r['ne'] != b[k]['ne'] or any(n == 0 for n in r['ne']): continue
        x, y = load_tensor(r), load_tensor(b[k])
        err = np.abs(x-y); peak = max(float(np.max(np.abs(x))), 1e-30)
        rms = max(float(np.sqrt(np.mean(x*x))),1e-30)
        out.append(dict(name=k[0], occurrence=k[1], shape=r['ne'], op=r['op'],
                        equal=bool(np.array_equal(x,y)), max_abs=float(err.max()),
                        peak_relative=float(err.max()/peak),
                        rms_relative=float(np.sqrt(np.mean(err*err))/rms),
                        different=int(np.count_nonzero(x!=y)), total=x.size,
                        sources=r['sources']))
    return {'reference':ref,'candidate':cand,'rows':out,
            'only_reference':[list(k) for k in a.keys()-b.keys()],
            'only_candidate':[list(k) for k in b.keys()-a.keys()]}

if __name__ == '__main__':
    ap=argparse.ArgumentParser();ap.add_argument('reference');ap.add_argument('candidate');ap.add_argument('--output',required=True)
    args=ap.parse_args(); result=compare(args.reference,args.candidate)
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print('Common tensors:',len(result['rows']))
    nonzero=[r for r in result['rows'] if not r['equal']]
    print('First differences:')
    for row in nonzero[:30]:
        print(row['name'],row['op'],row['shape'],'max_abs',row['max_abs'],'relative_rms',row['rms_relative'])
    print('Only-reference:',result['only_reference'][:12])
    print('Only-candidate:',result['only_candidate'][:12])
