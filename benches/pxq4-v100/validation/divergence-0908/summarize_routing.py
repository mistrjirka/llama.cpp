#!/usr/bin/env python3
"""Distinguish permutations of chosen experts from actual routing changes."""
from pathlib import Path
import json
import numpy as np
from analyze_traces import load_tensor
D=Path('/models/llama-pxq4-build/divergence-0908')
V=Path(__file__).resolve().parent

def data(tag):
 return {r['name']:r for r in json.loads((D/f'trace-{tag}.trace.json').read_text())}

a=data('pxa');results={}
for tag in ['port','mmvf','fallback']:
 b=data(tag); rows=[]
 for layer in range(40):
  name=f'ffn_moe_topk-{layer}'
  if name not in a or name not in b:continue
  x=load_tensor(a[name]).ravel().astype(int); y=load_tensor(b[name]).ravel().astype(int)
  logits=load_tensor(a[f'ffn_moe_logits-{layer}']).ravel()
  ordered=np.sort(logits)[::-1]
  rows.append(dict(layer=layer,slot_mismatches=int(np.sum(x!=y)),
    selected_set_equal=set(x)==set(y),reference=x.tolist(),candidate=y.tolist(),
    boundary_logit_margin=float(ordered[7]-ordered[8])))
 print(tag,'changed expert sets',[(r['layer'],r['boundary_logit_margin'],r['reference'],r['candidate']) for r in rows if not r['selected_set_equal']])
 results[tag]=rows
(V/'routing.json').write_text(json.dumps(results,indent=2)+'\n')
# Align semantically corresponding operations with different graph names.
b=data('mmvf');pairs=[('ffn_inp_normed-0','attn_post_norm-0'),('ffn_moe_gate_par-0','ffn_moe_swiglu-0'),('ffn_moe_down-0','ffn_moe_down-0'),('l_out-0','l_out-0')]
for xn,yn in pairs:
 x=load_tensor(a[xn]);y=load_tensor(b[yn]);print(xn,yn,'relative RMS',np.linalg.norm(x-y)/np.linalg.norm(x),'max',np.max(np.abs(x-y)))
