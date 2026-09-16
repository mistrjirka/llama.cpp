#!/usr/bin/env python3
"""Run numerical checks and assert the optimized GPU route actually executed."""
import collections,os,re,subprocess,sys
env=os.environ.copy()
env['GGML_CUDA_DIAG_ROUTER_MATCH']='1'
for name in ('GGML_CUDA_DIAG_DISABLE_TOPK_MOE','GGML_CUDA_DISABLE_FUSION','ROUTER_BENCHMARK'):
    env.pop(name,None)
p=subprocess.run([sys.argv[1]],env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
print(p.stdout,end='')
if p.returncode: print(p.stderr,end='',file=sys.stderr);sys.exit(p.returncode)
calls=collections.Counter();staged=collections.Counter()
for name,n,bytes_ in re.findall(r'ROUTER_FUSED name=(\w+) rows=(\d+) staged_logit_bytes=(\d+)',p.stderr):
    calls[name,int(n)]+=1
    if int(bytes_): staged[name,int(n)]+=1
errors=[]
for n in (1,7,8,9,33,128,257,512):
    for name in ('direct_logits','alias_logits'):
        if calls[name,n]<1: errors.append(f'{name}/{n}: optimized route not exercised')
if any(name=='reference_logits' for name,n in calls): errors.append('reference unexpectedly fused')
if not sum(staged.values()): errors.append('no aliased multi-block input was protected')
print('FUSED_PATH_COVERAGE',dict((str(k),v) for k,v in calls.items()))
print('STAGED_PATH_COVERAGE',dict((str(k),v) for k,v in staged.items()))
if errors: print('\n'.join(errors),file=sys.stderr);sys.exit(1)
print('PASS numerical oracle, alias invariance and actual fused-kernel coverage')
