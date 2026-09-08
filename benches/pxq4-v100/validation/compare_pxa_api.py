#!/usr/bin/env python3
"""Compare full-vocabulary distributions through PXA's released public HTTP API.
No dependency on unavailable binary-matched C++ headers. n_probs=V avoids a
partial-top-N normalizer issue visible in the public source.
"""
import json,os,pathlib,socket,subprocess,time,urllib.request,urllib.error,numpy as np
P=pathlib.Path(__file__).resolve().parent;D=pathlib.Path('/models/llama-pxq4-build/validation-logits')
meta=json.loads((D/'native.json').read_text());V=meta['n_vocab'];z=np.memmap(D/'native.f32',dtype='<f4',mode='r',shape=(len(meta['rows']),V))
release=pathlib.Path('/workspace/pxa-release-0908/unpacked/pxa-v2026.09.07-rc1')
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='GPU-1d64d3a4-4aea-87ec-9048-33a5217efd79',LD_LIBRARY_PATH=str(release/'lib'),PXA_AUTO_SPEC='0',PXA_AUTO_SAMPLERS='0')
with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
cmd=[str(release/'bin/llama-server'),'-m','/models/PXA-Fusion4-35B-GGUF/PXA-Fusion4-35B-PXQ4.gguf','--host','127.0.0.1','--port',str(port),'-ngl','99','-c','2048','-np','1','-b','256','-ub','256','-fa','on','-ctk','f16','-ctv','f16','--ctx-checkpoints','0']
def request(path,body=None):
    data=None if body is None else json.dumps(body).encode();q=urllib.request.Request(f'http://127.0.0.1:{port}{path}',data=data,headers={'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(q,timeout=180) as f:return json.load(f)
    except urllib.error.HTTPError as e:
        message=e.read().decode(errors='replace');(P/'pxa-api-error.txt').write_text(message)
        raise RuntimeError(f'PXA HTTP {e.code}: {message[:1000]}') from e
rows=[]
with (P/'pxa-api-server.log').open('w') as log:
 proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=log)
 try:
  for _ in range(600):
   if proc.poll() is not None:raise RuntimeError('PXA server exited')
   try:
    if request('/health')['status']=='ok':break
   except Exception:time.sleep(.1)
  else:raise TimeoutError('PXA startup')
  for sample in meta['datasets']:
   for pos in (400,511):
    i=next(i for i,r in enumerate(meta['rows']) if r['sample']==sample['name'] and r['position']==pos)
    result=request('/completion',{'prompt':sample['tokens'][:pos+1],'n_predict':1,'temperature':0,'n_probs':V,'post_sampling_probs':False,'cache_prompt':False,'repeat_penalty':1.0,'ignore_eos':False,'stream':False})
    if 'completion_probabilities' not in result:raise RuntimeError('missing probability field '+str(result.keys()))
    item=result['completion_probabilities'][0]
    probs=item.get('top_logprobs',item.get('probs'))
    if not probs:raise RuntimeError('unexpected probability keys '+str(item.keys()))
    if len(probs)!=V:raise RuntimeError(f'truncated probabilities {len(probs)} != {V}')
    lp=np.full(V,np.nan,dtype='f8')
    for q in probs:lp[q['id']]=q['logprob']
    if not np.all(np.isfinite(lp)):raise RuntimeError('nonfinite or missing vocabulary entry')
    u=np.asarray(z[i],dtype='f8');ln=u-u.max()-np.log(np.exp(u-u.max()).sum());ref=np.exp(lp);test=np.exp(ln)
    mass=float(ref.sum());assert abs(mass-1)<1e-4,mass
    positive=ref>0;kl=float(ref[positive]@(lp[positive]-ln[positive]));row=dict(sample=sample['name'],position=pos,vocab_count=V,probability_sum=mass,top1_pxa=int(lp.argmax()),top1_port=int(u.argmax()),kl_pxa_to_port=kl,tv=float(np.abs(ref-test).sum()/2))
    rows.append(row);(P/'pxa-api-comparison.json').write_text(json.dumps({'argv':cmd,'rows':rows},indent=2));print(row,flush=True)
 finally:
  proc.terminate()
  try:proc.wait(15)
  except subprocess.TimeoutExpired:proc.kill();proc.wait()
