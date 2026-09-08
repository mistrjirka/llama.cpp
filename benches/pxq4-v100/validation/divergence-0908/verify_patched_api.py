#!/usr/bin/env python3
"""Verify the UTF-8 fix in a CUDA build; do not mix API and logit schedules."""
import hashlib,json,os,pathlib,socket,subprocess,time,urllib.request,urllib.error,math
P=pathlib.Path(__file__).resolve().parent
B=pathlib.Path('/models/pxa-fullprobs-build/bin/llama-server')
with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='GPU-1d64d3a4-4aea-87ec-9048-33a5217efd79',PXA_AUTO_SPEC='0',PXA_AUTO_SAMPLERS='0')
for name in ['PXA_REFERENCE','PXA_ENHANCE','GGML_CUDA_CUBLAS_COMPUTE_TYPE']:env.pop(name,None)
cmd=[str(B),'-m','/models/PXA-Fusion4-35B-GGUF/PXA-Fusion4-35B-PXQ4.gguf','-ngl','99','-c','2048','-np','1','-b','256','-ub','256','-fa','on','-ctk','f16','-ctv','f16','--host','127.0.0.1','--port',str(port),'--ctx-checkpoints','0']
def request(path,body=None):
 q=urllib.request.Request(f'http://127.0.0.1:{port}{path}',data=None if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(q,timeout=120) as r:return r.status,json.load(r)
with (P/'patched-cuda-api.log').open('w') as log:
 proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=log)
 try:
  for _ in range(800):
   if proc.poll() is not None:raise RuntimeError('server exited')
   try:
    if request('/health')[1].get('status')=='ok':break
   except (OSError,ValueError):time.sleep(.1)
  else:raise TimeoutError('startup')
  fixtures=json.loads((P/'tokens.json').read_text());rows=[]
  for sample in fixtures[:2]:
   status,d=request('/completion',dict(prompt=sample['tokens'][:401],n_predict=1,n_probs=248320,temperature=0,post_sampling_probs=False,cache_prompt=False,repeat_penalty=1.0,stream=False))
   top=d['completion_probabilities'][0];items=top.get('top_logprobs',top.get('probs'))
   assert len(items)==248320
   # Commit 896c189 fixes serialization but does not add token IDs to this schema.
   # Raw C-API logits, not token strings, are used for the distribution comparison.
   has_ids=all('id' in item for item in items)
   if has_ids:assert {int(item['id']) for item in items}==set(range(248320))
   probabilities=[math.exp(item['logprob']) if 'logprob' in item else float(item['prob']) for item in items]
   assert all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities)
   rows.append(dict(sample=sample['name'],http_status=status,entries=len(items),has_token_ids=has_ids,entry_keys=sorted(items[0]),probability_sum=sum(probabilities),zero_probabilities=sum(p==0 for p in probabilities),prompt_n=d['timings']['prompt_n']))
  info=dict(binary=str(B),binary_sha256=hashlib.sha256(B.read_bytes()).hexdigest(),argv=cmd,rows=rows)
  (P/'patched-cuda-api.json').write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info,indent=2))
 finally:
  proc.terminate()
  try:proc.wait(15)
  except subprocess.TimeoutExpired:proc.kill();proc.wait()
