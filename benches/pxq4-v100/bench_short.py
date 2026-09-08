#!/usr/bin/env python3
"""Serial, V100-only PXQ4 smoke/throughput tests with normalized timed TG."""
import argparse, hashlib, json, os, pathlib, statistics, subprocess, time, urllib.request, socket
P = pathlib.Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument('--binary', default='/models/llama-pxq4-build/bin/llama-server')
ap.add_argument('--label', default='rpb-sweep')
ap.add_argument('--rpb', default='2,4,8')
ap.add_argument('--reps', type=int, default=3)
ap.add_argument('--tokens', type=int, default=256)
ap.add_argument('--pxa', action='store_true')
ap.add_argument('--env', action='append', default=[])
a = ap.parse_args()
model='/models/PXA-Fusion4-35B-GGUF/PXA-Fusion4-35B-PXQ4.gguf'
res=[]
def post(port, payload):
    req=urllib.request.Request(f'http://127.0.0.1:{port}/completion',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=300) as r: return json.load(r)
for rpb in map(int,a.rpb.split(',')):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1",0)); port=sock.getsockname()[1]
    env=os.environ.copy(); env['CUDA_VISIBLE_DEVICES']='GPU-1d64d3a4-4aea-87ec-9048-33a5217efd79'
    env['GGML_CUDA_PXQ4_RPB']=str(rpb)
    env['PXA_AUTO_SPEC']='0'; env['PXA_AUTO_SAMPLERS']='0'
    for pair in a.env:
        k,v=pair.split('=',1); env[k]=v
    cmd=[a.binary,'-m',model,'-ngl','99','-c','8192','-b','512','-ub','512','-fa','on','-ctk','f16','-ctv','f16','-np','1','-t','16','--host','127.0.0.1','--port',str(port)]
    if a.pxa: env['LD_LIBRARY_PATH']=str(pathlib.Path(a.binary).parent.parent/'lib')
    else: cmd+=['--no-webui']
    log=P/'results'/f'{a.label}-r{rpb}.log'
    with log.open('w') as lf:
        proc=subprocess.Popen(cmd,env=env,stdout=lf,stderr=lf)
        try:
            for _ in range(400):
                if proc.poll() is not None: raise RuntimeError(f'server exited: {log}')
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=2) as r:
                        if json.load(r).get('status')=='ok': break
                except Exception: time.sleep(.1)
            else: raise TimeoutError(str(log))
            rows=[]
            for i in range(a.reps+1):
                payload={'prompt':'The capital of France is','n_predict':a.tokens,'temperature':0,'seed':1234,'ignore_eos':True,'cache_prompt':False,'return_tokens':True,'stream':False}
                t=time.perf_counter(); d=post(port,payload); elapsed=time.perf_counter()-t
                ti=d['timings']; n=ti['predicted_n']
                item={'i':i,'wall_s':elapsed,'tg_steps_s':(n-1)*1000/ti['predicted_ms'],'timings':ti,'sha256':hashlib.sha256(d['content'].encode()).hexdigest(),'content':d['content'],'tokens':d.get('tokens')}
                rows.append(item)
            out={'label':a.label,'rpb':rpb,'cmd':cmd,'overrides':a.env,'rows':rows,'median_tg_steps_s':statistics.median(r['tg_steps_s'] for r in rows[1:]),'median_wall_s':statistics.median(r['wall_s'] for r in rows[1:])}
            res.append(out)
            (P/'results'/f'{a.label}.json').write_text(json.dumps(res,indent=2,ensure_ascii=False))
            print(a.label,'rpb',rpb,'TG timed',round(out['median_tg_steps_s'],3),'wall',round(out['median_wall_s'],3),'hashes',len(set(r['sha256'] for r in rows)), 'first:',repr(rows[1]['content'][:100]),flush=True)
        finally:
            proc.terminate()
            try: proc.wait(timeout=10)
            except subprocess.TimeoutExpired: proc.kill(); proc.wait()
