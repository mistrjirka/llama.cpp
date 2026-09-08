#!/usr/bin/env python3
"""Identical model + token IDs + cache size; explicit no speculation; normalized TG."""
import argparse,hashlib,json,os,pathlib,socket,statistics,subprocess,time,urllib.request
P=pathlib.Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--engine',choices=['fork','pxa'],required=True);p.add_argument('--fixture',choices=['synthetic','code'],default='synthetic');p.add_argument('--reps',type=int,default=2);p.add_argument('--env',action='append',default=[]);p.add_argument('--label');p.add_argument('--short',action='store_true');a=p.parse_args()
label=a.label or f'{a.engine}-{a.fixture}';out=P/'results'/f'{label}.json';log=P/'results'/f'{label}.log'
model='/models/PXA-Fusion4-35B-GGUF/PXA-Fusion4-35B-PXQ4.gguf'
binary='/models/llama-pxq4-build/bin/llama-server' if a.engine=='fork' else '/workspace/pxa-release-0908/unpacked/pxa-v2026.09.07-rc1/bin/llama-server'
with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='GPU-1d64d3a4-4aea-87ec-9048-33a5217efd79';env['PXA_AUTO_SPEC']='0';env['PXA_AUTO_SAMPLERS']='0'
if a.engine=='pxa':env['LD_LIBRARY_PATH']=str(pathlib.Path(binary).parent.parent/'lib')
for pair in a.env:k,v=pair.split('=',1);env[k]=v
slots=P/'results'/f'{label}-slot';slots.mkdir(exist_ok=True)
cmd=[binary,'-m',model,'-ngl','99','-c','131072','-np','1','-t','16','-b','2048','-ub','2048','-fa','on','-ctk','f16','-ctv','f16','-cram','0','--ctx-checkpoints','0','--slot-save-path',str(slots),'--no-warmup','--host','127.0.0.1','--port',str(port)]
if a.engine=='fork':cmd+=['--no-webui']
def post(path,obj=None):
    q=urllib.request.Request(f'http://127.0.0.1:{port}{path}',data=None if obj is None else json.dumps(obj).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(q,timeout=1800) as r:return json.load(r)
def run(prompt,n,cache=True):return post('/completion',{'prompt':prompt,'n_predict':n,'temperature':0,'seed':1234,'ignore_eos':True,'cache_prompt':cache,'id_slot':0,'stream':False,'return_tokens':True})
res={'engine':a.engine,'fixture':a.fixture,'cmd':cmd,'env':a.env,'rows':[]}
with log.open('w') as lf:
    proc=subprocess.Popen(cmd,env=env,stdout=lf,stderr=lf,start_new_session=True)
    try:
        for _ in range(1200):
            if proc.poll() is not None:raise RuntimeError(log.read_text()[-6000:])
            try:
                if post('/health').get('status')=='ok':break
            except Exception:time.sleep(.1)
        else:raise TimeoutError(str(log))
        if a.fixture=='synthetic':seq=json.load(open('/workspace/oai-qwen38-pp-lab/results/ornith15-seq101k.json'))[:101000]
        else:
            fixture=P/'results'/'code-prefix-101k.json'
            if fixture.exists():seq=json.loads(fixture.read_text())
            else:
                repo=P.parent.parent
                texts=[];files=[]
                for f in sorted((repo/'src').rglob('*.cpp')):
                    txt=f.read_text(errors='replace');texts.append(f'\n// FILE {f.relative_to(repo)}\n'+txt);files.append({'path':str(f.relative_to(repo)),'sha256':hashlib.sha256(txt.encode()).hexdigest()})
                    if sum(map(len,texts))>650000:break
                seq=post('/tokenize',{'content':''.join(texts),'add_special':True})['tokens'][:101000]
                fixture.write_text(json.dumps(seq));(P/'results'/'code-fixture-manifest.json').write_text(json.dumps(files,indent=2))
        assert len(seq)==101000,len(seq)
        res['token_ids_sha256']=hashlib.sha256(json.dumps(seq).encode()).hexdigest();res['distinct_tokens']=len(set(seq))
        if a.short:
            for rep in range(a.reps+1):
                t=time.perf_counter();z=run(seq[:5432],200,False);wall=time.perf_counter()-t;ti=z['timings']
                row={'rep':rep,'pp':ti['prompt_per_second'],'tg_steps_s':199000/ti['predicted_ms'],'wall_s':wall,'timings':ti,'sha256':hashlib.sha256(z['content'].encode()).hexdigest(),'content':z['content']}
                res['rows'].append(row);out.write_text(json.dumps(res,indent=2,ensure_ascii=False));print(label,row['pp'],row['tg_steps_s'],flush=True)
        else:
            t=time.perf_counter();z=run(seq[:100000],1);res['cold']={'wall_s':time.perf_counter()-t,'timings':z['timings']}
            print(label,'cold',res['cold'],flush=True)
            saved=post('/slots/0?action=save',{'filename':'prefix.bin'});res['save']=saved
            out.write_text(json.dumps(res,indent=2,ensure_ascii=False))
            for n in (64,512):
                for rep in range(a.reps+1):
                    restored=post('/slots/0?action=restore',{'filename':'prefix.bin'});assert restored.get('n_restored')==100000,restored
                    t=time.perf_counter();z=run(seq,n);wall=time.perf_counter()-t;ti=z['timings']
                    assert ti['prompt_n']==1000 and ti['predicted_n']==n,ti
                    row={'n':n,'rep':rep,'warmup':rep==0,'pp':ti['prompt_per_second'],'tg_steps_s':(n-1)*1000/ti['predicted_ms'],'wall_s':wall,'timings':ti,'sha256':hashlib.sha256(z['content'].encode()).hexdigest(),'content':z['content']}
                    res['rows'].append(row);out.write_text(json.dumps(res,indent=2,ensure_ascii=False))
                    print(label,n,rep,'pp',round(row['pp'],2),'tg',round(row['tg_steps_s'],2),'wall',round(wall,3),flush=True)
    finally:
        proc.terminate()
        try:proc.wait(timeout=10)
        except subprocess.TimeoutExpired:proc.kill();proc.wait()
