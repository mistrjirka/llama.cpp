#!/usr/bin/env python3
"""Integration tests use fresh production baseline and isolated candidate binaries.
All runs require the outer Development Sandbox gpu:all lock.
"""
from __future__ import annotations
import argparse, base64, concurrent.futures as cf, hashlib, importlib.util
import json, os, pathlib, re, statistics, struct, threading, time, zlib
R=pathlib.Path('/workspace/oai-qwen38-pp-lab/results/mtp-production-merge-0907')
F=R.parent/'parallel-refined-0907'
spec=importlib.util.spec_from_file_location('fixture',R.parent/'parallel-research-0907/sweep.py')
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
s.PORT=32610;s.OUT=R
BASE_ENV=s.ENV.copy()
for key in list(BASE_ENV):
    if key.startswith(('LLAMA_EXPERIMENT_','GGML_CUDA_VOLTA_Q8_MULTI','GGML_CUDA_VOLTA_Q8_REF')) or key in ('GGML_CUDA_VOLTA_Q8_COOP','GGML_CUDA_VOLTA_Q8_PV_F32','LD_PRELOAD','LLAMA_ARG_BACKEND_SAMPLING','LLAMA_MTP_KV_ONLY','LLAMA_MTP_BULK_HIDDEN','LLAMA_SAMPLING_VIEW'):
        BASE_ENV.pop(key)
original_args=s.args
f=json.loads((F/'real-fixture.json').read_text())
for i in range(4):
    for ext in ('','.draft','.spec'):
        p=R/f'real100k-{i}.bin{ext}'
        if not p.exists():p.symlink_to(F/p.name)

def setup(binary='final-bin',disable=False,interpose=False):
    B=R/binary;s.BIN=str(B/'llama-server');s.ENV=BASE_ENV.copy();s.ENV['LD_LIBRARY_PATH']=str(B)
    if disable:
        s.ENV.update({k:'0' for k in ('LLAMA_MTP_KV_ONLY','LLAMA_MTP_BULK_HIDDEN','LLAMA_SAMPLING_VIEW')})
    if interpose:
        s.ENV['LD_PRELOAD']=str(R/'validate-kv.so')+':'+str(R/'verify-views.so')
        s.ENV['KV_CHECK_DIR']=str(R)
    s.args=original_args

def start(depth,label):
    p,l=s.start(depth,label)
    assert str(pathlib.Path(s.BIN).parent/'libllama') in pathlib.Path(f'/proc/{p.pid}/maps').read_text()
    return p,l

def restore():
    for i in range(4):s.req(f'/slots/{i}?action=erase',{})
    for i in range(4):assert s.req(f'/slots/{i}?action=restore',{'filename':f'real100k-{i}.bin'})['n_restored']==100000

def parallel(prompts,n):
    b=threading.Barrier(len(prompts));t=time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=len(prompts)) as ex:
        out=list(ex.map(lambda i:s.comp(i,prompts[i],n,b),range(len(prompts))))
    for r in out:assert r['timings']['predicted_n']==n,r
    return out,time.perf_counter()-t

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as inp:
        for block in iter(lambda:inp.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def write(name,value):
    (R/name).write_text(json.dumps(value,indent=2)+'\n')

def parity():
    rows=[]
    for label,binary,disabled in [('old-production','baseline-bin',False),('merged-disabled','final-bin',True),('merged-default','final-bin',False)]:
        setup(binary,disabled);p,l=start(3,'parity-'+label)
        try:
            restore()
            # No speculative token is requested: compare target probability records on fixed histories.
            probs=[]
            for i in range(4):
                x=s.req('/completion',{'prompt':f['prompts'][i],'id_slot':i,'cache_prompt':True,'n_predict':1,'temperature':0.0,'seed':1234,'n_probs':20,'post_sampling_probs':False,'return_tokens':True,'ignore_eos':True})
                probs.append({'tokens':x['tokens'],'probs':x.get('completion_probabilities')})
            restore()
            out=[s.comp(i,f['prompts'][i],64) for i in range(4)]
            s.req('/slots/0?action=save',{'filename':'parity-slot.bin'})
            hashes={e:sha(R/('parity-slot.bin'+e)) for e in ('','.draft','.spec')}
            s.req('/slots/0?action=erase',{});s.req('/slots/0?action=restore',{'filename':'parity-slot.bin'})
            tail=s.comp(0,f['prompts'][0]+out[0]['tokens'],32)
            rows.append({'arm':label,'probability_records':probs,'requests':out,'snapshot_hashes':hashes,'continued':tail})
            write('parity-raw.json',rows)
            print('PARITY',label,[x['sha'][:10] for x in out],flush=True)
        finally:s.stop(p,l)
    old,off,on=rows
    checks={
        'target_probability_records_equal_old':old['probability_records']==on['probability_records'],
        'default_vs_disabled_tokens_equal':all(a['tokens']==b['tokens'] for a,b in zip(off['requests'],on['requests'])),
        'default_vs_disabled_counters_equal':all((a['timings'].get('draft_n'),a['timings'].get('draft_n_accepted'))==(b['timings'].get('draft_n'),b['timings'].get('draft_n_accepted')) for a,b in zip(off['requests'],on['requests'])),
        'default_vs_disabled_snapshot_bytes_equal':off['snapshot_hashes']==on['snapshot_hashes'],
        'default_vs_disabled_restore_continuation_equal':off['continued']['tokens']==on['continued']['tokens'],
        'old_production_output_tokens_equal':all(a['tokens']==b['tokens'] for a,b in zip(old['requests'],on['requests']))}
    write('parity-summary.json',checks);print('CHECKS',checks,flush=True)
    assert all(checks.values()),checks

def image_smoke():
    # Locally constructed test input; no network image fetch required.
    def chunk(k,d):return struct.pack('!I',len(d))+k+d+struct.pack('!I',zlib.crc32(k+d)&0xffffffff)
    width=height=112
    raw=b''.join(b'\0'+bytes([255,255,255])*width for _ in range(height))
    png=b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('!2I5B',width,height,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(raw))+chunk(b'IEND',b'')
    result=s.req('/v1/chat/completions',{'messages':[{'role':'user','content':[{'type':'text','text':'Describe the image briefly.'},{'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(png).decode()}}]}],'max_tokens':16,'temperature':0.0,'seed':1234})
    assert result.get('choices') and result.get('usage',{}).get('completion_tokens',0)>0,result
    return {'completion_tokens':result['usage']['completion_tokens'],'no_server_error':True}

def interposer():
    setup(interpose=True);p,l=start(3,'merged-validators')
    try:
        restore();out,_=parallel(f['prompts'],32)
        for i in range(4):s.req(f'/slots/{i}?action=save',{'filename':f'checked{i}.bin'})
        for i in range(4):s.req(f'/slots/{i}?action=erase',{})
        for i in range(4):s.req(f'/slots/{i}?action=restore',{'filename':f'checked{i}.bin'})
        resumed,_=parallel([f['prompts'][i]+out[i]['tokens'] for i in range(4)],16)
        image=image_smoke()
        write('validator-requests.json',{'initial':out,'restored':resumed,'image':image})
    finally:s.stop(p,l)
    text=(R/'merged-validators.log').read_text(errors='replace')
    kv=re.search(r'KV_CHECK TOTAL calls=(\d+) checks=(\d+) bytes=(\d+)',text)
    views=re.search(r'VIEW_CHECK outputs=(\d+) hidden_rows=(\d+)',text)
    assert kv and int(kv[2])>=4 and views and int(views[1])>0 and int(views[2])>0,text[-2500:]
    assert 'cache-only refresh=1, bulk hidden reads=1' in text
    checks={'kv_calls':int(kv[1]),'kv_shapes':int(kv[2]),'compared_bytes':int(kv[3]),'sampling_views':int(views[1]),'hidden_rows':int(views[2]),'image':image}
    write('validator-summary.json',checks);print('VALIDATORS',checks,flush=True)

def qwen_args(depth):
    return [s.BIN,'-m','/models/Qwen3.8-27B/Qwen3.8-27B-UD-Q5_K_XL.gguf','--host','127.0.0.1','--port',str(s.PORT),'--ctx-size','32768','--parallel','4','--gpu-layers','all','--device','CUDA1,CUDA0','--tensor-split','14,35','--split-mode','layer','--fit','off','--flash-attn','on','--batch-size','512','--ubatch-size','256','--pipeline-copies','1','--cache-type-k','q8_0','--cache-type-v','q8_0','--cache-ram','0','--no-cache-idle-slots','--spec-type','draft-mtp','--spec-draft-n-max','3','--spec-draft-type-k','q8_0','--spec-draft-type-v','q8_0','--spec-draft-ubatch','128','--spec-mtp-defer-prompt','--slots','--perf','--slot-save-path',str(R),'--no-warmup']

def qwen():
    setup(interpose=True);s.args=qwen_args;p,l=start(3,'qwen-validators')
    try:
        text=pathlib.Path('common/sampling.cpp').read_text()
        toks=s.req('/tokenize',{'content':text,'add_special':True})['tokens'][:2000]
        s.comp(0,toks,0);prompts=[]
        for i in range(4):
            suffix=s.req('/tokenize',{'content':f'\nReview memory lifetime and concurrency risk {i}.','add_special':False})['tokens']
            prompts.append(toks+suffix)
        out,_=parallel(prompts,48);write('qwen-requests.json',out)
    finally:s.stop(p,l)
    text=(R/'qwen-validators.log').read_text(errors='replace')
    kv=re.search(r'KV_CHECK TOTAL calls=(\d+) checks=(\d+) bytes=(\d+)',text)
    views=re.search(r'VIEW_CHECK outputs=(\d+) hidden_rows=(\d+)',text)
    assert kv and int(kv[2])>=3 and views and int(views[1])>0 and int(views[2])>0,text[-2500:]
    write('qwen-summary.json',{'kv_calls':int(kv[1]),'kv_shapes':int(kv[2]),'sampling_views':int(views[1]),'hidden_rows':int(views[2])});print('QWEN PASS',kv.group(),views.group(),flush=True)

def zero():
    setup();orig=s.args
    def args(_):
        cmd=orig(3);cmd[cmd.index('--spec-draft-n-max')+1]='0';return cmd
    s.args=args;p,l=start(3,'zero-budget')
    try:
        restore();out,_=parallel(f['prompts'],32)
        assert all(x['timings'].get('draft_n',0)==0 and x['timings'].get('draft_n_accepted',0)==0 for x in out),out
        write('zero-summary.json',{'completed_requests':len(out),'all_draft_counts_zero':True});print('ZERO PASS',flush=True)
    finally:s.stop(p,l)

def bank():
    # More parked conversations than active slots, including a genuine process restart.
    prompts=[f['histories'][i][i*10000:i*10000+4000] for i in range(3)]
    tail=f['prompts'][0][100000:];rows={}
    def bank_comp(prompt,n):
        # Explicit id_slot bypasses automatic replacement/parking by design.
        return s.req('/completion',{'prompt':prompt,'cache_prompt':True,'n_predict':n,
            'ignore_eos':True,'return_tokens':True,'temperature':0.0,'seed':1234})
    def args(depth):
        a=original_args(3)
        for k,v in [('--ctx-size','16000'),('--kv-unified-per-slot','8000'),('--parallel','1'),('--cache-ram','4096'),('--slot-prompt-similarity','0.5')]:a[a.index(k)+1]=v
        return a
    setup();s.args=args;p,l=start(3,'final-bank-control')
    try:
        bank_comp(prompts[0],0);rows['reference']=bank_comp(prompts[0]+tail,64)
    finally:s.stop(p,l)
    setup();s.args=args;p,l=start(3,'final-bank-park')
    try:
        for prompt in prompts:bank_comp(prompt,0)
        rows['saved']=s.req('/prompt-cache?action=save',{'filename':'parked-bank.bin'})
        assert (R/'parked-bank.bin').stat().st_size>10000,rows['saved']
    finally:s.stop(p,l)
    setup();s.args=args;p,l=start(3,'final-bank-restart')
    try:
        rows['loaded']=s.req('/prompt-cache?action=restore',{'filename':'parked-bank.bin'})
        rows['resumed']=bank_comp(prompts[0]+tail,64)
        assert rows['resumed']['timings']['cache_n']==len(prompts[0]),rows['resumed']
        assert rows['reference']['tokens']==rows['resumed']['tokens'],rows
    finally:s.stop(p,l)
    write('final-bank-result.json',rows);print('PARKED BANK PASS',rows['saved'],rows['loaded'],flush=True)

def perf():
    rows=[]
    for depth in (0,1,3):
        for block,arm in enumerate(('baseline-bin','final-bin','final-bin','baseline-bin')):
            setup(arm);p,l=start(depth,f'perf-d{depth}-{block}-{arm}')
            try:
                for rep in range(3):
                    restore();out,wall=parallel(f['prompts'],128)
                    assert all(x['timings']['cache_n']==100000 for x in out),out
                    row={'depth':depth,'arm':arm,'block':block,'warmup':rep==0,'wall_s':wall,'mean_tg':statistics.mean(x['timings']['predicted_per_second'] for x in out),'requests':out}
                    rows.append(row);write('performance-raw.json',rows)
                    print('PERF',depth,arm,rep,round(wall,4),round(row['mean_tg'],3),flush=True)
            finally:s.stop(p,l)
    summary={}
    for depth in (0,1,3):
        d={}
        for arm in ('baseline-bin','final-bin'):
            rs=[r for r in rows if r['depth']==depth and r['arm']==arm and not r['warmup']]
            d[arm]={'n':len(rs),**{k:{'mean':statistics.mean(r[k] for r in rs),'min':min(r[k] for r in rs),'max':max(r[k] for r in rs)} for k in ('wall_s','mean_tg')}}
        d['wall_reduction_percent']=100*(1-d['final-bin']['wall_s']['mean']/d['baseline-bin']['wall_s']['mean']);summary[str(depth)]=d
    write('performance-summary.json',summary);print('PERF SUMMARY',json.dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('suite',choices=['parity','interposer','qwen','zero','bank','perf']);a=p.parse_args()
    globals()[a.suite]()
