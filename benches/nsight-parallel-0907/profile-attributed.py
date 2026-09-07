#!/usr/bin/env python3
"""Nsight trace of four cached Ornith requests; NVTX phases exclude restore/startup.
NVTX lives in this driver; phase timestamps select child-server GPU events.
Do not interpret instrumented latency as a replacement for unprofiled benchmarks.
"""
import concurrent.futures as cf
import contextlib, ctypes, hashlib, importlib.util, json, os, pathlib, subprocess, threading, time
R=pathlib.Path('/workspace/oai-qwen38-pp-lab/results/nsight-parallel-0907')
O=R.parent/'parallel-research-0907'
B=R.parent/'parallel-experiments-0907/final-bin'
sp=importlib.util.spec_from_file_location('fixture',O/'sweep.py')
s=importlib.util.module_from_spec(sp); sp.loader.exec_module(s)
s.PORT=32507;s.BIN=str(B/'llama-server');s.ENV['LD_LIBRARY_PATH']=str(B)
s.ENV['LD_PRELOAD']=str(R/'mtp_markers.so')
markers=ctypes.CDLL(str(R/'markers.so'));markers.mark_push.argtypes=[ctypes.c_char_p]
@contextlib.contextmanager
def region(name):
    markers.mark_push(name.encode())
    try: yield
    finally: markers.mark_pop()
def batch(prompts, n):
    barrier=threading.Barrier(4)
    with cf.ThreadPoolExecutor(max_workers=4) as ex:
        rows=list(ex.map(lambda i:s.comp(i,prompts[i],n,barrier),range(4)))
    return rows
variants=[('attributed-mtp3',3,True,None)]
results=[]
for name,depth,compact,pack in variants:
    for k in list(s.ENV):
        if k.startswith('LLAMA_EXPERIMENT_') or k.startswith('GGML_CUDA_VOLTA_Q8_MULTI'):s.ENV.pop(k,None)
    if compact:s.ENV['LLAMA_EXPERIMENT_PREFIX_FIRST']='1'
    if pack:
        s.ENV['GGML_CUDA_VOLTA_Q8_MULTI']='1';s.ENV['GGML_CUDA_VOLTA_Q8_MULTI_PACK']=str(pack)
    with region(name+'/startup'):
        p,log=s.start(depth,'../nsight-parallel-0907/'+name)
    try:
        with region(name+'/warmup'):
            s.restore();batch(s.PROMPTS,16)
        with region(name+'/restore'):
            s.restore()
        with region(name+'/PP'):
            t=time.perf_counter();pp=batch(s.PROMPTS,1);ppwall=time.perf_counter()-t
        for r in pp:
            assert r['timings']['cache_n']==100000 and r['timings']['prompt_n']==1000,r
        prompts=[q+r['tokens'] for q,r in zip(s.PROMPTS,pp)]
        with region(name+'/TG'):
            t=time.perf_counter();tg=batch(prompts,128);tgwall=time.perf_counter()-t
        for r in tg:
            assert r['timings']['cache_n']>=101000 and r['timings']['prompt_n']<=2,r
        results.append({'variant':name,'server_pid':p.pid,'args':s.args(depth),'env':{k:v for k,v in s.ENV.items() if k.startswith(('GGML_','LLAMA_EXPERIMENT'))},'pp_wall_s':ppwall,'tg_wall_s':tgwall,'pp':pp,'tg':tg})
        (R/'attributed-measurements.json').write_text(json.dumps(results,indent=2))
        print('PROFILE',name,'PP',round(ppwall,3),'TG',round(tgwall,3),'TG_each',[round(r['timings']['predicted_per_second'],2) for r in tg],flush=True)
    finally:
        with region(name+'/shutdown'):s.stop(p,log)
print('DONE',flush=True)
