#!/usr/bin/env python3
"""Short-context regression of the actual Qwen tensor split, FP16 draft KV and shortlist.
Not a capacity or long-context performance measurement.
"""
import importlib.util,json,pathlib,re
R=pathlib.Path('/workspace/oai-qwen38-pp-lab/results/mtp-production-merge-0907')
sp=importlib.util.spec_from_file_location('regression',R/'regression.py')
t=importlib.util.module_from_spec(sp);sp.loader.exec_module(t)
s=t.s
shortlist=pathlib.Path('/workspace/llama-v100-optimized/data/mtp-shortlists/qwen38-27b-exact-131072.i32')
assert shortlist.is_file(),shortlist
rows=[]
for label,binary,disabled,checked in [('baseline','baseline-bin',False,False),('defaults','final-bin',False,False),('checked','final-bin',False,True)]:
    t.setup(binary,disabled,checked);s.ENV['GGML_CUDA_QWEN35_MTP_SHORTLIST']=str(shortlist)
    def args(n):
        a=t.qwen_args(n)
        for key,value in [('--parallel','1'),('--split-mode','tensor'),('--tensor-split','4,5'),('--batch-size','4096'),('--ubatch-size','2048'),('--spec-draft-ubatch','512'),('--spec-draft-type-k','f16'),('--spec-draft-type-v','f16')]:
            a[a.index(key)+1]=value
        return a
    s.args=args;p,l=t.start(3,'qwen-tp-'+label)
    try:
        if not rows:
            code=pathlib.Path('common/sampling.cpp').read_text()
            parent=s.req('/tokenize',{'content':code,'add_special':True})['tokens'][:2000]
            suffix=s.req('/tokenize',{'content':'\nExplain two concrete memory-safety risks and a regression test.','add_special':False})['tokens']
        s.comp(0,parent,0)
        result=s.comp(0,parent+suffix,64)
        s.req('/slots/0?action=save',{'filename':'qwen-tp-state.bin'})
        hashes={e:t.sha(R/('qwen-tp-state.bin'+e)) for e in ('','.draft','.spec')}
        rows.append({'label':label,'result':result,'hashes':hashes})
        print('QWEN TP',label,result['sha'],flush=True)
    finally:s.stop(p,l)
checks={'production_defaults_output_equal':rows[0]['result']['tokens']==rows[1]['result']['tokens'],
        'production_defaults_snapshot_equal':rows[0]['hashes']==rows[1]['hashes']}
text=(R/'qwen-tp-checked.log').read_text(errors='replace')
kv=re.search(r'KV_CHECK TOTAL calls=(\d+) checks=(\d+) bytes=(\d+)',text)
views=re.search(r'VIEW_CHECK outputs=(\d+) hidden_rows=(\d+)',text)
checks['cache_and_view_validators_passed']=bool(kv and int(kv[2])>1 and views and int(views[1])>0)
t.write('qwen-production-profile.json',{'checks':checks,'rows':rows,'kv_check':kv.group() if kv else None,'views':views.group() if views else None})
print('QWEN PRODUCTION PROFILE',checks,flush=True)
assert all(checks.values()),checks
