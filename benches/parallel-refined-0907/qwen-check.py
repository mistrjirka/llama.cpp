#!/usr/bin/env python3
import concurrent.futures as cf,importlib.util,json,pathlib,threading
import os
R=pathlib.Path(os.environ.get('REFINED_RESULTS_DIR','/workspace/oai-qwen38-pp-lab/results/parallel-refined-0907'));O=R.parent/'parallel-research-0907'
z=importlib.util.spec_from_file_location('s',pathlib.Path(__file__).resolve().parent/'fixture.py');s=importlib.util.module_from_spec(z);z.loader.exec_module(s)
s.PORT=32524;s.OUT=R;s.BIN=str(R/'host-bin/llama-server');s.ENV['LD_LIBRARY_PATH']=str(R/'host-bin')
s.ENV.update({'LLAMA_EXPERIMENT_SAMPLING_VIEW':'1','LLAMA_EXPERIMENT_MTP_BULK_HIDDEN':'1','LD_PRELOAD':str(R/'verify_views.so')})
def args(n):
 return [s.BIN,'-m','/models/Qwen3.8-27B/Qwen3.8-27B-UD-Q5_K_XL.gguf','--host','127.0.0.1','--port',str(s.PORT),'--ctx-size','32768','--parallel','4','--gpu-layers','all','--device','CUDA1,CUDA0','--tensor-split','14,35','--split-mode','layer','--fit','off','--flash-attn','on','--batch-size','512','--ubatch-size','256','--pipeline-copies','1','--cache-type-k','q8_0','--cache-type-v','q8_0','--cache-ram','0','--no-cache-idle-slots','--spec-type','draft-mtp','--spec-draft-n-max','3','--spec-draft-type-k','f16','--spec-draft-type-v','f16','--spec-draft-ubatch','128','--spec-mtp-defer-prompt','--slots','--perf','--no-warmup']
s.args=args;p,l=s.start(3,'qwen-views-check')
try:
 code=pathlib.Path('common/sampling.cpp').read_text()[:16000]
 text='<|im_start|>system\nYou are a C++ reviewer.<|im_end|>\n<|im_start|>user\n'+code
 parent=s.req('/tokenize',{'content':text,'add_special':True})['tokens'][:2000]
 s.comp(0,parent,0)
 qs=[]
 for i in range(4):
  end=s.req('/tokenize',{'content':f'\nReview risk {i}: memory lifetime, concurrency, and sampling. Give two tests.\n<|im_end|>\n<|im_start|>assistant\n<think>\n','add_special':False})['tokens']
  qs.append(parent+end)
 b=threading.Barrier(4)
 with cf.ThreadPoolExecutor(max_workers=4) as ex:rs=list(ex.map(lambda i:s.comp(i,qs[i],48,b),range(4)))
 for r in rs:assert r['timings']['predicted_n']==48,r
 (R/'qwen-views-check.json').write_text(json.dumps(rs,indent=2));print('Qwen four-slot MTP completed with view validation',flush=True)
finally:s.stop(p,l)
