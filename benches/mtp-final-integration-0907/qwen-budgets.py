#!/usr/bin/env python3
"""Second-architecture request-ceiling check; not a performance benchmark."""
import concurrent.futures as cf,importlib.util,json,pathlib,threading
R=pathlib.Path('/workspace/oai-qwen38-pp-lab/results/mtp-final-integration-0907')
z=importlib.util.spec_from_file_location('s',R.parent/'parallel-research-0907/sweep.py');s=importlib.util.module_from_spec(z);z.loader.exec_module(s)
s.PORT=32705;s.OUT=R;s.BIN=str(R/'candidate-bin/llama-server');s.ENV['LD_LIBRARY_PATH']=str(R/'candidate-bin')
for k in list(s.ENV):
 if k.startswith('LLAMA_EXPERIMENT_') or k.startswith('LLAMA_MTP_') or k=='LLAMA_KV_INDEXED_RM' or k.startswith('GGML_CUDA_VOLTA_Q8_MULTI') or k.startswith('GGML_CUDA_VOLTA_Q8_REF') or k in ['LD_PRELOAD','GGML_CUDA_VOLTA_Q8_COOP','GGML_CUDA_VOLTA_Q8_PV_F32']:s.ENV.pop(k)
# Exercise merged defaults, not research toggles.
def args(n):
 return [s.BIN,'-m','/models/Qwen3.8-27B/Qwen3.8-27B-UD-Q5_K_XL.gguf','--host','127.0.0.1','--port',str(s.PORT),'--ctx-size','32768','--parallel','4','--gpu-layers','all','--device','CUDA1,CUDA0','--tensor-split','14,35','--split-mode','layer','--fit','off','--flash-attn','on','--batch-size','512','--ubatch-size','256','--pipeline-copies','1','--cache-type-k','q8_0','--cache-type-v','q8_0','--cache-ram','0','--no-cache-idle-slots','--spec-type','draft-mtp','--spec-draft-n-max','3','--spec-draft-type-k','q8_0','--spec-draft-type-v','q8_0','--spec-draft-ubatch','128','--spec-mtp-defer-prompt','--slots','--perf','--no-warmup']
s.args=args;p,l=s.start(3,'qwen-budget')
try:
 text=pathlib.Path('common/sampling.cpp').read_text()[:16000]
 parent=s.req('/tokenize',{'content':'<|im_start|>system\nYou are a C++ reviewer.<|im_end|>\n<|im_start|>user\n'+text,'add_special':True})['tokens'][:2000]
 s.comp(0,parent,1)
 prompts=[]
 for i in range(4):
  tail=s.req('/tokenize',{'content':f'\nReview risk {i}: memory lifetime, concurrency and sampling. Give two tests.\n<|im_end|>\n<|im_start|>assistant\n<think>\n','add_special':False})['tokens']
  prompts.append(parent+tail)
 caps=[0,1,3,0];barrier=threading.Barrier(4)
 def work(i):
  barrier.wait()
  r=s.req('/completion',{'prompt':prompts[i],'id_slot':i,'n_predict':48,'return_tokens':True,'cache_prompt':True,'ignore_eos':True,'temperature':0.0,'seed':1234,'speculative_n_max':caps[i]})
  assert r['generation_settings']['speculative_n_max']==caps[i]
  assert len(r['tokens'])==48 and r['timings']['predicted_n']==48
  assert (r['timings'].get('draft_n',0)==0)==(caps[i]==0),r['timings']
  return r
 with cf.ThreadPoolExecutor(max_workers=4) as ex:rs=list(ex.map(work,range(4)))
 r=s.req('/completion',{'prompt':prompts[3]+rs[3]['tokens'],'id_slot':3,'n_predict':32,'return_tokens':True,'cache_prompt':True,'ignore_eos':True,'temperature':0.0,'seed':1234})
 assert r['generation_settings']['speculative_n_max']==3 and r['timings']['draft_n']>0 and r['timings']['prompt_n']<=2,r['timings']
 report={'caps':caps,'timings':[x['timings'] for x in rs],'default_resume':r['timings'],'all_checks_passed':True}
 (R/'qwen-summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
finally:s.stop(p,l)
