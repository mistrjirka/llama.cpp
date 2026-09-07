#!/usr/bin/env python3
import hashlib,importlib.util,json,pathlib
import os
R=pathlib.Path(os.environ.get('REFINED_RESULTS_DIR','/workspace/oai-qwen38-pp-lab/results/parallel-refined-0907'))
O=R.parent/'parallel-research-0907'
z=importlib.util.spec_from_file_location('s',pathlib.Path(__file__).resolve().parent/'fixture.py');s=importlib.util.module_from_spec(z);z.loader.exec_module(s)
s.PORT=32520;s.OUT=R;s.BIN=str(R/'baseline-bin/llama-server');s.ENV['LD_LIBRARY_PATH']=str(R/'baseline-bin')
for k in list(s.ENV):
 if k.startswith('LLAMA_EXPERIMENT_') or k.startswith('GGML_CUDA_VOLTA_Q8_MULTI'):s.ENV.pop(k)
p,log=s.start(3,'prepare')
try:
 repo=pathlib.Path('/workspace/llama-parallel-all-0907')
 paths=sorted(list((repo/'tools/server').glob('*.cpp'))+list((repo/'src').glob('*.cpp'))+list((repo/'common').glob('*.cpp')))
 text='<|im_start|>system\nYou are a C++ code reviewer. Use the supplied repository source. Identify concrete issues, cite identifiers, and propose executable tests.\n<|im_end|>\n<|im_start|>user\nRepository source follows:\n'
 provenance={}
 for f in paths:
  data=f.read_text();text+='\nFILE '+str(f.relative_to(repo))+'\n'+data
  provenance[str(f.relative_to(repo))]=hashlib.sha256(data.encode()).hexdigest()
  if len(text)>1200000:break
 tokens=s.req('/tokenize',{'content':text,'add_special':True})['tokens']
 assert len(tokens)>110000,len(tokens)
 parent=tokens[:98000]
 tasks=[
  'Audit sequence-state restoration. Find two concrete correctness risks and propose regression tests.',
  'Audit the handling of CUDA synchronization and asynchronous output buffers. Explain a safe optimization and its hazards.',
  'Audit speculative decoding rollback with recurrent state. Explain what must survive a model swap and how to test it.',
  'Audit multi-agent scheduling and prefix caching. Identify a measurable bottleneck and design a controlled experiment.'
 ]
 histories=[];prompts=[]
 def tok(t):return s.req('/tokenize',{'content':t,'add_special':False})['tokens']
 for i,task in enumerate(tasks):
  h=parent+tok('\nAgent review topic '+str(i)+': '+task+'\nAdditional source:\n')
  h+=tokens[98000+2100*i:98000+2100*i+(100000-len(h))]
  assert len(h)==100000
  suffix=tok('\nTask: '+task+'\nDo not repeat the source. Give a precise analysis.\n<|im_end|>\n<|im_start|>assistant\n<think>\n')
  histories.append(h);prompts.append(h+suffix)
 (R/'real-fixture.json').write_text(json.dumps({'parent':parent,'histories':histories,'prompts':prompts,'sources':provenance,'corpus':'actual repository C++ source; distinct review instructions; raw ChatML'}))
 s.comp(0,parent,0)
 for i in [1,2,3,0]:
  r=s.comp(i,histories[i],0)
  assert r['timings']['cache_n']==98000,r['timings']
  print('prepared',i,r['timings']['cache_n'],r['timings']['prompt_n'],flush=True)
 for i in range(4):s.req(f'/slots/{i}?action=save',{'filename':f'real100k-{i}.bin'})
 print('saved four real source histories',flush=True)
finally:s.stop(p,log)
