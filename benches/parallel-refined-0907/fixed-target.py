#!/usr/bin/env python3
import importlib.util,json,pathlib,subprocess,os
R=pathlib.Path('/workspace/oai-qwen38-pp-lab/results/parallel-refined-0907');G=R.parent/'group-pipeline-0907'
z=importlib.util.spec_from_file_location('fixture',R.parent/'parallel-research-0907/sweep.py')
s=importlib.util.module_from_spec(z);z.loader.exec_module(s)
variants=[('plain',R/'final-bin',{},1),
 ('q8-refined',R/'final-bin',{'GGML_CUDA_VOLTA_Q8_MULTI':'1','GGML_CUDA_VOLTA_Q8_REFINED':'1','GGML_CUDA_VOLTA_Q8_REFINED_AUTO':'1','GGML_CUDA_VOLTA_Q8_COOP':'1'},1),
 ('group-control',G/'candidate-bin',{},1),
 ('group2',G/'candidate-bin',{'LLAMA_EXPERIMENT_VERIFY_GROUP_SEQS':'2'},2),
 ('group1',G/'candidate-bin',{'LLAMA_EXPERIMENT_VERIFY_GROUP_SEQS':'1'},2),
 ('plain-repeat',R/'final-bin',{},1)]
results=[]
for name,B,flags,copies in variants:
 env=s.ENV.copy()
 for k in list(env):
  if k.startswith('LLAMA_EXPERIMENT_') or k.startswith('GGML_CUDA_VOLTA_Q8_MULTI') or k.startswith('GGML_CUDA_VOLTA_Q8_REFINED') or k=='GGML_CUDA_VOLTA_Q8_COOP':env.pop(k)
 env.update(flags);env.update({'LD_LIBRARY_PATH':str(B),'VERIFY_FIXTURE':str(R),'VERIFY_OUTPUT':str(R/f'logits-{name}.bin')})
 args=s.args(0);args[0]=str(R/'verify-target');args[args.index('--pipeline-copies')+1]=str(copies)
 args+=['--spec-type','draft-mtp','--spec-draft-n-max','3']
 with (R/f'fixed-{name}.log').open('wb') as out:
  p=subprocess.run(args,env=env,stdout=out,stderr=subprocess.STDOUT,timeout=120)
 row={'name':name,'returncode':p.returncode,'flags':flags,'copies':copies}
 results.append(row);(R/'fixed-results.json').write_text(json.dumps(results,indent=2))
 print('FIXED',name,p.returncode,(R/f'fixed-{name}.log').read_text(errors='replace')[-500:],flush=True)
 if p.returncode!=0:break
