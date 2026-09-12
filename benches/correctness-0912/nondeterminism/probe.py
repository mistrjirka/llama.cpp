#!/usr/bin/env python3
"""Repeat exact cold/restored requests; preserve tokens, sampling and pre-sampling probabilities."""
from pathlib import Path
import argparse, hashlib, json, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import validate_streamk_e2e as v
from benchmark_streamk_regression import PROFILES
v.ROOT = Path('/models/.bench-ornith-mtp4/nondeterminism-0912')
v.ROOT.mkdir(exist_ok=True)
FIXED = Path('/models/.bench-ornith-mtp4/int8-streamk-fixed-src/build-matched-force/bin/llama-server')
UPSTREAM = Path('/workspace/llama-upstream-head-0911/build-sm70-75-force-mmq/bin/llama-server')
CONTROLS = {
 'base': {}, 'graphs-off': {'GGML_CUDA_DISABLE_GRAPHS':'1'},
 'fusion-off': {'GGML_CUDA_DISABLE_FUSION':'1'},
 'int8-off': {'GGML_CUDA_TURING_INT8_QK':'0'},
 'blocking': {'CUDA_LAUNCH_BLOCKING':'1'},
 'upstream': {},
}
def run(case, mode, port, reps, profile_name, binary=None, count=64):
 cfg=PROFILES[profile_name]
 binary=binary or (UPSTREAM if case=='upstream' else FIXED)
 label=f'det-{profile_name}-{mode}-{case}-n{count}' + ('-'+binary.parent.parent.name if binary != FIXED and binary != UPSTREAM else '')
 s=v.Server(binary,label,port,cfg['state'])
 s.args=[str(s.binary),'--model',cfg['model'],'--host','127.0.0.1','--port',str(port),
 '--ctx-size',str(cfg['ctx']),'--parallel','1','--fit','off','--gpu-layers','all',
 '--flash-attn','on','--batch-size',str(cfg['batch']),'--ubatch-size',str(cfg['ubatch']),
 '--cache-type-k','q8_0','--cache-type-v','q8_0','--cache-ram','0','--ctx-checkpoints','0',
 '--slot-save-path',str(cfg['state']),'--no-warmup','--perf','--skip-chat-parsing',
 '--pipeline-copies','1','--reasoning','off','--reasoning-format','none']+cfg['placement']
 if case=='upstream':
  j=s.args.index('--pipeline-copies');del s.args[j:j+2]
 s.env['CUDA_VISIBLE_DEVICES']=cfg['cuda'];s.env.update(CONTROLS[case])
 s.manifest.update(argv=s.args, env={k:val for k,val in s.env.items() if k.startswith(('GGML_', 'CUDA_', 'LLAMA_'))})
 tokens=json.loads(cfg['seq'].read_text())
 tokens=tokens[:cfg['prefix']+1000] if mode=='cached' else tokens[:1024]
 if mode=='cached': assert v.saved_tokens(cfg['state']/cfg['filename'])==tokens[:cfg['prefix']]
 rows=[]
 with s:
  for i in range(reps):
   if mode=='cached':s.restore(0,cfg['filename'],cfg['prefix'])
   z=s.request('/completion',{'prompt':tokens,'id_slot':0,'cache_prompt':mode=='cached',
    'n_predict':count,'ignore_eos':True,'temperature':0,'seed':1234,'return_tokens':True,
    'n_probs':5,'post_sampling_probs':False})
   assert z['timings']['cache_n']==(cfg['prefix'] if mode=='cached' else 0),z['timings']
   assert z['generation_settings']['temperature']==0,z['generation_settings']
   out=z['tokens'];assert len(out)==count,len(out)
   h=hashlib.sha256(json.dumps(out).encode()).hexdigest()
   prev=rows[0]['response']['tokens'] if rows else out
   first=next((j for j,(a,b) in enumerate(zip(prev,out)) if a!=b),None)
   rows.append({'rep':i,'hash':h,'first_diff':first,'response':z})
   print(label,i,h[:12],'diff',first,'pp',round(z['timings']['prompt_per_second'],2),'tg',round(z['timings']['predicted_per_second'],2),'first',out[:8],flush=True)
 report={'label':label,'unique_hashes':len({r['hash'] for r in rows}),'rows':rows,'manifest':s.manifest}
 (v.ROOT/(label+'.json')).write_text(json.dumps(report,indent=2)+'\n')
 return report
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--cases',nargs='+',default=['base','graphs-off','fusion-off','upstream']);p.add_argument('--modes',nargs='+',default=['cached','cold']);p.add_argument('--profile',default='ornith-rtx');p.add_argument('--port',type=int,default=41400);p.add_argument('--reps',type=int,default=3);p.add_argument('--binary',type=Path);p.add_argument('--count',type=int,default=64)
 a=p.parse_args();i=0
 for mode in a.modes:
  for case in a.cases:
   r=run(case,mode,a.port+i,a.reps,a.profile,a.binary,a.count);i+=1
   print('SUMMARY',r['label'],r['unique_hashes'],flush=True)
