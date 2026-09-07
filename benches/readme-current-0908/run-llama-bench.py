#!/usr/bin/env python3
import json, os, pathlib, subprocess, statistics
R=pathlib.Path(__file__).resolve().parent
MODEL='/models/Qwen3.8-27B/Qwen3.8-27B-UD-Q5_K_XL.gguf'
BINS={
 'upstream': pathlib.Path('/workspace/upstream-fair-0907/build/bin/llama-bench'),
 'fork': pathlib.Path('/workspace/llama-v100-sync-0907/build-sm70-75/bin/llama-bench'),
}
ARMS=['upstream','fork','fork','upstream']

def clean_env(bin):
    e=os.environ.copy(); e['LD_LIBRARY_PATH']=str(bin.parent)
    for k in list(e):
        if k.startswith(('GGML_CUDA_VOLTA_','GGML_CUDA_QWEN35_','QWEN4EXP_','LLAMA_EXPERIMENT_')) or k in ('GGML_CUDA_GRAPH_OPT','GGML_CUDA_TURING_CUBLAS_MIN_BATCH','GGML_CUDA_PREFETCH_WEIGHTS'):
            e.pop(k,None)
    return e

def run(gpu,label,arm):
    b=BINS[label]
    cmd=[str(b),'-m',MODEL,'-p','512,2048','-n','128','-r','3','-b','2048','-ub','512','-ctk','q8_0','-ctv','q8_0','-ngl','999','-sm','none','-mg','0','-fa','on','-dev',gpu,'-o','jsonl']
    p=subprocess.run(cmd,env=clean_env(b),text=True,capture_output=True,check=True,timeout=600)
    (R/f'llama-bench-{gpu}-{arm}-{label}.stdout').write_text(p.stdout)
    (R/f'llama-bench-{gpu}-{arm}-{label}.stderr').write_text(p.stderr)
    rows=[]
    for line in p.stdout.splitlines():
        try:x=json.loads(line)
        except:continue
        if isinstance(x,dict) and 'avg_ts' in x:
            x['test'] = f"pp{x.get('n_prompt')}" if x.get('n_prompt',0) else f"tg{x.get('n_gen')}"
            rows.append(x)
    print(gpu,arm,label,[(x.get('test'),x.get('avg_ts')) for x in rows],flush=True)
    return {'gpu':gpu,'arm':arm,'label':label,'rows':rows,'argv':cmd}

def main():
  allrows=[]
  for gpu in ['CUDA0','CUDA1']:
    for arm,label in enumerate(ARMS):
      x=run(gpu,label,arm);allrows.append(x);(R/'llama-bench-raw.json').write_text(json.dumps(allrows,indent=2))
  summary={}
  for gpu in ['CUDA0','CUDA1']:
    summary[gpu]={}
    for label in ['upstream','fork']:
      tests={}
      for a in allrows:
        if a['gpu']!=gpu or a['label']!=label:continue
        for r in a['rows']:
          tests.setdefault(r['test'],[]).append(r['avg_ts'])
      summary[gpu][label]={t:{'mean':statistics.mean(v),'values':v,'n_processes':len(v)} for t,v in tests.items()}
    for t in sorted(set(summary[gpu]['upstream']) & set(summary[gpu]['fork'])):
      u=summary[gpu]['upstream'][t]['mean'];f=summary[gpu]['fork'][t]['mean']
      summary[gpu].setdefault('change_pct',{})[t]=100*(f/u-1)
  (R/'llama-bench-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
  print('SUMMARY',json.dumps(summary,indent=2),flush=True)
main()
