#!/usr/bin/env python3
"""Publication plots from exact Nsight intervals. Matplotlib's default cycle only."""
import json,pathlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=pathlib.Path('/workspace/oai-qwen38-pp-lab/results/mtp-gantt-0907')
for arm in ['baseline','patched']:
 E=json.loads((R/(arm+'-mtp3x4-events.json')).read_text())
 draft=sorted([x for x in E if x['lane']=='Draft orchestration'],key=lambda x:x['s'])
 lo=draft[4]['s'];hi=draft[7]['s']
 rows=['Draft orchestration','Prompt copies','KV tail removal','Target readback/sampling','Draft readback/sampling','Cache refresh','CUDA synchronization','Tesla V100-SXM2-32GB','NVIDIA GeForce RTX 2080 Ti']
 labels=['Draft generation (CPU scope)','Copy whole prompt','Remove draft tail','Target sampling (CPU)','Draft sampling / wait (CPU)','Refresh draft KV (CPU scope)','CUDA synchronization (CPU)','V100 — GPU kernels / copies','RTX 2080 Ti — GPU kernels / copies']
 fig,ax=plt.subplots(figsize=(16,6.8))
 for kind,title in [('target','Target verification'),('draft','Draft model'),('refresh','KV refresh'),('copy','GPU transfers'),('cpu','CPU work or waits'),('wait','CUDA wait calls'),('host-other','Other launches')]:
  es=[x for x in E if x['kind']==kind and x['lane'] in rows and x['e']>lo and x['s']<hi]
  if not es:continue
  ys=[len(rows)-rows.index(x['lane'])-1 for x in es];ls=[max(x['s'],lo)-lo for x in es];ws=[min(x['e'],hi)-max(x['s'],lo) for x in es]
  ax.barh(ys,ws,left=ls,height=.58,label=title,linewidth=0)
 ax.set_yticks(list(reversed(range(len(rows)))),labels,fontsize=11)
 ax.set_xlim(0,hi-lo);ax.set_xlabel('Elapsed milliseconds — three measured draft/verify rounds',fontsize=11)
 ax.set_title(('Before' if arm=='baseline' else 'After indexed tail removal and skipping unused prompt copies')+'\nFour Ornith agents, MTP3, 100k context each, Q8 target/draft KV',loc='left',fontsize=15,pad=15)
 ax.legend(loc='upper left',bbox_to_anchor=(0,-.14),ncol=3,frameon=False,fontsize=10)
 ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
 for spine in ['top','right']:ax.spines[spine].set_visible(False)
 fig.text(.012,.012,'Actual Nsight timestamps; empty GPU lanes indicate no recorded GPU work. CPU scopes can contain waits. Profiling adds overhead.',fontsize=9)
 fig.tight_layout(rect=(0,.065,1,1))
 fig.savefig(R/(arm+'-gantt.png'),dpi=150);fig.savefig(R/(arm+'-gantt.svg'));plt.close(fig)
 print(arm,lo,hi,flush=True)
