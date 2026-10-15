"""Combine immutable granularity runs and quantify training-region support."""
import json
from pathlib import Path
import numpy as np
import torch
from prf.region_report import report,aggregate

root=Path('results')
paths=[root/'wikiqa-regions']+[root/f'wikiqa-{k}-regions' for k in [1000,4096,8192]]
paths=[p for p in paths if (p/'results.json').exists()]
payloads=[json.loads((p/'results.json').read_text()) for p in paths]
combined=dict(payloads[0])
for key in ['results','partitions','comparisons']:
    combined[key]=[row for payload in payloads for row in payload[key]]
assert len({p['environment']['embedding_sha256'] for p in payloads})==1
assert len({json.dumps(json.loads((p/'splits.json').read_text()),sort_keys=True) for p in paths})==1
report(combined,Path('docs'))
Path('docs/explorer.html').replace('docs/index.html')
# The site report is generated from the original run files, not a new experiment.
Path('docs/summary.md').unlink()
data=torch.load('cache/wikiqa-bge.pt',weights_only=True)
rows=[]
for path,payload in zip(paths,payloads):
    for partition in payload['partitions']:
        k=partition['regions']
        labels=torch.load(path/f'partition-{k}.pt',weights_only=True)['labels']
        train_ids=[d for i in data['splits']['train'] for d in data['qrels'][i]]
        test_ids=[d for i in data['splits']['test'] for d in data['qrels'][i]]
        seen=set(labels[train_ids].tolist())
        rows.append({'regions':k,'mean_documents':len(labels)/k,'median_documents':float(np.median(partition['sizes'])),
                     'positive_training_regions':len(seen),'test_answers_in_unseen_regions':sum(int(labels[d]) not in seen for d in test_ids)/len(test_ids),
                     'partition_seconds':partition['construction_seconds'],
                     'run_wall_seconds_approx':(path/'results.json').stat().st_mtime-(path/'config.json').stat().st_mtime})
Path('docs/granularity-diagnostics.json').write_text(json.dumps(rows,indent=2)+'\n')
# Plot only fine-grained runs; previous figures stay tied to the original experiment.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
points=aggregate(combined);ks=[r['regions'] for r in rows if r['regions']>=1000]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,len(ks),figsize=(5*len(ks),4.4),squeeze=False,sharey=True)
for ax,k in zip(axes[0],ks):
    for method,color,label in [('centroid','#596a73','Nearest centroid'),('learned-top','#087f86','Learned top probability'),('learned-sample','#cb7133','Learned sampled')]:
        rs=sorted([r for r in points if r['regions']==k and r['method']==method],key=lambda r:r['probes'])
        ax.plot([100*r['corpus_fraction'] for r in rs],[100*r['region_hit'] for r in rs],'-o',color=color,label=label,markersize=4)
    ax.set_xscale('log');ax.set_xlabel('Corpus searched (%) · logarithmic scale');ax.set_title(f'{k:,} regions');ax.set_ylim(0,102);ax.grid(alpha=.2)
axes[0,0].set_ylabel('Questions reaching an answer region (%)')
handles,names=axes[0,0].get_legend_handles_labels();fig.legend(handles,names,loc='lower center',ncol=3,frameon=False)
fig.suptitle('Finer regions: routing accuracy versus search workload',fontsize=16)
fig.tight_layout(rect=[0,.13,1,.93]);fig.savefig('docs/assets/fine-granularity.png',dpi=180);fig.savefig('docs/assets/fine-granularity.svg')
p=Path('docs/assets/fine-granularity.svg');p.write_text('\n'.join(l.rstrip() for l in p.read_text().splitlines())+'\n')
print(json.dumps(rows,indent=2))
