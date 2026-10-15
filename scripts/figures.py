"""Create publication figures from recorded region-routing runs."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from prf.region_report import aggregate

p=Path('results/wikiqa-regions/results.json')
data=json.loads(p.read_text());rows=aggregate(data)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                     'axes.spines.right':False,'figure.facecolor':'#f4f4ef','axes.facecolor':'#f4f4ef'})
colors={'centroid':'#596a73','largest-region':'#9b825d','random':'#9696a1','learned-top':'#087f86','learned-sample':'#cb7133'}
labels={'centroid':'Nearest centroid','largest-region':'Largest regions','random':'Random','learned-top':'Learned: top probability','learned-sample':'Learned: sampled'}
for metric,title,filename in [('region_hit','At least one answer in selected regions','routing-quality'),('ndcg@10','Final retrieval quality','retrieval-quality')]:
    fig,axes=plt.subplots(1,3,figsize=(13,4.2),sharey=True)
    for ax,k in zip(axes,[8,32,128]):
        for method in colors:
            rs=sorted([r for r in rows if r['regions']==k and r['method']==method],key=lambda r:r['probes'])
            ax.plot([100*r['corpus_fraction'] for r in rs],[r[metric] for r in rs],'-o',markersize=4,label=labels[method],color=colors[method])
        if metric=='ndcg@10':ax.axhline(data['dense_baseline']['ndcg@10'],color='#193238',ls=':',lw=1,label='Full-corpus BGE' if k==8 else None)
        ax.set_title(f'{k} regions');ax.set_xlabel('Corpus searched (%)');ax.set_xlim(0,102);ax.grid(alpha=.17)
    axes[0].set_ylabel('Answer-region hit rate' if metric=='region_hit' else 'nDCG@10')
    fig.suptitle(title,fontsize=17,x=.06,ha='left');handles,names=axes[0].get_legend_handles_labels()
    fig.legend(handles,names,loc='lower center',ncol=3,frameon=False)
    fig.tight_layout(rect=[0,.15,1,.91]);fig.savefig('docs/assets/'+filename+'.png',dpi=180);fig.savefig('docs/assets/'+filename+'.svg');plt.close(fig)
# Architecture is a schematic, explicitly not measured region geometry.
fig,ax=plt.subplots(figsize=(12,3));ax.axis('off')
boxes=[('Question','Frozen BGE-small'),('384-D query vector','Small region router'),('Probability over K cells','Choose a search budget'),('Selected answer regions','Cosine rank within cells')]
for i,(a,b) in enumerate(boxes):
    x=.02+i*.25
    ax.text(x,.7,a,transform=ax.transAxes,fontsize=12,color='#193238')
    ax.text(x,.44,b,transform=ax.transAxes,fontsize=10,color='#49656a',bbox=dict(facecolor='#dfe9e4',edgecolor='none',pad=9))
    if i<3:ax.annotate('',xy=(x+.235,.52),xytext=(x+.205,.52),xycoords='axes fraction',arrowprops=dict(arrowstyle='->',color='#087f86',lw=2))
ax.text(.02,.09,'The embedding model and answer vectors stay fixed. Only the router learns from relevance labels.',transform=ax.transAxes,fontsize=12)
fig.tight_layout();fig.savefig('docs/assets/architecture.svg');fig.savefig('docs/assets/architecture.png',dpi=180)

for path in Path('docs/assets').glob('*.svg'):
    path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
