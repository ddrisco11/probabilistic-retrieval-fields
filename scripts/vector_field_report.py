"""Static scientific figures and a standalone rotatable 3D field viewer."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path('results/wikiqa-vector-field')
p=json.loads((root/'results.json').read_text());pts=json.loads((root/'points.json').read_text())
s=p['statistics']; colors={'question':'#cb7133','answer':'#087f86'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
for dim in [2,3]:
    fig=plt.figure(figsize=(13,5.5))
    for panel,view in enumerate(['endpoints','displacements'],1):
        ax=fig.add_subplot(1,2,panel,projection='3d' if dim==3 else None)
        for kind in colors:
            group=[v for v in pts if v['kind']==kind]
            a=np.array([v[view]['start'] for v in group]);b=np.array([v[view]['end'] for v in group]);d=b-a
            if dim==2:
                ax.quiver(a[:,0],a[:,1],d[:,0],d[:,1],angles='xy',scale_units='xy',scale=1,color=colors[kind],alpha=.45,width=.0025)
                ax.scatter(a[:,0],a[:,1],s=4,c=colors[kind],alpha=.5,label=kind+'s')
                ax.set_aspect('equal',adjustable='box')
                allpoints=np.vstack([a,b]);ax.update_datalim(allpoints[:,:2]);ax.autoscale_view()
            else:
                ax.quiver(a[:,0],a[:,1],a[:,2],d[:,0],d[:,1],d[:,2],color=colors[kind],alpha=.4,arrow_length_ratio=.2,linewidth=.6)
                ax.scatter(a[:,0],a[:,1],a[:,2],s=3,c=colors[kind],label=kind+'s');ax.set_zlabel('PC3')
        if dim==3:
            allpoints=np.array([point[view][end] for point in pts for end in ['start','end']]);lo=allpoints.min(0);hi=allpoints.max(0);pad=(hi-lo)*.05
            ax.set_xlim(lo[0]-pad[0],hi[0]+pad[0]);ax.set_ylim(lo[1]-pad[1],hi[1]+pad[1]);ax.set_zlim(lo[2]-pad[2],hi[2]+pad[2])
        ax.set_xlabel('PC1');ax.set_ylabel('PC2');ax.set_title('PCA of pooled endpoints' if view=='endpoints' else 'PCA of displacement vectors');ax.legend(loc='upper left',fontsize=8)
    fig.suptitle(f'Small → aligned base · {dim}D PCA · {len(pts)} held-out texts',fontsize=15)
    fig.tight_layout(rect=[0,0,1,.92]);fig.savefig(f'docs/assets/vector-field-{dim}d.png',dpi=180,bbox_inches='tight',pad_inches=.25);plt.close(fig)
fig,ax=plt.subplots(figsize=(7,4))
for key,label in [('endpoint_variance_ratio','Pooled endpoints'),('displacement_variance_ratio','Displacements')]:
    ax.plot(np.arange(1,385),np.cumsum(p['spectra'][key])*100,label=label)
ax.set(xlabel='Number of principal components',ylabel='Training variance retained (%)',xlim=(1,384),ylim=(0,100));ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig('docs/assets/vector-field-spectrum.png',dpi=180);plt.close(fig)
percent=lambda v:f'{100*v:.1f}%'
lines=['# BGE-small to BGE-base vector field','','Each arrow connects the same text across encoders after a train-fitted alignment. Arrows are observed model differences, not learned query-to-answer movements.','',
       '| Measurement | Value |','|---|---:|',
       f"| Base centered energy retained by 768→384 alignment | {percent(s['base_centered_energy_retained_384d'])} |",
       f"| Held-out displacement RMS / shuffled-alignment control | {s['rms_displacement']:.4f} / {s['shuffled_alignment_test_rms']:.4f} |",
       f"| Endpoint PCA training variance: 2D / 3D | {percent(s['endpoint_pca_training_variance']['2'])} / {percent(s['endpoint_pca_training_variance']['3'])} |",
       f"| Endpoint PCA held-out arrow energy: 2D / 3D | {percent(s['endpoint_pca_test_displacement_energy']['2'])} / {percent(s['endpoint_pca_test_displacement_energy']['3'])} |",
       f"| Displacement PCA held-out centered arrow energy: 2D / 3D | {percent(s['displacement_pca_test_centered_energy']['2'])} / {percent(s['displacement_pca_test_centered_energy']['3'])} |",
       f"| Global drift share of held-out arrow energy | {percent(s['global_drift_energy_fraction'])} |",
       f"| Centered neighbor / random direction cosine | {s['locality']['centered_neighbor_direction_cosine']:.4f} / {s['locality']['centered_random_direction_cosine']:.4f} |",
       f"| 10-neighbor displacement MSE improvement over global mean | {percent(s['locality']['knn_mse_improvement_over_mean'])} |",'']
for kind,v in p['groups'].items():
    lines.append(f"{kind.capitalize()}: {v['count']:,} held-out texts, displacement RMS {v['rms_displacement']:.4f}; within-type neighbor/random direction cosine {v['locality']['neighbor_direction_cosine']:.4f}/{v['locality']['random_direction_cosine']:.4f}; neighbor prediction MSE improvement {percent(v['locality']['knn_mse_improvement_over_mean'])}.")
lines+=['','Native full-corpus answer retrieval (243 answerable test questions):','', '| Encoder | nDCG@10 | Recall@10 |','|---|---:|---:|']
for k,v in p['retrieval'].items():lines.append(f"| {k} | {v['ndcg@10']:.4f} | {v['recall@10']:.4f} |")
lines+=['',f"Base-minus-small nDCG difference: {p['base_minus_small_ndcg']['difference']:.4f}, paired 95% interval {p['base_minus_small_ndcg']['ci95']}.",'','[Protocol](../../docs/vector_field_protocol.md) · [Interactive 2D/3D viewer](https://ddrisco11.github.io/probabilistic-retrieval-fields/vector-field.html)']
(root/'summary.md').write_text('\n'.join(lines)+'\n')
html='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Small → base embedding field</title><style>body{font:17px system-ui;color:#193238;background:#f4f4ef;max-width:1100px;margin:auto;padding:35px 24px}h1{font-size:44px;letter-spacing:-.04em}p{line-height:1.6}select,input{font:inherit;margin:8px}canvas{width:100%;height:570px;background:#fff;border:1px solid #ccd6d2}label{display:inline-block}a{color:#087f86}.note{padding-left:18px;border-left:3px solid #cb7133}.stats{display:flex;flex-wrap:wrap;gap:25px}.stats strong{display:block;font-size:25px}img{width:100%}</style>
<h1>How does embedding geometry change?</h1><p>Each arrow starts at a frozen BGE-small embedding and ends at the aligned BGE-base embedding of <strong>the same text</strong>. Questions are orange; answer sentences are teal. All plotted texts are held out from alignment and PCA fitting.</p>
<p class="note">Small has 384 dimensions; base has 768. A training-only orthogonal alignment compresses base into small's coordinates. These arrows depend on that alignment. They are not raw movements between naturally shared axes, and the base encoder is still required to observe their endpoints.</p>
<div class="stats"><div><strong>__ENERGY__</strong>base centered energy retained</div><div><strong>__PCA__</strong>arrow energy visible in spatial 3D</div><div><strong>__KNN__</strong>neighbor prediction MSE improvement</div></div>
<p>Dots mark small-model origins; arrowheads mark aligned-base destinations. Arrow lengths are shown at their actual projected scale. PCA axes are fixed from training data; rotations only change the viewing angle.</p>
<label>View <select id="basis"><option value="endpoints">Spatial PCA (pooled endpoints)</option><option value="displacements">Movement PCA (displacements)</option></select></label>
<label>Dimensions <select id="dim"><option>3</option><option>2</option></select></label>
<label>Texts <select id="kind"><option value="all">Questions and answers</option><option value="question">Questions</option><option value="answer">Answers</option></select></label><br>
<label>Rotate <input id="yaw" type="range" min="-180" max="180" value="25"></label><label>Tilt <input id="pitch" type="range" min="-80" max="80" value="20"></label>
<canvas id="field" aria-label="Projected small-to-base displacement arrows"></canvas><p id="caption"></p>
<h2>How much does PCA hide?</h2><img src="assets/vector-field-spectrum.png" alt="Cumulative training variance retained by endpoint and displacement PCA"><p>A low-dimensional field is a partial view. Movement PCA emphasizes displacement variation; spatial PCA emphasizes the overall endpoint cloud. The statistical analysis uses all held-out points in the aligned 384-dimensional space; this viewer shows a fixed sample of 500.</p>
<p><a href="https://github.com/ddrisco11/probabilistic-retrieval-fields/blob/main/results/wikiqa-vector-field/summary.md">Measured statistics</a> · <a href="https://github.com/ddrisco11/probabilistic-retrieval-fields/blob/main/docs/vector_field_protocol.md">Method and limitations</a> · <a href="index.html">Region experiments</a></p>
<script>const points=__POINTS__;const $=id=>document.getElementById(id);function draw(){let basis=$('basis').value,dim=+$('dim').value,kind=$('kind').value,yaw=+$('yaw').value*Math.PI/180,pitch=+$('pitch').value*Math.PI/180;const project=p=>{if(dim===2)return [p[0],p[1]];let a=p[0]*Math.cos(yaw)-p[2]*Math.sin(yaw),b=p[0]*Math.sin(yaw)+p[2]*Math.cos(yaw);return [a,p[1]*Math.cos(pitch)-b*Math.sin(pitch)]};let rows=points.filter(p=>kind==='all'||p.kind===kind).map(p=>({a:project(p[basis].start),b:project(p[basis].end),kind:p.kind}));let all=points.flatMap(p=>[project(p[basis].start),project(p[basis].end)]),xmin=Math.min(...all.map(p=>p[0])),xmax=Math.max(...all.map(p=>p[0])),ymin=Math.min(...all.map(p=>p[1])),ymax=Math.max(...all.map(p=>p[1]));let c=$('field'),w=c.clientWidth,h=570,dpr=devicePixelRatio||1;c.width=w*dpr;c.height=h*dpr;let ctx=c.getContext('2d');ctx.scale(dpr,dpr);let scale=Math.min((w-70)/(xmax-xmin),(h-80)/(ymax-ymin));const screen=p=>[w/2+(p[0]-(xmin+xmax)/2)*scale,h/2-(p[1]-(ymin+ymax)/2)*scale];for(let r of rows){let a=screen(r.a),b=screen(r.b),angle=Math.atan2(b[1]-a[1],b[0]-a[0]);ctx.strokeStyle=ctx.fillStyle=r.kind==='question'?'#cb7133':'#087f86';ctx.globalAlpha=.55;ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke();ctx.beginPath();ctx.arc(...a,1.6,0,7);ctx.fill();ctx.beginPath();ctx.moveTo(...b);ctx.lineTo(b[0]-4*Math.cos(angle-.5),b[1]-4*Math.sin(angle-.5));ctx.lineTo(b[0]-4*Math.cos(angle+.5),b[1]-4*Math.sin(angle+.5));ctx.closePath();ctx.fill()}ctx.globalAlpha=1;ctx.fillStyle='#193238';ctx.font='14px system-ui';ctx.fillText(dim===2?'PC1 → / PC2 ↑':'Orthographic view of PC1, PC2 and PC3',18,25);$('caption').textContent=rows.length+' held-out arrows displayed. Orange: questions. Teal: answer sentences. '+(basis==='endpoints'?'Axes capture endpoint variation.':'Axes capture movement variation.');$('yaw').disabled=$('pitch').disabled=dim===2}for(let id of ['basis','dim','kind','yaw','pitch'])$(id).oninput=draw;window.onresize=draw;draw();</script></html>'''
html=html.replace('__POINTS__',json.dumps(pts)).replace('__ENERGY__',percent(s['base_centered_energy_retained_384d'])).replace('__PCA__',percent(s['endpoint_pca_test_displacement_energy']['3'])).replace('__KNN__',percent(s['locality']['knn_mse_improvement_over_mean']))
Path('docs/vector-field.html').write_text(html)
print('\n'.join(lines))
