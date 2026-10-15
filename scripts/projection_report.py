"""Publish measured continuous-projection results and inspect learned weights."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=json.loads(Path('results/wikiqa-projections/results.json').read_text())
names=list(dict.fromkeys(r['method'] for r in p['results']))
rows=[]
for name in names:
    group=[r for r in p['results'] if r['method']==name]
    keys=['expected_squared_distance','ndcg@10','recall@10','effective_heads','nonzero_heads','head_dispersion','maximum_weight']
    rows.append({'method':name,**{key:float(np.mean([r[key] for r in group])) for key in keys}})
dense={'method':'Frozen BGE',**p['dense_baseline']}
lines=['# Continuous weighted projections','','Frozen BGE-small / global WikiQA retrieval; three training seeds.','','Every head contributes its full assigned weight. Reward is negative squared distance between normalized destination and answer. Lower distance is better; higher nDCG is better.','', '| Method | Expected distance | nDCG@10 | Recall@10 | Effective heads | Nonzero heads | Head dispersion |','|---|---:|---:|---:|---:|---:|---:|']
for r in [dense]+rows:
    lines.append(f"| {r['method']} | {r['expected_squared_distance']:.4f} | {r['ndcg@10']:.4f} | {r['recall@10']:.4f} | {r['effective_heads']:.2f} | {r['nonzero_heads']:.2f} | {r['head_dispersion']:.5f} |")
lines+=['',f"Measured total experiment runtime: {p['wall_seconds']:.1f} seconds, excluding prior embedding creation and report generation.",'','Effective heads is 1 / sum(weight²); nonzero heads counts strictly positive learned weights. Dispersion is the expected squared distance between two independently drawn heads. Large effective count with near-zero dispersion means redundant destinations, not diverse hypotheses.','','See [protocol](../../docs/projections_protocol.md), [interactive weights](../../docs/projections.html), and results.json for per-query scores, weights, training curves and paired bootstrap intervals.']
Path('results/wikiqa-projections/summary.md').write_text('\n'.join(lines)+'\n')
fig,axes=plt.subplots(1,3,figsize=(13,4.5))
labels=['Frozen BGE','1 head','4 heads','8 heads','8 sparse heads']
for ax,key,title in zip(axes,['expected_squared_distance','ndcg@10','head_dispersion'],['Distance to relevant answers ↓','Retrieval nDCG@10 ↑','Destination dispersion']):
    ax.bar(range(5),[r[key] for r in [dense]+rows],color=['#596a73','#087f86','#20999c','#55b6b0','#cb7133'])
    ax.set_xticks(range(5),labels,rotation=25,ha='right');ax.set_title(title);ax.spines[['top','right']].set_visible(False)
fig.suptitle('Continuous projections · all probability mass retained',fontsize=15)
fig.tight_layout();fig.savefig('docs/assets/projections.png',dpi=180)
compact={'summary':[dense]+rows,'runs':[{'method':r['method'],'seed':r['seed'],'queries':[{'query_id':q['query_id'],'weights':q['weights'],'distance':q['expected_squared_distance'],'dispersion':q['head_dispersion']} for q in r['per_query']]} for r in p['results']]}
html='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Weighted query projections</title>
<style>body{font:17px system-ui;color:#193238;background:#f4f4ef;max-width:1050px;margin:auto;padding:40px 24px}h1{font-size:44px;letter-spacing:-.04em}p{line-height:1.6}select{font:inherit;padding:8px;margin:8px}table{border-collapse:collapse;width:100%}td,th{padding:12px;text-align:left;border-bottom:1px solid #bcc}.scroll{overflow:auto}img{width:100%}.bar{background:#087f86;height:24px;min-width:0}.weight{display:grid;grid-template-columns:70px 1fr 80px;gap:15px;margin:10px 0}a{color:#087f86}.note{border-left:3px solid #cb7133;padding-left:18px}</style>
<h1>Where should a query move?</h1><p>A small adapter produces continuous search destinations and assigns each a weight. Frozen BGE embeds both questions and answers. No fixed answer regions are used.</p>
<p class="note"><strong>All probability mass counts.</strong> Training minimizes Σ weight × squared distance to a labeled answer. Multiple positives are averaged. There is no cumulative-weight cutoff, best-head reward, or extra diversity objective.</p>
<img src="assets/projections.png" alt="Measured answer distance, retrieval quality and destination dispersion for frozen BGE and four projection models">
<h2>Measured results</h2><div class="scroll"><table><thead><tr><th>Model</th><th>Distance ↓</th><th>nDCG@10 ↑</th><th>Effective heads</th><th>Nonzero heads</th></tr></thead><tbody id="summary"></tbody></table></div>
<h2>Inspect the learned weights</h2><label>Model and seed <select id="run"></select></label><label>Held-out query <select id="query"></select></label><div id="weights"></div><p id="detail"></p>
<p>Softmax assigns positive weight to every head. Sparsemax can learn exact zeros. Effective head count measures how spread the weights are; it does not establish that destinations differ. Destination dispersion measures that distinction.</p>
<h2>Why multiple heads can collapse</h2><p>For squared distance, the weighted score equals distance to the weighted-average destination plus a dispersion term. For a fixed query, the dispersion term is constant across candidate answers. Ranking is therefore exactly equivalent to ranking by one weighted-average vector. A different nonlinear distance reward could preserve distinct modes while still using every weight, but is not part of this experiment.</p>
<p>Checkpoints use validation expected distance, not retrieval nDCG. Better distance does not necessarily improve ranking. These are exploratory results on 243 answerable test questions; earlier experiments used the same benchmark.</p>
<p><a href="https://github.com/ddrisco11/probabilistic-retrieval-fields/blob/main/docs/projections_protocol.md">Protocol and reproduction</a> · <a href="index.html">Earlier region experiment</a></p>
<script>const data=__DATA__;const $=id=>document.getElementById(id);for(const r of data.summary){let tr=document.createElement('tr');for(const v of [r.method,r.expected_squared_distance.toFixed(4),r['ndcg@10'].toFixed(4),r.effective_heads.toFixed(2),r.nonzero_heads.toFixed(2)]){let td=document.createElement('td');td.textContent=v;tr.append(td)}$('summary').append(tr)}
data.runs.forEach((r,i)=>$('run').add(new Option(r.method+' · seed '+r.seed,i)));function draw(){let q=data.runs[+$('run').value].queries[+$('query').value];$('weights').replaceChildren();q.weights.forEach((w,i)=>{let row=document.createElement('div');row.className='weight';let name=document.createElement('span');name.textContent='Head '+(i+1);let track=document.createElement('div');let bar=document.createElement('div');bar.className='bar';bar.style.width=(100*w)+'%';track.append(bar);let value=document.createElement('span');value.textContent=(100*w).toFixed(2)+'%';row.append(name,track,value);$('weights').append(row)});$('detail').textContent='All weights sum to '+q.weights.reduce((a,b)=>a+b,0).toFixed(6)+'. Expected distance: '+q.distance.toFixed(4)+'. Destination dispersion: '+q.dispersion.toFixed(6)+'.'}function chooseRun(){let prev=$('query').value;$('query').replaceChildren();data.runs[+$('run').value].queries.forEach((q,i)=>$('query').add(new Option(q.query_id,i)));if(prev)$('query').value=prev;draw()}$('run').onchange=chooseRun;$('query').onchange=draw;chooseRun();</script></html>'''
Path('docs/projections.html').write_text(html.replace('__DATA__',json.dumps(compact).replace('</','<\\/')))
print('\n'.join(lines))
