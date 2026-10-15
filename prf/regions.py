"""Relevance-trained routing over fixed spherical k-means answer regions."""
import copy
import hashlib
import json
import platform
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .retrieval import metrics, paired_bootstrap


def partition(documents,k,seed=2026,iterations=40):
    if not 1<=k<=len(documents):raise ValueError('Invalid region count')
    g=torch.Generator().manual_seed(seed)
    # Seeded spherical k-means++; corpus only, never question labels.
    centers=[documents[torch.randint(len(documents),(1,),generator=g).item()]]
    for _ in range(1,k):
        distances=(1-documents@torch.stack(centers).T).min(-1).values.clamp_min(0).square()
        if distances.sum()==0:raise ValueError('Not enough distinct directions')
        centers.append(documents[torch.multinomial(distances,1,generator=g).item()])
    centers=torch.stack(centers)
    previous=None
    for _ in range(iterations):
        labels=(documents@centers.T).argmax(-1)
        if previous is not None and torch.equal(labels,previous):break
        previous=labels.clone()
        means=torch.zeros_like(centers)
        means.index_add_(0,labels,documents)
        empty=means.norm(dim=-1)<1e-10
        means[empty]=centers[empty]
        centers=F.normalize(means,dim=-1)
    labels=(documents@centers.T).argmax(-1)
    return centers,labels


def region_targets(qrels,labels,k):
    target=torch.zeros(len(qrels),k)
    for i,rel in enumerate(qrels):
        if rel:
            target[i]=torch.bincount(labels[list(rel)],minlength=k).float()/len(rel)
    return target


class RegionRouter(nn.Module):
    def __init__(self,centers,hidden=64):
        super().__init__()
        self.register_buffer('centers',centers.clone())
        self.net=nn.Sequential(nn.Linear(centers.shape[1],hidden),nn.Tanh(),nn.Linear(hidden,len(centers)))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
        # Fixed nearest-centroid residual prior prevents discarding useful geometry.
        self.log_temperature=nn.Parameter(torch.tensor(np.log(.1),dtype=torch.float32))

    def forward(self,q):
        return (q@self.centers.T)/self.log_temperature.exp().clamp(.01,2)+self.net(q)


def fit_router(data,centers,targets,config,seed):
    torch.manual_seed(seed)
    model=RegionRouter(centers,config['hidden'])
    opt=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
    train_ids=torch.tensor(data['splits']['train']);valid=data['splits']['validation']
    best=float('inf');best_state=None;history=[];started=time.perf_counter()
    for epoch in range(config['epochs']):
        model.train();losses=[]
        for batch in train_ids[torch.randperm(len(train_ids))].split(config['batch_size']):
            loss=-(targets[batch]*model(data['queries'][batch]).log_softmax(-1)).sum(-1).mean()
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            val=float(-(targets[valid]*model(data['queries'][valid]).log_softmax(-1)).sum(-1).mean())
        history.append({'epoch':epoch,'train_cross_entropy':float(np.mean(losses)),'validation_cross_entropy':val})
        if val<best:best=val;best_state=copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model,history,time.perf_counter()-started


def select_regions(probabilities,probes,mode,generator=None):
    probes=min(probes,len(probabilities))
    if probes<1:raise ValueError('probes must be positive')
    if mode=='sample':
        # Sampling without replacement guarantees every probe visits a distinct cell.
        return torch.multinomial(probabilities,probes,replacement=False,generator=generator)
    return torch.argsort(probabilities,descending=True,stable=True)[:probes]


def route_metrics(probabilities,target,selected):
    coverage=float(target[selected].sum())
    return {'region_hit':float(coverage>0),'answer_region_coverage':coverage,
            'region_cross_entropy':float(-(target*probabilities.clamp_min(1e-12).log()).sum()),
            'region_brier':float((probabilities-target).square().sum()),
            'assigned_probability_to_relevant_regions':float(probabilities[target>0].sum())}


def evaluate_router(data,centers,labels,target,model,method,probes,seed):
    groups=[torch.where(labels==j)[0] for j in range(len(centers))]
    sizes=torch.bincount(labels,minlength=len(centers)).float()
    g=torch.Generator().manual_seed(seed+10000)
    rows=[]
    with torch.no_grad():
        for i in data['splits']['test']:
            q=data['queries'][i]
            start=time.perf_counter()
            if method.startswith('learned'):p=model(q[None,:]).softmax(-1)[0]
            elif method=='centroid':p=(q@centers.T/.1).softmax(-1)
            elif method=='largest-region':p=sizes/sizes.sum()
            else:p=torch.ones(len(centers))/len(centers)
            selection=select_regions(p,probes,'sample' if method in {'learned-sample','random'} else 'top',g)
            routing_end=time.perf_counter()
            # Membership lists are precomputed; only selected documents are scored.
            candidates=torch.cat([groups[j] for j in selection.tolist()]).sort().values
            scores=data['documents'][candidates]@q
            order=torch.argsort(scores,descending=True,stable=True)[:10]
            ranking=candidates[order].tolist()
            end=time.perf_counter()
            rows.append({'query_id':data['query_ids'][i],**route_metrics(p,target[i],selection),
                         **metrics(ranking,data['qrels'][i]),'documents_scored':len(candidates),
                         'corpus_fraction':len(candidates)/len(labels),'region_probes':len(selection),
                         'routing_ms':1000*(routing_end-start),'total_ms':1000*(end-start),
                         'selected_regions':selection.tolist(),'ranking':[data['document_ids'][j] for j in ranking]})
    keys=[k for k,v in rows[0].items() if isinstance(v,(float,int))]
    summary={k:float(np.mean([r[k] for r in rows])) for k in keys}
    summary.update(p50_ms=float(np.median([r['total_ms'] for r in rows])),
                   p95_ms=float(np.quantile([r['total_ms'] for r in rows],.95)))
    return summary,rows


def run_regions(data,config,output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    (out/'config.json').write_text(json.dumps(config,indent=2))
    (out/'splits.json').write_text(json.dumps({s:[data['query_ids'][i] for i in ids] for s,ids in data['splits'].items()},indent=2))
    results=[];partitions=[];histories={}
    # Full-corpus baseline is independent of partition and computed once.
    from .models import Policy
    from .experiment import evaluate
    dense,dense_rows=evaluate(Policy(data['documents'].shape[1],'dense'),data,data['splits']['test'],1,len(data['documents']))
    for k in config['regions']:
        t=time.perf_counter();centers,labels=partition(data['documents'],k,config['partition_seed'])
        elapsed=time.perf_counter()-t
        targets=region_targets(data['qrels'],labels,k)
        sizes=torch.bincount(labels,minlength=k)
        torch.save({'centers':centers,'labels':labels},out/f'partition-{k}.pt')
        partitions.append({'regions':k,'construction_seconds':elapsed,'sizes':sizes.tolist(),
                           'empty_cells':int((sizes==0).sum()),
                           'assignment_sha256':hashlib.sha256(labels.numpy().tobytes()).hexdigest()})
        for seed in config['seeds']:
            model,history,seconds=fit_router(data,centers,targets,config,seed)
            torch.save(model.state_dict(),out/f'router-{k}-seed{seed}.pt')
            histories[f'{k}/{seed}']=history
            for method in ['centroid','largest-region','random','learned-top','learned-sample']:
                for probes in config['probes']:
                    if probes>k:continue
                    agg,rows=evaluate_router(data,centers,labels,targets,model,method,probes,seed)
                    results.append({'method':method,'regions':k,'probes':probes,'seed':seed,
                                    'training_seconds':seconds if method.startswith('learned') else 0,
                                    'trainable_parameters':sum(p.numel() for p in model.parameters()) if method.startswith('learned') else 0,
                                    **agg,'per_query':rows})
            print(f'regions={k} seed={seed} completed',flush=True)
    comparisons=[]
    for k in config['regions']:
        for probes in config['probes']:
            if probes>k:continue
            for metric in ['region_hit','answer_region_coverage','ndcg@10']:
                a=[r for r in results if r['regions']==k and r['probes']==probes and r['method']=='learned-top']
                b=[r for r in results if r['regions']==k and r['probes']==probes and r['method']=='centroid']
                av=np.mean([[q[metric] for q in r['per_query']] for r in a],axis=0)
                bv=np.mean([[q[metric] for q in r['per_query']] for r in b],axis=0)
                comparisons.append({'regions':k,'probes':probes,'metric':metric,**paired_bootstrap(av,bv)})
    payload={'results':results,'partitions':partitions,'histories':histories,'comparisons':comparisons,
             'dense_baseline':dense,'dense_per_query':dense_rows,'provenance':data['metadata'],
             'environment':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,
                            'platform':platform.platform(),'device':'cpu','threads':1,
                            'embedding_sha256':hashlib.sha256(data['documents'].numpy().tobytes()+data['queries'].numpy().tobytes()).hexdigest()}}
    from .storage import save_region_results
    save_region_results(payload,out)
    from .region_report import report
    report(payload,out)
    return payload

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--config',default='configs/wikiqa-regions.json');p.add_argument('--output',required=True)
    a=p.parse_args();run_regions(torch.load(a.data,weights_only=True),json.loads(Path(a.config).read_text()),a.output)
