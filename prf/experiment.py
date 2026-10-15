import copy
import hashlib
import json
import platform
import resource
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .models import Policy
from .retrieval import retrieve, metrics, paired_bootstrap


def evaluate(model, data, ids, samples, budget, seed=910, geometry=False):
    rows = []
    model.eval()
    with torch.no_grad(), torch.random.fork_rng():
        torch.manual_seed(seed)
        # Warm up without consuming the measured random sequence.
        with torch.random.fork_rng():
            p = model.locations(data['queries'][ids[:1]], samples)[0]
            retrieve(p, data['documents'], budget)
        for i in ids:
            t = time.perf_counter()
            points = model.locations(data['queries'][i:i+1], samples)[0]
            inference_ms = 1000*(time.perf_counter()-t)
            rank, candidates, cost = retrieve(points, data['documents'], budget)
            rel = data['qrels'][i]
            row = {"query_id":data['query_ids'][i],"type":data['types'][i],
                   **metrics(rank, rel), **cost,"model_ms":inference_ms,
                   "total_ms":inference_ms+cost['retrieval_ms'],
                   "candidate_recall":len(set(candidates)&set(rel))/len(rel),
                   "ranking":[data['document_ids'][j] for j in rank]}
            if data['groups'] is not None:
                relevant_groups = set(data['groups'][list(rel)].tolist())
                hits = set(candidates)&set(rel)
                covered = set(data['groups'][list(hits)].tolist())
                row['relevant_cluster_coverage'] = len(covered)/len(relevant_groups)
            rows.append(row)
    keys = [k for k,v in rows[0].items() if isinstance(v,(int,float))]
    aggregate = {k:float(np.mean([r[k] for r in rows])) for k in keys}
    aggregate.update({"p50_ms":float(np.median([r['total_ms'] for r in rows])),
                      "p95_ms":float(np.quantile([r['total_ms'] for r in rows],.95))})
    return aggregate, rows


def relevance_surrogate(model, q, docs, targets, samples, temperature):
    points, weights = model.surrogate_points(q, samples)
    logits = F.normalize(points,dim=-1) @ docs.T / temperature
    if model.kind == 'multi':
        # Document-wise maximum lets heads specialize. This is a stronger distinct
        # objective than averaging independent per-vector losses; report it explicitly.
        return -(logits.squeeze(2).max(1).values.log_softmax(-1)*targets).sum(-1).mean()
    losses = -(logits.log_softmax(-1)*targets[:,None,None,:]).sum(-1).mean(-1)
    return (losses*weights).sum(-1).mean() if weights is not None else losses.mean()


def reward_loss(model, q, docs, positive, samples, depth=10):
    if samples < 2:
        raise ValueError('RLOO requires at least two samples')
    dist = model.distribution(q)
    actions = dist.sample((samples,)).detach()
    with torch.no_grad():
        scores = F.normalize(actions,dim=-1) @ docs.T
        indices = torch.argsort(scores,dim=-1,descending=True,stable=True)[...,:depth]
        rewards = positive[None,:,:].expand(samples,-1,-1).gather(-1,indices).sum(-1) / positive.sum(-1)[None,:]
        baseline = (rewards.sum(0,keepdim=True)-rewards)/(samples-1)
        advantage = rewards-baseline
    return -(advantage*dist.log_prob(actions)).mean(), rewards


def train(model, data, config, seed):
    if model.kind in {'dense','perturb'}:
        return [], 0.0
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=0)
    ids = data['splits']['train']
    docs = data['documents']
    positives = torch.zeros(len(data['queries']),len(docs))
    for i in ids:
        positives[i,list(data['qrels'][i])] = 1
    targets = positives / positives.sum(-1,keepdim=True).clamp_min(1)
    history, best, best_state = [], -float('inf'), None
    started = time.perf_counter()
    for epoch in range(config['epochs']):
        model.train()
        order = torch.tensor(ids)[torch.randperm(len(ids))]
        losses, rewards_all = [], []
        strategy = config['strategy']
        use_rl = model.stochastic and (strategy == 'reward' or (strategy == 'hybrid' and epoch >= config['epochs']//2))
        for batch in order.split(config['batch_size']):
            q = data['queries'][batch]
            if use_rl:
                loss, rewards = reward_loss(model,q,docs,positives[batch],config['train_samples'])
                rewards_all.extend(rewards.flatten().tolist())
            else:
                loss = relevance_surrogate(model,q,docs,targets[batch],config['train_samples'],config['temperature'])
            loss = loss + config['regularization']*model.regularizer(q)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            opt.step()
            losses.append(float(loss.detach()))
        # Fixed RNG validation, and fork_rng in evaluate preserves training RNG.
        score,_ = evaluate(model,data,data['splits']['validation'],config['selection_samples'],config['candidate_budget'])
        row = {'epoch':epoch,'loss':float(np.mean(losses)),'validation_ndcg':score['ndcg@10'],
               'objective':'reward' if use_rl else 'surrogate'}
        if rewards_all:
            row.update(reward_mean=float(np.mean(rewards_all)),reward_variance=float(np.var(rewards_all)))
        history.append(row)
        if score['ndcg@10'] > best:
            best, best_state = score['ndcg@10'], copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return history, time.perf_counter()-started


def geometry_snapshot(model,data,query_index=0,seed=80):
    with torch.no_grad(), torch.random.fork_rng():
        torch.manual_seed(seed)
        q = data['queries'][query_index:query_index+1]
        samples = F.normalize(model.locations(q,128)[0],dim=-1)
        docs = data['documents']
        # Projection learned from documents alone, shared across methods.
        center = docs.mean(0)
        _,_,v = torch.linalg.svd(docs-center,full_matrices=False)
        project = lambda x: ((x-center)@v[:2].T).tolist()
        nearest = (samples@docs.T).argmax(-1)
        cosine = samples@samples.T
        diversity = float((1-cosine)[~torch.eye(len(samples),dtype=torch.bool)].mean()) if len(samples)>1 else 0.
        diag = {'sample_cosine_distance':diversity,'distinct_top1_documents':len(nearest.unique())}
        if model.stochastic:
            mu,std,w = model.parameters_for(q)
            diag.update(mean_latent_std=float(std.mean()),min_latent_std=float(std.min()),
                        mixture_weights=w[0].tolist(),effective_components=float(torch.exp(-(w*w.log()).sum())))
        if data['groups'] is not None:
            diag['nearest_cluster_occupancy'] = torch.bincount(data['groups'][nearest],minlength=8).tolist()
        return {'documents':project(docs),'samples':project(samples),'query':project(q)[0],
                'relevant':list(data['qrels'][query_index]),'query_id':data['query_ids'][query_index],
                'diagnostics':diag}


def run(data, config, output):
    out = Path(output)
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(config.get('threads',1))
    torch.use_deterministic_algorithms(True)
    fingerprint = hashlib.sha256(data['documents'].numpy().tobytes()+data['queries'].numpy().tobytes()).hexdigest()
    (out/'config.json').write_text(json.dumps(config,indent=2))
    (out/'splits.json').write_text(json.dumps({k:[data['query_ids'][i] for i in v] for k,v in data['splits'].items()},indent=2))
    results, geometries = [], {}
    for seed in config['seeds']:
        for kind in config['models']:
            strategies = config['strategies'] if kind in {'gaussian','mixture'} else ['surrogate']
            for strategy in strategies:
                torch.manual_seed(seed)
                model = Policy(data['documents'].shape[1],kind,config['hidden'],config['components'],config['prior_std'])
                cfg = dict(config,strategy=strategy)
                history, elapsed = train(model,data,cfg,seed)
                label = kind + ('-'+strategy if kind in {'gaussian','mixture'} else '')
                prefix = f'{label}-seed{seed}'
                torch.save(model.state_dict(),out/f'{prefix}.pt')
                (out/f'{prefix}-history.json').write_text(json.dumps(history,indent=2))
                for samples in config['sample_budgets']:
                    agg,rows = evaluate(model,data,data['splits']['test'],samples,config['candidate_budget'],seed=seed+10000)
                    result = {'model':label,'seed':seed,'requested_samples':samples,
                              'trainable_parameters':sum(p.numel() for p in model.parameters()),
                              'training_seconds_including_validation':elapsed,
                              'checkpoint_bytes':(out/f'{prefix}.pt').stat().st_size,**agg,'per_query':rows}
                    results.append(result)
                if seed == config['seeds'][0]:
                    # Preselected first test query in each scenario; no cherry-picked geometry.
                    for typ in sorted(set(data['types'][i] for i in data['splits']['test'])):
                        idx = next(i for i in data['splits']['test'] if data['types'][i]==typ)
                        geometries[label+'/'+typ] = geometry_snapshot(model,data,idx)
                print(prefix, 'nDCG@10',round(results[-1]['ndcg@10'],4),flush=True)
    comparisons = []
    for samples in config['sample_budgets']:
        base = [r for r in results if r['model']=='deterministic' and r['requested_samples']==samples]
        if not base:
            continue
        base_values = np.mean([[q['ndcg@10'] for q in r['per_query']] for r in base],axis=0)
        for label in sorted({r['model'] for r in results}):
            chosen = [r for r in results if r['model']==label and r['requested_samples']==samples]
            values = np.mean([[q['ndcg@10'] for q in r['per_query']] for r in chosen],axis=0)
            comparisons.append({'model':label,'requested_samples':samples,'reference':'deterministic',
                                **paired_bootstrap(values,base_values)})
    payload = {'results':results,'comparisons':comparisons,'geometry':geometries,
               'provenance':{'data':data['metadata'],'embedding_sha256':fingerprint,'python':sys.version,
                             'torch':torch.__version__,'numpy':np.__version__,'platform':platform.platform(),
                             'device':'cpu','threads':torch.get_num_threads(),
                             'process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
                             'memory_note':'Process high-water RSS across all runs; not per-model memory.'}}
    (out/'results.json').write_text(json.dumps(payload,indent=2))
    from .report import write_report
    write_report(payload,out)
    return payload
