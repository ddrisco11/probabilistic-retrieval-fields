"""Continuous weighted projections with a literal expected-distance objective."""
import argparse
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


def sparsemax(logits):
    """Simplex projection: learned exact zeros, without a cumulative-mass cutoff."""
    values = logits - logits.max(-1, keepdim=True).values
    ordered = values.sort(dim=-1, descending=True).values
    ranks = torch.arange(1, values.shape[-1]+1, device=values.device, dtype=values.dtype)
    cumulative = ordered.cumsum(-1)
    support = 1 + ranks * ordered > cumulative
    count = support.sum(-1, keepdim=True)
    threshold = (cumulative.gather(-1, count-1)-1)/count
    return (values-threshold).clamp_min(0)


class WeightedProjections(nn.Module):
    def __init__(self, dim, heads=8, hidden=64, gate='softmax'):
        super().__init__()
        self.dim, self.heads, self.gate = dim, heads, gate
        self.body = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh())
        self.offset = nn.Linear(hidden, heads*dim)
        self.weight = nn.Linear(hidden, heads)
        nn.init.normal_(self.offset.weight, std=.001)
        nn.init.normal_(self.offset.bias, std=.001)
        nn.init.zeros_(self.weight.weight)
        nn.init.zeros_(self.weight.bias)

    def forward(self, queries):
        h = self.body(queries)
        # Offset implicitly contains direction and magnitude; no fixed regions.
        destinations = F.normalize(queries[:, None, :] + self.offset(h).view(-1, self.heads, self.dim), dim=-1)
        logits = self.weight(h)
        weights = sparsemax(logits) if self.gate == 'sparsemax' else logits.softmax(-1)
        return destinations, weights


def expected_distance(points, weights, answers, mask):
    """Mean over labeled positives, weighted sum over ALL projection heads."""
    distances = (points[:, :, None, :] - answers[:, None, :, :]).square().sum(-1)
    per_head = (distances*mask[:, None, :]).sum(-1)/mask.sum(-1, keepdim=True)
    return (weights*per_head).sum(-1)


def expected_scores(points, weights, documents):
    """Negative expected squared distance, including all assigned mass.

    Algebraic reduction is exact, not a search approximation or mass truncation.
    """
    mean = (weights[..., None]*points).sum(-2)
    norm_term = (weights*points.square().sum(-1)).sum(-1, keepdim=True)
    return 2*mean@documents.T - norm_term - documents.square().sum(-1)


def answer_batch(data, ids):
    rels = [list(data['qrels'][int(i)]) for i in ids]
    width = max(map(len, rels))
    answers = torch.zeros(len(ids), width, data['documents'].shape[-1])
    mask = torch.zeros(len(ids), width)
    for row, docs in enumerate(rels):
        answers[row, :len(docs)] = data['documents'][docs]
        mask[row, :len(docs)] = 1
    return answers, mask


def fit(data, config, heads, gate, seed):
    torch.manual_seed(seed)
    model = WeightedProjections(data['queries'].shape[-1], heads, config['hidden'], gate)
    opt = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=0)
    train = torch.tensor(data['splits']['train'])
    valid = data['splits']['validation']
    va, vm = answer_batch(data, valid)
    history = []; best = float('inf'); state = None; best_epoch = None
    start = time.perf_counter()
    for epoch in range(config['epochs']+1):
        losses = []
        if epoch:
            model.train()
            for ids in train[torch.randperm(len(train))].split(config['batch_size']):
                a, mask = answer_batch(data, ids)
                points, weights = model(data['queries'][ids])
                loss = expected_distance(points, weights, a, mask).mean()
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                opt.step(); losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            points, weights = model(data['queries'][valid])
            val = float(expected_distance(points, weights, va, vm).mean())
        history.append({'epoch': epoch, 'train_distance': float(np.mean(losses)) if losses else None,
                        'validation_distance': val})
        if val < best:
            best = val; state = copy.deepcopy(model.state_dict()); best_epoch = epoch
    model.load_state_dict(state)
    return model, history, best_epoch, time.perf_counter()-start


def evaluate(data, model=None):
    rows = []
    with torch.no_grad():
        for i in data['splits']['test']:
            q = data['queries'][i:i+1]
            start = time.perf_counter()
            points, weights = model(q) if model is not None else (q[:, None, :], torch.ones(1, 1))
            scores = expected_scores(points, weights, data['documents'])[0]
            ranking = scores.argsort(descending=True, stable=True)[:10].tolist()
            elapsed = time.perf_counter()-start
            a, mask = answer_batch(data, [i])
            distance = float(expected_distance(points, weights, a, mask)[0])
            w = weights[0]; z = points[0]
            # Expected pair distance between two independently drawn heads.
            dispersion = float((w[:, None]*w[None, :]*(z[:, None, :]-z[None, :, :]).square().sum(-1)).sum())
            rows.append({'query_id': data['query_ids'][i], **metrics(ranking, data['qrels'][i]),
                         'expected_squared_distance': distance, 'reward': -distance,
                         'effective_heads': float(1/w.square().sum()), 'nonzero_heads': int((w>0).sum()),
                         'maximum_weight': float(w.max()), 'head_dispersion': dispersion,
                         'weighted_query_movement': float((w*(z-q[0]).square().sum(-1)).sum()),
                         'total_ms': elapsed*1000, 'weights': w.tolist(),
                         'ranking': [data['document_ids'][j] for j in ranking]})
    keys = [k for k,v in rows[0].items() if isinstance(v, (int,float))]
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}, rows


def run(data, config, output):
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    start = time.perf_counter()
    (out/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    (out/'splits.json').write_text(json.dumps({s:[data['query_ids'][i] for i in ids] for s,ids in data['splits'].items()}, indent=2)+'\n')
    dense, dense_rows = evaluate(data)
    results = []; histories = {}
    for setting in config['models']:
        heads, gate = setting['heads'], setting['gate']
        name = f'{heads}-head-{gate}'
        for seed in config['seeds']:
            model, history, epoch, seconds = fit(data, config, heads, gate, seed)
            summary, rows = evaluate(data, model)
            torch.save(model.state_dict(), out/f'{name}-{seed}.pt')
            histories[f'{name}/{seed}'] = history
            results.append({'method': name, 'seed': seed, 'heads': heads, 'gate': gate,
                            'best_epoch': epoch, 'training_seconds': seconds, **summary, 'per_query': rows})
            print(f'{name} seed={seed} epoch={epoch} distance={summary["expected_squared_distance"]:.4f} ndcg={summary["ndcg@10"]:.4f}', flush=True)
    comparisons = []
    names = list(dict.fromkeys(r['method'] for r in results))
    for name in names:
        selected = [r for r in results if r['method']==name]
        single = [r for r in results if r['method']=='1-head-softmax']
        for metric in ['expected_squared_distance','ndcg@10','recall@10']:
            av = np.mean([[q[metric] for q in r['per_query']] for r in selected], axis=0)
            for reference, bv in [('dense', [q[metric] for q in dense_rows]), ('1-head-softmax', np.mean([[q[metric] for q in r['per_query']] for r in single], axis=0))]:
                comparisons.append({'method': name, 'reference': reference, 'metric': metric, **paired_bootstrap(av,bv)})
    payload = {'results': results, 'dense_baseline': dense, 'dense_per_query': dense_rows,
               'histories': histories, 'comparisons': comparisons, 'provenance': data['metadata'],
               'wall_seconds': time.perf_counter()-start,
               'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                               'platform': platform.platform(), 'threads': 1, 'device': 'cpu',
                               'embedding_sha256': hashlib.sha256(data['documents'].numpy().tobytes()+data['queries'].numpy().tobytes()).hexdigest()}}
    (out/'results.json').write_text(json.dumps(payload, indent=2)+'\n')
    return payload


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--data', required=True); p.add_argument('--config', required=True); p.add_argument('--output', required=True)
    args = p.parse_args()
    run(torch.load(args.data, weights_only=True), json.loads(Path(args.config).read_text()), args.output)
