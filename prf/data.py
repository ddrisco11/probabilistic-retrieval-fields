import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

SCENARIOS = ["single", "two", "several", "overlap", "ambiguous", "imbalanced"]


def synthetic(seed=2026, dim=24, queries=240, per_cluster=24):
    if dim < 8 or queries < 30:
        raise ValueError("Synthetic generator requires dim>=8 and queries>=30")
    g = torch.Generator().manual_seed(seed)
    centers = F.normalize(torch.randn(8, dim, generator=g), dim=-1)
    # Harder partially overlapping clusters.
    centers[7] = F.normalize(centers[6] + .2*centers[7], dim=-1)
    groups = torch.arange(8).repeat_interleave(per_cluster)
    docs = F.normalize(centers[groups] + .10*torch.randn(len(groups), dim, generator=g), dim=-1)
    rotation, _ = torch.linalg.qr(torch.randn(dim, dim, generator=g))
    qs, rels, types = [], [], []
    for i in range(queries):
        scenario = SCENARIOS[i % len(SCENARIOS)]
        a = int(torch.randint(0, 6, (1,), generator=g))
        selected = {"single":[a], "two":[a,(a+2)%6], "several":[a,(a+2)%6,(a+4)%6],
                    "overlap":[6], "ambiguous":[a,(a+3)%6], "imbalanced":[a,(a+1)%6]}[scenario]
        weights = torch.tensor([.85,.15]) if scenario == "imbalanced" else torch.ones(len(selected))/len(selected)
        signal = (centers[selected]*weights[:,None]).sum(0)
        if scenario == "overlap":
            signal = (centers[6]+centers[7])/2
        if scenario == "ambiguous":
            signal = centers[a] # second region cannot be read off as a simple average
        q = signal + .45*(signal @ rotation) + .08*torch.randn(dim, generator=g)
        qs.append(q)
        relevant = torch.where(torch.isin(groups, torch.tensor(selected)))[0].tolist()
        if scenario == "imbalanced":
            relevant = torch.where(groups == selected[0])[0].tolist() + torch.where(groups == selected[1])[0][:3].tolist()
        rels.append({d:1 for d in relevant})
        types.append(scenario)
    # Stratified fixed partition, shared by every optimization seed.
    split = {"train":[], "validation":[], "test":[]}
    for scenario in SCENARIOS:
        ids = torch.tensor([i for i,t in enumerate(types) if t == scenario])
        ids = ids[torch.randperm(len(ids), generator=g)].tolist()
        n = len(ids)
        split["train"] += ids[:int(n*.6)]
        split["validation"] += ids[int(n*.6):int(n*.8)]
        split["test"] += ids[int(n*.8):]
    return {"documents":docs, "queries":F.normalize(torch.stack(qs),dim=-1),
            "qrels":rels, "splits":split, "types":types, "groups":groups,
            "query_ids":[str(i) for i in range(queries)], "document_ids":[str(i) for i in range(len(docs))],
            "metadata":{"kind":"synthetic", "seed":seed, "dimension":dim}}


def read_beir(folder):
    folder = Path(folder)
    def lines(name):
        return [json.loads(x) for x in (folder/name).read_text().splitlines() if x.strip()]
    corpus, queries = lines("corpus.jsonl"), lines("queries.jsonl")
    if len({x['_id'] for x in corpus}) != len(corpus) or len({x['_id'] for x in queries}) != len(queries):
        raise ValueError("Duplicate identifiers")
    docmap = {x['_id']:i for i,x in enumerate(corpus)}
    qmap = {x['_id']:i for i,x in enumerate(queries)}
    qrels = [{} for _ in queries]
    splits = {}
    for name in ("train", "test"):
        ids = set()
        with (folder/"qrels"/f"{name}.tsv").open() as f:
            for row in csv.DictReader(f, delimiter='\t'):
                q, d, score = qmap[row['query-id']], docmap[row['corpus-id']], float(row['score'])
                if score > 0:
                    ids.add(q)
                    qrels[q][d] = score
        splits[name] = sorted(ids)
    if set(splits['train']) & set(splits['test']):
        raise ValueError("Train/test query leakage")
    if len(splits['train']) < 5 or not splits['test']:
        raise ValueError("Insufficient positive train/test judgments")
    # Hash ordering is stable independent of source order and training random seed.
    ordered = sorted(splits['train'], key=lambda i:hashlib.sha256(('prf-2026:'+queries[i]['_id']).encode()).hexdigest())
    n = max(1, round(.2*len(ordered)))
    splits['validation'], splits['train'] = ordered[:n], ordered[n:]
    return corpus, queries, qrels, splits


def encode_beir(folder, output, model_name, revision):
    from sentence_transformers import SentenceTransformer
    from huggingface_hub import model_info
    import time
    corpus, queries, qrels, splits = read_beir(folder)
    sha = model_info(model_name, revision=revision).sha
    source_hashes = {str(p.relative_to(folder)):hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted(Path(folder).rglob('*')) if p.is_file() and p.suffix in {'.jsonl','.tsv'}}
    started = time.perf_counter()
    model = SentenceTransformer(model_name, revision=sha, device='cpu')
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    docs = model.encode([(x.get('title','')+' '+x['text']).strip() for x in corpus],
                        normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=True)
    prefix = 'Represent this sentence for searching relevant passages: ' if model_name == 'BAAI/bge-small-en-v1.5' else ''
    qs = model.encode([prefix+x['text'] for x in queries], normalize_embeddings=True,
                      convert_to_tensor=True, show_progress_bar=True)
    bundle = {"documents":docs.cpu(), "queries":qs.cpu(), "qrels":qrels,"splits":splits,
              "query_ids":[x['_id'] for x in queries],"document_ids":[x['_id'] for x in corpus],
              "types":["benchmark"]*len(queries),"groups":None,
              "metadata":{"kind":"beir", "model":model_name,"revision":sha,"source_sha256":source_hashes,
                          "encoding_seconds":time.perf_counter()-started,"normalize":True,"query_prefix":prefix,
                          "max_seq_length":model.max_seq_length,"device":"cpu"}}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, output)
    Path(str(output)+'.json').write_text(json.dumps(bundle['metadata'],indent=2))
    return bundle
