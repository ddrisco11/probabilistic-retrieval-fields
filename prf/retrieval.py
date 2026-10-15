"""Exact cosine search with deterministic tie breaking and explicit budgets."""
import time
import numpy as np
import torch
from torch.nn import functional as F


def retrieve(points, documents, candidate_budget=40, final_k=10):
    if candidate_budget < len(points) or final_k < 1:
        raise ValueError("Budget must cover every search; final_k must be positive")
    start = time.perf_counter()
    scores = F.normalize(points, dim=-1) @ documents.T
    retrieval_start = time.perf_counter()
    candidates = []
    for j in range(len(points)):
        depth = min(len(documents), candidate_budget // len(points) + (j < candidate_budget % len(points)))
        candidates.append(torch.argsort(scores[j], descending=True, stable=True)[:depth])
    retrieval_end = time.perf_counter()
    union = torch.unique(torch.cat(candidates), sorted=True)
    # Same aggregation for every model: max cosine across all search locations.
    fused = scores[:, union].max(0).values
    ranking = union[torch.argsort(fused, descending=True, stable=True)][:final_k]
    end = time.perf_counter()
    return ranking.tolist(), union.tolist(), {
        "index_calls": len(points), "returned_slots": sum(map(len, candidates)),
        "unique_candidates": len(union), "scored_pairs": len(points)*len(documents),
        "similarity_ms": 1000*(retrieval_start-start),
        "selection_ms": 1000*(retrieval_end-retrieval_start),
        "dedup_rank_ms": 1000*(end-retrieval_end), "retrieval_ms": 1000*(end-start),
    }


def metrics(ranking, qrels, cutoff=10):
    ranking = list(dict.fromkeys(ranking))[:cutoff]
    positives = {int(d): float(r) for d, r in qrels.items() if r > 0}
    if not positives:
        raise ValueError("Evaluation query has no positive judgments")
    gains = [positives.get(d, 0) for d in ranking]
    dcg = sum((2**r-1)/np.log2(i+2) for i,r in enumerate(gains))
    ideal = sorted(positives.values(), reverse=True)[:cutoff]
    idcg = sum((2**r-1)/np.log2(i+2) for i,r in enumerate(ideal))
    return {f"ndcg@{cutoff}": float(dcg/idcg),
            f"recall@{cutoff}": sum(r > 0 for r in gains)/len(positives),
            f"mrr@{cutoff}": next((1/(i+1) for i,r in enumerate(gains) if r > 0), 0.0)}


def paired_bootstrap(a, b, seed=712, replicates=2000):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.ndim != 1 or len(a) == 0:
        raise ValueError("Paired nonempty 1D measurements required")
    delta = a-b
    rng = np.random.default_rng(seed)
    means = delta[rng.integers(0, len(delta), (replicates, len(delta)))].mean(1)
    return {"difference": float(delta.mean()), "ci95": np.quantile(means, [.025,.975]).tolist(),
            "unit": "held-out query; seed-averaged before bootstrap"}
