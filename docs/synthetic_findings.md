# Synthetic findings

The controlled corpus has 192 documents in 24 dimensions, with six query scenarios and 240 queries. A fixed partition supplies 144 training, 48 validation and 48 test queries. Three optimization seeds share the geometry and split. These small synthetic problems are diagnostic and do not establish real-world retrieval performance.

At four requested samples and 40 returned candidate slots:

| Model | Mean nDCG@10 | Candidate recall |
|---|---:|---:|
| Frozen dense | 0.9034 | 0.7638 |
| Deterministic adapter | 0.9920 | 0.7985 |
| Deterministic multi-vector | 0.9930 | 0.5927 |
| Gaussian, reward objective | 0.9977 | 0.5055 |
| Mixture, reward objective | 0.9976 | 0.5228 |

The near-ceiling nDCG values do not establish stochastic superiority. Relevant documents are concentrated in clusters, and a method can fill the top ten with relevant documents while covering fewer relevant documents across its candidate union. In this setup the single-vector deterministic adapter has substantially higher candidate recall than the sampled policies at the selected budget.

The sampling sweep, per-query cluster coverage, mixture weights, latent scales, and original-space sample cosine distances are saved with the results. The PCA views are descriptive projections only. The no-regularization run is a matched configuration ablation; it is not a new test-driven parameter search.

Surrogate and hybrid runs sometimes select identical checkpoints because validation favors the surrogate phase. The multi-vector surrogate uses max aggregation across heads, whereas stochastic surrogate losses average per-location losses. These objective differences limit representation-only conclusions.

Full tables: [regularized](../results/synthetic/summary.md) and [without regularization](../results/synthetic-no-kl/summary.md).
