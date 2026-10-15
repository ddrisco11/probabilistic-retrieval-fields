# Continuous weighted projections

Frozen BGE-small / global WikiQA retrieval; three training seeds.

Every head contributes its full assigned weight. Reward is negative squared distance between normalized destination and answer. Lower distance is better; higher nDCG is better.

| Method | Expected distance | nDCG@10 | Recall@10 | Effective heads | Nonzero heads | Head dispersion |
|---|---:|---:|---:|---:|---:|---:|
| Frozen BGE | 0.5442 | 0.7281 | 0.8748 | 1.00 | 1.00 | 0.00000 |
| 1-head-softmax | 0.4198 | 0.6820 | 0.8025 | 1.00 | 1.00 | 0.00000 |
| 4-head-softmax | 0.4198 | 0.6852 | 0.8059 | 2.76 | 4.00 | 0.00089 |
| 8-head-softmax | 0.4199 | 0.6833 | 0.8026 | 5.92 | 8.00 | 0.00071 |
| 8-head-sparsemax | 0.4195 | 0.6832 | 0.8008 | 3.88 | 4.03 | 0.00017 |

Measured total experiment runtime: 48.5 seconds, excluding prior embedding creation and report generation.

Effective heads is 1 / sum(weight²); nonzero heads counts strictly positive learned weights. Dispersion is the expected squared distance between two independently drawn heads. Large effective count with near-zero dispersion means redundant destinations, not diverse hypotheses.

See [protocol](../../docs/projections_protocol.md), [interactive weights](../../docs/projections.html), and results.json for per-query scores, weights, training curves and paired bootstrap intervals.
