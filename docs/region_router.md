# Central hypothesis: probabilistic region routing

Domain-specific retrieval often requires aligning queries with relevant documents. This project tests a small router that maps an unchanged query embedding to probabilities over regions of a fixed answer/document space. Region granularity is an explicit accuracy/cost control.

This is a distinct experiment from sampling arbitrary Gaussian locations. Both are evaluated; region routing is the primary benchmark experiment.

## Target

Partition the frozen answer corpus into disjoint cells R_1,...,R_K using document-only spherical k-means. Each answer has a deterministic region assignment. Given query embedding q, predict softmax probabilities p(R_j|q). Train on the empirical distribution of relevant answer IDs across cells. A query with relevant answers in three cells contributes probability mass to all three.

The target means: if one labeled relevant answer is drawn uniformly, which cell contains it? It does not mean the independent probability that each cell contains any answer. A multi-label sigmoid model would represent that second quantity.

## Granularity and routing

A Cartesian grid with b bins on each of d axes requires b^d cells. Learned K-way partitions avoid that explicit explosion, although choice of K, cluster imbalance and high-dimensional geometry still matter. Compare K in {8,32,128}, top regions in {1,2,4,8,16}, and sampled regions against deterministic top-probability routing. The region router can output probabilities without requiring random inference.

Partitioning the corpus is unsupervised index construction and may include all searchable answer texts. Router fitting and checkpoint selection use train/validation question labels only. The initial implementation trains a router per encoder/partition/task. It does not establish a universal pretrained router transferable between unrelated corpora; region IDs change when partitions change.

## Evaluation

Report top-1 region hit, top-k any-answer region hit, relevant-answer region coverage (fraction of relevant answers in selected cells), regional probability cross entropy and Brier score, fraction and count of corpus documents examined, nDCG@10, Recall@10, MRR@10 and measured latency. Include region-size distribution and a largest-region prior, nearest-centroid routing, random routing and full-corpus dense retrieval.

Within selected cells rank answer embeddings by cosine with the unchanged query embedding. This isolates routing quality; a separate fine-ranking adapter can be investigated later. Routing cannot fix ordering errors among answers within the same region by itself.

High region accuracy with coarse cells is not sufficient: a single cell containing the whole corpus has perfect coverage and no pruning benefit. Probability calibration must be measured; softmax alone does not establish it. Lack of a full-encoder training run prevents measured cost/performance claims against encoder fine-tuning.
