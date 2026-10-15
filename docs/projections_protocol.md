# Continuous projections with expected-distance reward

## Formulation

The frozen `BAAI/bge-small-en-v1.5` model embeds questions and answer sentences into 384-dimensional unit vectors. A shared 64-unit tanh MLP maps a query to K residual offsets and K weight logits. Each destination is `normalize(query + offset)`. An offset encodes a direction and magnitude implicitly. Only the adapter is trained; no region assignments enter this experiment.

For each query, training minimizes:

```
loss = mean over labeled relevant answers a of
       sum over every head i of weight_i * ||destination_i - a||²
reward = -loss
```

Thus weights 0.6 and 0.4 give exactly `0.6 * reward_1 + 0.4 * reward_2`. No weights are dropped, renormalized after truncation, sampled, or replaced by a maximum. Multiple positive answers receive equal weight per question. Negative answers are not added to this literal distance objective. This isolates the requested objective rather than mixing it with a contrastive ranking loss.

Squared Euclidean distance on unit vectors is `2 - 2*cosine`. Normalization maintains compatibility with frozen cosine retrieval; radial magnitude of the final destination is not an independently searched dimension. Offset magnitude can change the destination's direction before normalization.

## Models and learned counts

Frozen BGE is the no-adaptation baseline. Learned models use one, four, or eight softmax-weighted heads, plus eight sparsemax-weighted heads. Sparsemax projects logits onto the probability simplex, allowing learned exact zero weights while retaining a sum of one. This lets the model select an active subset, capped at eight; it does not dynamically grow the network or avoid computing inactive output heads. No threshold on cumulative probability is applied. All models start near the original query with equal weights and small independent random offsets.

Report both `1 / sum(weight²)` (effective head count) and the number of strictly positive weights. Neither proves distinct destinations. Weighted pairwise squared distance measures destination dispersion. There is no diversity regularizer, entropy bonus, or penalty on active count.

## What squared distance implies

Let `m = sum_i weight_i * destination_i`. The identity

```
sum_i weight_i * ||destination_i - answer||²
  = ||m - answer||² + sum_i weight_i * ||destination_i - m||²
```

holds when weights sum to one. For a fixed query, the second term is independent of the candidate answer. Ranking by expected squared distance therefore equals ranking by distance to the weighted-average vector. With unit document vectors it is also equivalent to a dot-product ranking using that vector. The implementation uses this exact algebraic reduction, preserving all mass and avoiding redundant full-corpus searches. It is not evidence of a multimodal retrieval speedup.

Furthermore, expected reward is linear in the weights. Holding destinations fixed, the best unconstrained weighting can concentrate on the destination with the greatest expected reward (or split between ties). Multiple heads are not inherently rewarded for covering different answers under this objective. Diverse output locations may collapse, or retain substantial weights while becoming geometrically redundant. A nonlinear distance reward would be a different experiment, even if its final head aggregation remained an arithmetic weighted sum.

## Data, training and evaluation

Use the same cached embeddings, pinned model/dataset revisions, deduplicated 26,196-answer corpus, and official answerable WikiQA splits as [the region protocol](wikiqa_protocol.md): 873 train, 126 validation, 243 test questions. This remains an adapted global retrieval task, not the official candidate-restricted leaderboard. Earlier experiments examined this test set, so the new comparison is exploratory. Pretraining exposure and unjudged cross-question relevance remain limitations.

Seeds are 11, 22 and 33. Train for 60 epochs with AdamW, learning rate 0.001, zero weight decay, batches of 64, gradient norm clipping at 5, one CPU thread. Select the minimum validation expected-distance checkpoint, including epoch-zero initialization. Do not select checkpoints using test data or nDCG. Full-corpus test ranking uses negative expected squared distance with stable document-ID-order ties. Report expected distance, nDCG@10, recall@10, weight statistics and movement. Lower expected distance is better; higher retrieval metrics are better. These objectives can disagree.

Paired bootstrap intervals resample test questions after averaging seeds, for comparisons with frozen BGE and the single-head model. They do not capture new-dataset uncertainty. Runtime includes adapter training, evaluation and bootstrap, and excludes the previously cached embedding creation. All relevant answer labels used in training belong to training questions; validation labels only select checkpoints.

## Reproduction

```bash
python -m prf.projections --data cache/wikiqa-bge.pt \
  --config configs/wikiqa-projections.json --output results/wikiqa-projections
MPLCONFIGDIR="$PWD/cache/matplotlib" python scripts/projection_report.py
```

Run from the repository root after installing the benchmark and development dependencies. Output directories must be new. Local `.pt` checkpoints and embeddings are ignored by Git; public result files contain numeric measurements, query IDs and hashed answer IDs, without raw dataset text or conversation transcripts. The report script expects the canonical result directory above.
