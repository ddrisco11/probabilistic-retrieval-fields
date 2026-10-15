# Benchmark and evaluation protocol

## SciFact source and permissions

The initial benchmark is the BEIR-formatted SciFact archive, with 5,183 documents and 1,109 query texts. Its train qrels cover 809 queries and test qrels cover 300 queries. Query texts in the combined queries file are not themselves permission to train on every query. The loader builds partitions from qrels and rejects overlapping train/test identifiers.

Sources: [BEIR dataset listing](https://github.com/beir-cellar/beir/wiki/Datasets-available), [SciFact dataset card](https://huggingface.co/datasets/BeIR/scifact/blob/main/README.md), [original SciFact license](https://github.com/allenai/scifact/blob/master/LICENSE.md).

License metadata differs by artifact: the BEIR Hugging Face card lists CC-BY-SA-4.0; original SciFact claims/evidence are CC-BY-4.0, abstracts are ODC-By-1.0, and original repository code is Apache-2.0. Do not relabel these as the project's code license. Dataset and pretrained-model files are excluded from the publication package; retain source links and source hashes when reproducing. This is the BEIR retrieval task, not the original SciFact entailment/rationale evaluation.

## Splits and frozen encoding

Train/validation partition uses SHA256 ordering of `prf-2026:<query-id>` over official training query IDs: first rounded 20% validation, remaining training. Official test qrels are used only for final evaluation. Encoding test query text is permitted for evaluation; no test qrels enter fitting, loss calculation or checkpoint selection.

Documents concatenate title and abstract. The encoder is `sentence-transformers/all-MiniLM-L6-v2`, evaluated on CPU with gradients disabled. Both queries and documents are L2 normalized. The exact resolved model commit, tokenizer/model maximum length, source hashes and encoding wall time are stored next to the embedding cache. Truncation follows the model's maximum sequence length; long abstracts can lose information. All methods reuse the same cache and exact cosine index.

## Training

Single-vector and stochastic surrogate losses average log-softmax loss over all known positives, with all other corpus documents in the denominator. This is a full-corpus relevance surrogate, not hard-negative mining. Unjudged documents may contain unannotated positives. Graded labels affect evaluation; this first training implementation treats positive labels equally.

Mixture surrogate loss enumerates components, uses reparameterized Gaussian samples inside each, and weights their losses by learned probabilities. This preserves mixture-weight gradients without pretending categorical samples are reparameterizable. Multi-vector deterministic training uses max across heads for each document's logit.

Reward training uses per-action Recall@10 with an action-independent leave-one-out baseline across at least two IID samples of the same query. It does not optimize union coverage directly. Actions and rewards are detached in the score-function term. Hybrid training switches halfway through epochs; the best validation checkpoint may come from the surrogate phase, so identical hybrid/surrogate results can be legitimate.

The stochastic regularizer is the joint component/action KL to a uniform-component, query-centered Gaussian prior. For mixtures this upper-bounds the marginal KL; it is not exact mixture KL or spherical entropy. Deterministic models receive squared residual regularization. These penalties have different scales and require future validation tuning.

## Ranking and budgets

Requested samples: 1, 2, 4, 8, 16. Total returned slots: 40. Integer division allocates these slots to actual searches, with remainder assigned to earlier searches. Dense and deterministic single-vector methods use one search with depth 40. Four-head deterministic retrieval uses at most four searches. Stochastic policies use the requested number. Duplicates consume slots before union deduplication. Candidate ranking is maximum cosine over all generated locations, with document-index tie breaking. Return the first 10 after fusion.

Report nDCG@10 with exponential graded gains, Recall@10, truncated MRR@10, candidate recall, actual search calls, returned slots, unique candidates and all scored query-document pairs. Exact search still examines the entire corpus for every location; equal returned slots do **not** imply equal compute. Model, similarity, top-k selection, deduplication/ranking, and total query timings are recorded separately. Current exact implementation uses a stable full sort for deterministic ties, so latency is not optimized ANN latency.

## Statistics and selection

All methods share held-out queries. Three training seeds are predeclared. Fixed validation sampling RNG selects the best epoch at four requested samples. Test sampling uses an independent seed. Per-query scores and rankings are saved. Bootstrap differences resample queries after averaging seeds, relative to the deterministic single-vector baseline. Intervals are conditional on this dataset/configuration, do not represent between-dataset uncertainty, and have no multiple-testing correction. Seed standard deviations are reported separately.

Defaults are an exploratory fixed configuration, **not** a completed validation hyperparameter search. Do not interpret this as proof against all stochastic architectures. Synthetic geometry is intentionally small and partly easy; ceiling nDCG makes cluster coverage and candidate recall useful complements.

## Measurements not yet established

No fine-tuned encoder/LoRA cost comparison, GPU memory comparison, industrial ANN scaling, uncertainty calibration claim, or semantic subtopic coverage on SciFact is established. Synthetic known clusters support cluster coverage; SciFact relevance judgments do not supply subtopics. Peak RSS is process-wide and must not be used to rank model memory efficiency. Advanced architectures remain deferred until evidence warrants them.
