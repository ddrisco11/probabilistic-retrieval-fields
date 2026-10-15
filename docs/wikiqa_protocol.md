# BGE-small + WikiQA protocol

## Sources

- [WikiQA dataset on Hugging Face](https://huggingface.co/datasets/microsoft/wiki_qa)
- [WikiQA paper, EMNLP 2015](https://aclanthology.org/D15-1237/)
- [BGE-small-en-v1.5 model card](https://huggingface.co/BAAI/bge-small-en-v1.5)

WikiQA is a research question/answer-sentence relevance benchmark. Its source data has 20,360 training pairs, 2,733 validation pairs, and 6,165 test pairs. These are **pair counts**, not distinct question counts. The loader records distinct answerable question counts, exclusions, exact dataset/model commits, and source-file hashes in its manifest.

The dataset card specifies the Microsoft Research WikiQA data license rather than a general permissive code license. Raw sentences and questions are not redistributed here; reproduce from the linked source subject to its terms. BGE's model card lists MIT. This repository's code license does not change external data or model licenses.

## Task construction

Deduplicate answer sentence text across all official splits. Every unique sentence becomes one candidate in a global answer corpus, including negatively labeled candidates. Questions retain official train/validation/test membership. Reject overlapping question IDs or identical question text across splits. Exclude questions with no labeled positive from fitting and answer-location evaluation, and report their counts. This task therefore does not establish abstention/no-answer performance.

Positive judgments map each query to its deduplicated answer IDs. Shared answer text may legitimately occur across different questions; this is a transductive fixed-corpus setting, not held-out-document generalization. Partitioning sees all corpus embeddings but no question relevance labels. Unjudged answers from other questions can be relevant. Metrics are against the available labels only.

This is **not the official WikiQA candidate-restricted ranking protocol**. Global retrieval is harder and uses a different candidate set; do not compare these nDCG/MRR numbers to the WikiQA leaderboard. Official candidate lists are retained in the local bundle to support a future separate evaluation.

## Frozen encoder

Use `BAAI/bge-small-en-v1.5`, 384 dimensions, normalized embeddings. Prepend `Represent this sentence for searching relevant passages: ` to queries only, following the model card. Encode answer sentence text without a title or instruction. Resolve the supplied model revision to an immutable commit and record maximum sequence length and encoding time. No encoder weights are updated.

## Router and controls

Spherical k-means partitions the corpus into 8, 32, or 128 cells with seed 2026, k-means++ initialization and at most 40 iterations. The partition is shared across training seeds 11/22/33. Thus seed SD measures router/sampling variability, not partition variability. Region sizes and assignment hashes are saved. Empty cells, if any, are counted rather than silently removed.

The router predicts `softmax(cosine(q, centroid)/temperature + MLP(q))`. The MLP has one 64-unit tanh hidden layer and a zero-initialized output layer. Temperature is learned and bounded to [0.01,2]. This residual initialization makes the geometric prior a meaningful starting point. Cross-entropy uses the fraction of labeled relevant answers in each region. Validation cross-entropy selects the epoch; test labels are untouched until evaluation. No test-based hyperparameter search is performed.

Compare learned top-probability routing, learned sampling without replacement, nearest-centroid routing, largest-region routing and uniform random routing. Probe 1/2/4/8/16 cells where feasible. Selecting the same number of cells does not equalize document counts; report actual corpus fraction and scored-document count. Rank only documents in selected cells, using frozen query/document cosine. Full-corpus frozen dense retrieval provides a no-pruning endpoint.

The largest-region baseline is an occupancy prior, not a relevance-trained class prior. The nearest-centroid baseline uses fixed temperature 0.1; its top-cell ordering is unaffected by temperature, but its probability scores depend on it. Probability calibration comparisons require additional temperature calibration on validation data.

## Interpretation

Top-k any-answer region hit is the fraction of queries with at least one labeled answer in selected cells. Answer-region coverage is the fraction of each query's labeled answers whose cells were selected. Cross-entropy and squared error to the empirical regional distribution evaluate probability fit; neither alone establishes well-calibrated uncertainty. Soft targets represent a randomly drawn relevant answer, not independent cell-presence probabilities.

Report query-paired confidence intervals for learned-top minus centroid, averaging training seeds before bootstrap. These are exploratory, conditional intervals with no multiple-comparison correction. Latency includes routing, candidate-list construction and exact ranking, but excludes text encoding. Training time includes checkpoint validation; partition construction and embedding computation are reported separately. There is no measured cost comparison to encoder fine-tuning.

The router is corpus/partition-specific. It is not pretrained for arbitrary future corpora. High accuracy at coarse granularity can be trivial; assess it jointly with corpus fraction, final relevance quality and the nearest-centroid control.

## Upstream model exposure

This experiment does not audit BGE's pretraining/fine-tuning data for prior exposure to WikiQA or related Wikipedia text. It establishes the router's held-out-question protocol, not a claim that the frozen encoder has never encountered the benchmark. A second dataset and an out-of-domain test would strengthen generalization evidence.

## Fine-granularity follow-up

The follow-up keeps the same frozen BGE cache, official question splits, 64-unit router, optimization settings, partition seed and three training seeds. It tests 1,000, 4,096 and 8,192 regions. Probe budgets extend to 128, 512 and 1,024 respectively so that coarse and fine partitions can be compared at overlapping corpus-search fractions. Regions and probes are distinct: increasing region count makes cells smaller; increasing probes searches more cells.

For these larger partitions, k-means++ initialization incrementally maintains each document's distance to its nearest chosen center. This uses the same mathematical initialization criterion while avoiding recomputation against every previously chosen center. The original experiment retains its original initialization code path for reproducibility. Floating-point differences between paths can affect assignments; saved partition hashes identify the actual partitions.

These are additional exploratory tests on the previously inspected test split, not an independent confirmation dataset. Granularity settings were specified before inspecting their outcomes; no post-result tuning of the router is included. The number of training-supported cells, median documents per cell and fraction of test answers in unsupported cells are recorded in `granularity-diagnostics.json`. These support counts are diagnostic, not training input.

The fine-grained model remains a region classifier with a learned correction to centroid scores. It does not generate multiple continuous query displacement vectors. Consequently, these results test the granularity hypothesis for this classifier, not every possible multi-destination adapter.
