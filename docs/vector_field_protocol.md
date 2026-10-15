# BGE-small → BGE-base displacement field

## What an arrow represents

Both encoders embed the **same** WikiQA text. A question arrow connects its small embedding to its aligned base embedding; an answer arrow does the same for an answer sentence. This is an empirical field sampled at corpus points, not a continuous field evaluated everywhere, a trained adapter, or a question-to-relevant-answer displacement.

Use frozen [BGE-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) (384 dimensions) and [BGE-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) (768 dimensions). Both use normalized embeddings, a 512-token limit, identical query instructions, no answer instruction, and CPU inference with four threads. The exact revisions and embedding hashes are in the result metadata. Small's existing cached embeddings are reused; base is newly encoded. All 26,196 deduplicated answer sentences and 3,047 questions are encoded by each model.

## Coordinate alignment

The two encoders do not share coordinate axes. Subtracting raw coordinates, padding and subtracting without alignment, or independently plotting each model's PCA would not give an interpretable movement field.

Fit a centered rectangular orthogonal Procrustes map using paired training texts:

```
B = base_train - mean_base_train
S = small_train - mean_small_train
U, singular_values, Vh = svd(B.T @ S)
W = U @ Vh                         # 768 × 384, W.T @ W = I
aligned_base = (base - mean_base_train) @ W + mean_small_train
displacement = aligned_base - small
```

No scale parameter is fitted. No endpoint renormalization is applied for field analysis. This map is a compression, so it cannot preserve all base-model information. Report the held-out fraction of centered base energy retained. Also lift small into base coordinates with `W.T` and decompose full residual energy into the displayed 384-dimensional component and discarded orthogonal component. A shuffled training-pair alignment is a correspondence control, not a formal permutation significance test.

Arrows depend on the chosen alignment. Inward or outward patterns can reflect differences in covariance or compression and cannot automatically be interpreted as semantic improvements. Translation aligns training means by construction; the measured test drift is an out-of-sample residual.

## Splits

Use all official question splits for geometry, including questions without labeled positives. Deduplicate answer sentences globally. Remove answer texts that occur across multiple official splits from both field fitting and field evaluation. Fit alignment and PCA on training questions and training-exclusive answers only. Each text receives equal weight, so answers dominate the fit by count. Validation and test points do not fit any transform. Test statistics cover all held-out questions and test-exclusive answers. Split IDs and exclusions are saved.

All original corpus documents remain available for the separate retrieval evaluation. That evaluation uses the prior answerable-test subset and relevance judgments. Exact same-text identity, candidate lists, relevance labels, split memberships, and source file hashes are checked between the two embedding bundles.

## Two PCA views

1. **Spatial view:** fit one PCA to pooled small and aligned-base training endpoints. Project both endpoints through that same mean and basis. Draw arrows from small to aligned base in the first two or three PCs.
2. **Movement view:** fit PCA to centered training displacement vectors. Project both endpoints through those same direction axes. This emphasizes variation in movements; it is not the PCA that best preserves the original point cloud.

Never fit independent bases for arrow starts and ends. PCA components are ordered by explained variance. Display a fixed seeded sample of 500 held-out points at true projected arrow length. The browser allows rotation, switching 2D/3D, choosing PCA basis and filtering question/answer types. Quantitative statistics use all held-out points, not just the plotted sample.

Report training variance captured, held-out displacement energy captured in spatial PCA, and held-out displacement energy around the training mean captured in movement PCA. These have different denominators and are labeled separately. A visually simple field with low retained energy is not evidence that the full field is low-dimensional.

## Pattern checks

- Global drift energy fraction, residual RMS and question/answer breakdowns.
- Ten nearest training origins in small-model cosine space for each test point; compare centered displacement direction cosine with ten randomly selected training points. Subtract training mean displacement before direction comparisons to remove global drift.
- Average neighbors' full 384-dimensional displacements to predict the test displacement. Compare squared prediction error against the training-global-mean displacement. All neighbors and targets used for prediction come from training texts. No model fitting or hyperparameter search is performed; k=10 is fixed in advance.
- Repeat locality checks separately within question and answer types, with type-specific mean displacement and same-type neighbors, to reduce type separation as a confound.
- A paired bootstrap over held-out texts summarizes the pooled neighbor-minus-random directional difference. Texts can share topics or source articles; this interval is descriptive and does not model topic clustering or establish cross-dataset generalization.

This tests whether the observed field has locally predictable structure, not whether any adapter can replace base. Base embeddings are still computed for every observed endpoint.

## Retrieval context

Compare full-corpus cosine retrieval with native small, native base, and the aligned-base representation (renormalized for this retrieval comparison only). Report nDCG@10 and recall@10 on 243 answerable test questions. Native base improvement would show a reason to investigate its geometry; it would not prove that the aligned field retains the improvement or that a predicted movement reproduces it. The task is the same adapted global WikiQA retrieval task used earlier, not its official candidate-restricted leaderboard. Repeated benchmark use makes these exploratory findings.

## Reproduce

```bash
HF_HOME="$PWD/cache/huggingface" python -m prf.wikiqa \
  --dataset-revision 3f104672b5de699878fe7907afc486f0de325eb5 \
  --model-name BAAI/bge-base-en-v1.5 \
  --model-revision a5beb1e3e68b9ab74eb54cfd186867f64f240e1a \
  --output cache/wikiqa-bge-base.pt
python -m prf.vector_field --output results/wikiqa-vector-field
MPLCONFIGDIR="$PWD/cache/matplotlib" python scripts/vector_field_report.py
```

Prepare the small cache using the pinned command in the main README. Source parquet files must remain in the local Hugging Face cache; the analysis reads them offline and verifies recorded hashes. Output directories must be new. Embeddings and fitted alignment weights remain local and ignored by Git. Public artifacts include numeric statistics, projected sample coordinates, question IDs and hashed answer IDs; no raw text or conversation history is included.
