# Findings: frozen BGE-small region routing on WikiQA

## Main observation

Regional search achieves high answer-location coverage on this dataset. Learning the routing correction does not improve the tested tradeoff over nearest-centroid routing.

At 32 regions / 4 probes, the learned router achieves 91.36% any-answer region hit, 89.71% answer-region coverage, and 0.6709 nDCG@10 while scoring 13.53% of the corpus. Nearest-centroid routing achieves 93.00% hit, 91.46% coverage, and 0.6770 nDCG@10 while scoring 13.23%. The paired region-hit difference is −1.65 percentage points (95% interval −4.66 to +1.10); the nDCG difference is −0.0061 (−0.0302 to +0.0179). These intervals are exploratory and conditional on the single corpus partition.

At 128 regions / 16 probes, nearest-centroid routing reaches 97.53% hit while scoring 13.16% of the corpus; learned top routing reaches 90.67% while scoring 13.85%. The geometry-only control is strong. Stochastic sampling is not automatically advantageous just because the router outputs probabilities.

## Costs

The 32-region router has 26,721 trainable parameters. In this local run, 60 epochs including validation took a mean 0.33 seconds across three seeds using cached embeddings and one CPU thread. Encoding the complete question/answer corpus took 115.66 seconds on four CPU threads, excluding download/setup time. These are distinct costs. Partition construction times, per-query latency distributions and checkpoint provenance are in the result files.

The timing experiment is small and hardware-specific. It does not include full encoder fine-tuning, so it cannot quantify savings against that alternative. Fewer scored documents do not translate proportionally into latency savings because routing, candidate gathering, sorting and memory allocation contribute overhead.

## Interpretation and next experiments

The experiment demonstrates that answer regions can be predicted with high coverage, but much of that ability is already present in the frozen embedding geometry. A plausible explanation for the learned router's regression is limited supervision (873 answerable training questions) and overfitting; this is a hypothesis, not a proven diagnosis.

Next useful comparisons are a larger training set, multiple partition seeds, validation-tuned regularization, a calibrated nearest-centroid prior, and a separate query adapter for within-region ranking. Any new tuning must use validation labels. Subsequent test evaluations should be labeled as additional exploratory rounds or use a new held-out dataset to avoid repeated-test overinterpretation.

These results do not establish a general pretrained router, semantic uncertainty calibration, or superiority over encoder fine-tuning. They do establish a reproducible baseline for studying those questions.
