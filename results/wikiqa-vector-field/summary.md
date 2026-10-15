# BGE-small to BGE-base vector field

Each arrow connects the same text across encoders after a train-fitted alignment. Arrows are observed model differences, not learned query-to-answer movements.

| Measurement | Value |
|---|---:|
| Base centered energy retained by 768→384 alignment | 91.5% |
| Held-out displacement RMS / shuffled-alignment control | 0.6228 / 1.0836 |
| Endpoint PCA training variance: 2D / 3D | 6.3% / 8.6% |
| Endpoint PCA held-out arrow energy: 2D / 3D | 0.7% / 1.1% |
| Displacement PCA held-out centered arrow energy: 2D / 3D | 1.4% / 2.1% |
| Global drift share of held-out arrow energy | 0.1% |
| Centered neighbor / random direction cosine | 0.1337 / 0.0003 |
| 10-neighbor displacement MSE improvement over global mean | 10.2% |

Questions: 633 held-out texts, displacement RMS 0.6879; within-type neighbor/random direction cosine 0.1124/-0.0001; neighbor prediction MSE improvement 8.3%.
Answers: 5,134 held-out texts, displacement RMS 0.6143; within-type neighbor/random direction cosine 0.1297/0.0003; neighbor prediction MSE improvement 9.6%.

Native full-corpus answer retrieval (243 answerable test questions):

| Encoder | nDCG@10 | Recall@10 |
|---|---:|---:|
| small | 0.7281 | 0.8748 |
| base | 0.7621 | 0.8848 |
| aligned_base | 0.7673 | 0.8807 |

Base-minus-small nDCG difference: 0.0340, paired 95% interval [0.014227236805273633, 0.055467129566177394].

[Protocol](../../docs/vector_field_protocol.md) · [Interactive 2D/3D viewer](https://ddrisco11.github.io/probabilistic-retrieval-fields/vector-field.html)
