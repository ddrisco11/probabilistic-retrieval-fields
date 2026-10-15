# Portfolio description

Probabilistic Retrieval Fields investigates whether small relevance-trained routers can adapt search over frozen embeddings. The core experiment embeds questions and answer sentences using BGE-small, partitions the fixed answer corpus into regions, and trains a lightweight network to predict the regions containing labeled answers.

The repository includes a reproducible WikiQA-derived experiment, deterministic and stochastic controls, paired per-query comparisons, synthetic geometry experiments, and an interactive explorer for region granularity versus corpus coverage. The main engineering constraint is explicit: a useful router must preserve relevance while reducing the search workload.

This is exploratory research. Learned partition routing and probabilistic retrieval have substantial prior art. The project reports measured outcomes and limitations rather than claiming a new algorithm or an unmeasured advantage over encoder fine-tuning.
