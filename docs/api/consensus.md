# Consensus API

SRF is a non-convex method, so different random initializations can land on slightly different embeddings. The consensus transformers run several random starts and aggregate them into a single, stable embedding. They follow the scikit-learn transformer API (`fit`/`transform`), so you can drop them into a `Pipeline` in place of a single `SRF`. See [examples](../examples.md) for how to use them in a workflow.

::: pysrf.EnsembleFit
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

::: pysrf.AlignedConsensus
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3
