# Tarsiq Baseline 1

Frozen Qwen3 embeddings plus deterministic request features and multinomial
logistic regression. Phase 1 labels in `../data_pipeline` are read-only inputs.

The full pipeline is intentionally gated: `baseline1_pipeline.py prepare` audits
data and creates splits; embedding and training require explicit later commands.

