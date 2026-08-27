# Tarsiq Runtime Routing

This package combines three independent request-level signals: a constrained small-LLM domain call, a separate small-LLM tier call, and the selected dedicated Tier Router checkpoint. All three interfaces can use different models or endpoints and are launched concurrently. It then applies an explicit safety-biased tier policy, hard compatibility and price filters, and cheapest-capable model selection.

On the current Mac, the two Qwen calls may share one in-memory model instance behind a generation lock to reduce memory. They still use separate prompts, schemas, and interfaces. Set `share_model_instance: false`, or provide different hosted adapters, for true concurrent model execution.

The local default is pinned to `Qwen/Qwen3-4B-Instruct-2507` at revision `cdbee75f17c01a7cc42f958dc650907174af0554`. The domain and tier classifiers share this local model instance by default but remain independent calls.

The official tier semantics are empirical. T1–T4 represent the lowest domain-specific capability group whose core models reached the frozen 60% reliability threshold. The runtime does not reinterpret tiers as generic easy/medium/hard labels.

## Safety invariants

- Price ceilings and compatibility requirements are hard filters.
- No model below the final required capability tier is silently selected.
- Disagreement defaults upward; any downward override requires calibrated configuration.
- Full prompts are not logged by the runtime decision object.
- Long small-LLM inputs are chunked with overlap and conservatively aggregated; they are not silently truncated.
- Confidence thresholds in `config/routing.yaml` remain provisional until generated from validation predictions.

The dedicated checkpoint path intentionally remains null until the seed-17 study selects its final model.
