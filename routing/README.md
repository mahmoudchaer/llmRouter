# Tarsiq Runtime Routing

This package combines three independent request-level signals: a constrained small-LLM domain call, a separate small-LLM tier call, and the selected dedicated Tier Router checkpoint. All three interfaces can use different models or endpoints and are launched concurrently. It then applies an explicit safety-biased tier policy, hard compatibility and price filters, and cheapest-capable model selection.

On the current Mac, the two Qwen calls may share one in-memory model instance behind a generation lock to reduce memory. They still use separate prompts, schemas, and interfaces. Set `share_model_instance: false`, or provide different hosted adapters, for true concurrent model execution.

The active MVP uses one generic OpenRouter call returning both domain and tier. Its model is controlled only by `classifier_model` in `config/routing.yaml`; the adapter reads `OPENROUTER_API_KEY` and contains no model-specific behavior. NVIDIA Nemotron 3 Ultra Free is the current test model, while `qwen/qwen3-30b-a3b-instruct-2507` is documented as the planned paid fallback. JSON is enforced by a strict prompt, exact parser validation, and one generic retry because the free Nemotron endpoint does not support OpenRouter `response_format` enforcement.

The official three-tier semantics are empirical and preserve the frozen 60% reliability policy: T1 is old T1, T2 merges old T2/T3, and T3 is old T4. The runtime treats these as minimum capability requirements: cheap/small, capable mid-range, and strongest permitted by the customer's ceiling.

## Safety invariants

- Price ceilings and compatibility requirements are hard filters.
- If the requested tier is unavailable inside the customer's hard ceiling, the configurable fallback selects the strongest compatible model inside that ceiling and emits `capability_shortfall=true`; it never downgrades silently.
- Disagreement defaults upward; any downward override requires calibrated configuration.
- Full prompts are not logged by the runtime decision object.
- Long small-LLM inputs are chunked with overlap and conservatively aggregated; they are not silently truncated.
- Confidence thresholds in `config/routing.yaml` remain provisional until generated from validation predictions.

The dedicated checkpoint path intentionally remains null until the seed-17 study selects its final model.

Validation predictions can be converted into confidence-bucket and cumulative-threshold calibration artifacts. A trust threshold is recommended only when it has enough examples and satisfies configured under-routing and severe-under-routing limits; until then, the thresholds remain provisional.
