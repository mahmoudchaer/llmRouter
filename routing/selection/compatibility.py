from __future__ import annotations
from routing.policies.customer_budget import within_price_ceiling
from routing.schemas.model import ModelRecord
from routing.schemas.request import RoutingRequest


def compatibility_reasons(model: ModelRecord, request: RoutingRequest) -> list[str]:
    r=request.requirements; reasons=[]
    if not model.available: reasons.append("unavailable")
    if model.context_window < r.context_tokens: reasons.append("context_window")
    if r.requires_tools and not model.supports_tools: reasons.append("tool_support")
    if not r.input_modalities <= model.input_modalities: reasons.append("input_modalities")
    if not r.output_modalities <= model.output_modalities: reasons.append("output_modalities")
    if r.requires_structured_output and not model.supports_structured_output: reasons.append("structured_output")
    if r.provider_allowlist is not None and model.provider not in r.provider_allowlist: reasons.append("provider_allowlist")
    if model.provider in r.provider_blocklist: reasons.append("provider_blocklist")
    if r.model_allowlist is not None and model.model_id not in r.model_allowlist: reasons.append("model_allowlist")
    if model.model_id in r.model_blocklist: reasons.append("model_blocklist")
    if not within_price_ceiling(model, request.price_ceiling): reasons.append("price_ceiling")
    return reasons


def filter_compatible(models: list[ModelRecord], request: RoutingRequest) -> list[ModelRecord]:
    return [model for model in models if not compatibility_reasons(model, request)]

