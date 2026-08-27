from __future__ import annotations
from dataclasses import dataclass
from routing.schemas.model import ModelRecord
from routing.schemas.request import RoutingRequest
from .compatibility import filter_compatible


class NoSuitableModelError(RuntimeError): pass


@dataclass(frozen=True)
class Selection:
    model: ModelRecord
    compatible_model_ids: tuple[str, ...]
    capable_model_ids: tuple[str, ...]
    estimated_cost: float
    requested_tier: int
    selected_capability_tier: int
    capability_shortfall: bool
    selection_reason: str


class ModelSelector:
    def __init__(self,allow_capability_shortfall_fallback:bool=True):
        self.allow_capability_shortfall_fallback=allow_capability_shortfall_fallback

    def select(self, models: list[ModelRecord], request: RoutingRequest, domain: str, tier: int) -> Selection:
        compatible=filter_compatible(models,request)
        if not compatible:
            raise NoSuitableModelError("No model remains after hard compatibility and price-ceiling filters")
        capable=[m for m in compatible if m.capability_by_domain.get(domain,0)>=tier]
        if not capable:
            highest=max((m.capability_by_domain.get(domain,0) for m in compatible),default=0)
            if not self.allow_capability_shortfall_fallback or highest<=0:
                raise NoSuitableModelError(f"No model satisfies domain={domain}, tier={tier} within hard constraints; highest eligible capability={highest}")
            capable=[m for m in compatible if m.capability_by_domain.get(domain,0)==highest]
            shortfall=True;reason="requested tier unavailable within hard ceiling; selected strongest eligible capability"
        else:
            shortfall=False;reason="selected cheapest compatible model meeting requested capability"
        input_tokens=max(request.requirements.context_tokens,1);output_tokens=request.expected_output_tokens or 1
        def cost(m:ModelRecord)->float:return (input_tokens*m.input_price_per_1m+output_tokens*m.output_price_per_1m)/1_000_000
        chosen=min(capable,key=lambda m:(cost(m),m.input_price_per_1m+m.output_price_per_1m,m.model_id))
        selected_tier=chosen.capability_by_domain[domain]
        return Selection(chosen,tuple(m.model_id for m in compatible),tuple(m.model_id for m in capable),cost(chosen),tier,selected_tier,shortfall,reason)
