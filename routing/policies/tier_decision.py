from __future__ import annotations
from dataclasses import dataclass
from .confidence import ConfidencePolicyConfig
from routing.schemas.routing_result import LLMTierEstimate, TierRouterPrediction


@dataclass(frozen=True)
class TierDecision:
    tier: int
    policy_used: str
    disagreement: int
    reason: str


class TierDecisionPolicy:
    def __init__(self, confidence: ConfidencePolicyConfig = ConfidencePolicyConfig(),
                 large_disagreement_tiers: int = 2,
                 allow_high_confidence_downward_disagreement: bool = False):
        self.confidence = confidence
        self.large_disagreement_tiers = large_disagreement_tiers
        self.allow_high_confidence_downward_disagreement = allow_high_confidence_downward_disagreement

    def decide(self, dedicated: TierRouterPrediction, llm: LLMTierEstimate) -> TierDecision:
        gap = abs(dedicated.tier - llm.tier); band = self.confidence.band(dedicated.confidence)
        if dedicated.tier == llm.tier:
            return TierDecision(dedicated.tier, "agreement", 0, "Both tier signals agree")
        if gap >= self.large_disagreement_tiers:
            return TierDecision(max(dedicated.tier, llm.tier), "large_disagreement_safe_upward", gap,
                                "Large disagreement; selected safer higher tier")
        if band == "low":
            return TierDecision(max(dedicated.tier, llm.tier), "low_confidence_safe_upward", gap,
                                "Dedicated router confidence is below calibrated trust band")
        if dedicated.tier > llm.tier:
            return TierDecision(dedicated.tier, "dedicated_is_safer", gap,
                                "Dedicated router recommends the higher tier")
        if band == "high" and self.allow_high_confidence_downward_disagreement:
            return TierDecision(dedicated.tier, "calibrated_high_confidence_override", gap,
                                "Calibrated policy permits one-tier downward disagreement")
        return TierDecision(llm.tier, "disagreement_safe_upward", gap,
                            "Insufficient calibration evidence to downgrade below LLM estimate")
