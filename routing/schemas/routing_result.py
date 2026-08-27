from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal

Domain = Literal["math", "code", "logic", "knowledge", "instruction_following", "tool_use", "affective"]


@dataclass(frozen=True)
class DomainPrediction:
    domain: Domain
    raw_output: str | None = None
    chunks_classified: int = 1


@dataclass(frozen=True)
class LLMTierEstimate:
    tier: int
    raw_output: str | None = None
    chunks_classified: int = 1

    def __post_init__(self) -> None:
        if self.tier not in {1, 2, 3, 4}:
            raise ValueError("tier must be in 1..4")


@dataclass(frozen=True)
class LLMClassification:
    """Deprecated combined value retained only for artifact compatibility."""
    domain: Domain
    tier: int
    raw_output: str | None = None
    chunks_classified: int = 1

    def __post_init__(self) -> None:
        if self.tier not in {1,2,3,4}: raise ValueError("tier must be in 1..4")


@dataclass(frozen=True)
class TierRouterPrediction:
    tier: int
    confidence: float
    probabilities: dict[str, float]
    calibration_version: str | None = None

    def __post_init__(self) -> None:
        if self.tier not in {1, 2, 3, 4}:
            raise ValueError("tier must be in 1..4")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be in [0,1]")
        expected = {"T1", "T2", "T3", "T4"}
        if set(self.probabilities) != expected:
            raise ValueError("probabilities must contain T1..T4")
        if abs(sum(self.probabilities.values()) - 1.0) > 1e-4:
            raise ValueError("probabilities must sum to 1")


@dataclass(frozen=True)
class RoutingDecision:
    request_id: str
    domain: Domain
    final_tier: int
    selected_model: str
    selected_provider: str
    decision_policy_used: str
    reason_for_selection: str
    audit: dict[str, object] = field(default_factory=dict)
