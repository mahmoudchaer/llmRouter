from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class ConfidencePolicyConfig:
    high_confidence: float = 0.85
    low_confidence: float = 0.60

    def __post_init__(self) -> None:
        if not 0 <= self.low_confidence <= self.high_confidence <= 1:
            raise ValueError("Require 0 <= low <= high <= 1")

    def band(self, confidence: float) -> str:
        if confidence >= self.high_confidence: return "high"
        if confidence < self.low_confidence: return "low"
        return "medium"

