from __future__ import annotations
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelRecord:
    model_id: str
    provider: str
    capability_by_domain: dict[str, int]
    input_price_per_1m: float
    output_price_per_1m: float
    context_window: int
    supports_tools: bool = False
    input_modalities: frozenset[str] = frozenset({"text"})
    output_modalities: frozenset[str] = frozenset({"text"})
    supports_structured_output: bool = False
    available: bool = True
    latency_ms_p50: float | None = None
    reliability: float | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.context_window <= 0:
            raise ValueError("context_window must be positive")
        if self.input_price_per_1m < 0 or self.output_price_per_1m < 0:
            raise ValueError("Model prices must be non-negative")
        if any(tier not in {1, 2, 3, 4} for tier in self.capability_by_domain.values()):
            raise ValueError("Capability tiers must be in 1..4")

