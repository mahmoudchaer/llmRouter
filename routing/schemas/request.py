from __future__ import annotations
from dataclasses import dataclass, field


@dataclass(frozen=True)
class CustomerPriceCeiling:
    max_input_price_per_1m: float
    max_output_price_per_1m: float

    def __post_init__(self) -> None:
        if self.max_input_price_per_1m < 0 or self.max_output_price_per_1m < 0:
            raise ValueError("Price ceilings must be non-negative")


@dataclass(frozen=True)
class HardRequirements:
    context_tokens: int = 0
    requires_tools: bool = False
    input_modalities: frozenset[str] = frozenset({"text"})
    output_modalities: frozenset[str] = frozenset({"text"})
    requires_structured_output: bool = False
    provider_allowlist: frozenset[str] | None = None
    provider_blocklist: frozenset[str] = frozenset()
    model_allowlist: frozenset[str] | None = None
    model_blocklist: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if self.context_tokens < 0:
            raise ValueError("context_tokens must be non-negative")


@dataclass(frozen=True)
class RoutingRequest:
    request_id: str
    task_text: str
    price_ceiling: CustomerPriceCeiling
    requirements: HardRequirements = field(default_factory=HardRequirements)
    expected_output_tokens: int | None = None
    messages: tuple[dict[str, str], ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.task_text.strip():
            raise ValueError("task_text must not be empty")
        if self.expected_output_tokens is not None and self.expected_output_tokens < 0:
            raise ValueError("expected_output_tokens must be non-negative")

