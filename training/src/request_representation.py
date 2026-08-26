from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Request:
    prompt_id: str
    task_text: str
    context_sections: tuple[str, ...] = field(default_factory=tuple)
    messages: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    tool_count: int = 0
    structured_output: bool = False
    json_schema_required: bool = False
    modalities: tuple[str, ...] = ("text",)

    @classmethod
    def benchmark(cls, prompt_id: str, prompt: str) -> "Request":
        # The prompt is the task. It is not duplicated as external context.
        return cls(prompt_id=str(prompt_id), task_text=str(prompt))

