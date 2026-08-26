from __future__ import annotations
from dataclasses import dataclass,field
from typing import Any

@dataclass(frozen=True)
class RequestSection:
    kind: str
    text: str
    metadata: dict[str,Any]=field(default_factory=dict)

@dataclass(frozen=True)
class HierarchicalRequest:
    prompt_id: str
    sections: tuple[RequestSection,...]
    metadata: dict[str,Any]=field(default_factory=dict)

    @classmethod
    def benchmark(cls,prompt_id,prompt):
        return cls(str(prompt_id),(RequestSection("user_task",str(prompt)),),{"source":"benchmark"})

    def faithful_text(self):
        if len(self.sections)==1 and self.sections[0].kind=="user_task":return self.sections[0].text
        return "\n\n".join(f"<{s.kind}>\n{s.text}\n</{s.kind}>" for s in self.sections)

