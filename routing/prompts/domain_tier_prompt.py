from __future__ import annotations

DOMAINS = ("math", "code", "logic", "knowledge", "instruction_following", "tool_use", "affective")
DOMAIN_DEFINITIONS = {
    "math": "Arithmetic, algebra, geometry, probability, proofs, or quantitative problem solving.",
    "code": "Programming, software engineering, debugging, algorithms, APIs, or repository changes.",
    "logic": "Deductive, symbolic, commonsense, puzzle, rule-based, or abstract reasoning where reasoning is central.",
    "knowledge": "Factual, scientific, medical, financial, professional, academic, or retrieval-style questions.",
    "instruction_following": "Writing, transformation, creative, formatting, or constraint-following tasks where compliance is central.",
    "tool_use": "Selecting or using tools, functions, APIs, databases, bookings, workflows, or agent actions.",
    "affective": "Emotion, sentiment, empathy, interpersonal affect, or emotional-state understanding.",
}
TIER_POLICY = """The tier is the minimum domain-specific model-capability group needed to solve the request reliably. The frozen labels were derived from real model outcomes: a group is sufficient only when at least 60% of its evaluated core models succeeded.
T1: the lowest-capability group is sufficient.
T2: T1 is insufficient; the second capability group is the first sufficient group.
T3: groups T1-T2 are insufficient; the strong third group is first sufficient.
T4: groups T1-T3 are insufficient; the strongest/frontier group is first sufficient.
Choose the minimum sufficient tier. Do not classify merely from length or topic. Account for reasoning depth, constraints, specialization, ambiguity, and reliability required."""

def build_domain_prompt(request_text: str) -> str:
    domains="\n".join(f"- {name}: {description}" for name,description in DOMAIN_DEFINITIONS.items())
    return f"""Classify only the primary semantic domain of the current request. /no_think
{domains}
Boundary rules: highest precedence—if the request asks the assistant to call/use a tool, function, API, database, calendar, booking system, or other external action, choose tool_use. Merely discussing an API is not tool_use. Code manipulation/debugging is code even when it contains math; a mathematical calculation or proof is math even though it requires reasoning; logic is for non-mathematical reasoning/puzzles. If the requested deliverable is rewriting, summarizing, translating, formatting, constrained composition, or creative text, choose instruction_following—not knowledge. Knowledge is for answering factual/substantive questions, not transforming supplied content.
Examples: fix Python syntax→code; prove infinitely many primes→math; solve a seating deduction puzzle→logic; answer a factual science question→knowledge; rewrite text under style constraints→instruction_following; book a flight with tools→tool_use; identify a speaker's emotion→affective.
Return exactly one compact JSON object: {{"domain":"<allowed_domain>"}}. No tier and no explanation.
REQUEST:
{request_text}"""

def build_tier_prompt(request_text: str) -> str:
    return f"""Estimate only the minimum model capability tier required for this request. /no_think
{TIER_POLICY}
Return exactly one compact JSON object: {{"tier":<integer 1-4>}}. No domain and no explanation.
REQUEST:
{request_text}"""

def build_domain_tier_prompt(request_text: str) -> str:
    """Deprecated compatibility helper; runtime uses separate calls."""
    return build_domain_prompt(request_text)+"\n"+build_tier_prompt(request_text)
