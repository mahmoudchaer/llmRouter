import json
import pytest
from routing.models.small_llm import CallableDomainClassifier,CallableTierClassifier,parse_combined,parse_domain,parse_tier
from routing.schemas.routing_result import DomainPrediction,LLMTierEstimate
from routing.prompts.domain_tier_prompt import build_domain_prompt,build_tier_prompt


def test_strict_classification_parser():
    assert parse_domain('{"domain":"code"}').domain=="code"
    assert parse_tier('{"tier":3}').tier==3
    combined=parse_combined('{"domain":"code","tier":2}')
    assert combined.domain=="code" and combined.tier==2


@pytest.mark.parametrize("text",[
    '{"domain":"coding"}',
    '{"domain":"code","reason":"x"}',
    'not json',
])
def test_invalid_or_ambiguous_output_is_not_silently_forced(text):
    with pytest.raises((ValueError,json.JSONDecodeError)):parse_domain(text)


def test_prompt_uses_frozen_empirical_policy():
    domain_prompt=build_domain_prompt("Fix this Python syntax error");tier_prompt=build_tier_prompt("Fix this Python syntax error")
    assert '{"domain":"<allowed_domain>"}' in domain_prompt and "Frozen empirical tier policy" not in domain_prompt
    assert "at least 60%" in tier_prompt and "minimum domain-specific model-capability group" in tier_prompt
    assert '{"tier":<integer 1-3>}' in tier_prompt and "Allowed domains" not in tier_prompt

def test_llm_backends_are_swappable_independently():
    domain=CallableDomainClassifier(lambda _:DomainPrediction("math"));tier=CallableTierClassifier(lambda _:LLMTierEstimate(2))
    assert domain.classify_domain("x").domain=="math" and tier.classify_tier("x").tier==2
