import json
import pytest
from routing.models.small_llm import parse_domain,parse_tier
from routing.prompts.domain_tier_prompt import build_domain_prompt,build_tier_prompt


def test_strict_classification_parser():
    assert parse_domain('{"domain":"code"}').domain=="code"
    assert parse_tier('{"tier":3}').tier==3


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
