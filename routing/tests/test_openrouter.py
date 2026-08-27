import json
import os

import pytest

from routing.models.openrouter import OpenRouterCombinedClassifier


def test_openrouter_adapter_uses_configured_model_and_schema(monkeypatch):
    monkeypatch.setenv("TEST_OPENROUTER_KEY","secret")
    captured={}
    def transport(payload,headers):
        captured.update(payload=payload,headers=headers)
        return {"choices":[{"message":{"content":json.dumps({"domain":"code","tier":2})}}]}
    result=OpenRouterCombinedClassifier("stealth/ox-alpha","TEST_OPENROUTER_KEY",transport=transport).classify("fix Python")
    assert result.domain=="code" and result.tier==2
    assert captured["payload"]["model"]=="stealth/ox-alpha"
    schema=captured["payload"]["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["tier"]["enum"]==[1,2,3]
    assert captured["headers"]["Authorization"]=="Bearer secret"

def test_missing_api_key_fails_before_transport(monkeypatch):
    monkeypatch.delenv("MISSING_OPENROUTER_KEY",raising=False)
    called=False
    def transport(*_):
        nonlocal called;called=True
    with pytest.raises(RuntimeError,match="Missing MISSING_OPENROUTER_KEY"):
        OpenRouterCombinedClassifier("model","MISSING_OPENROUTER_KEY",transport=transport).classify("x")
    assert not called
