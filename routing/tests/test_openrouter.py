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
    assert "response_format" not in captured["payload"]
    assert '"tier":<integer 1-3>' in captured["payload"]["messages"][0]["content"]
    assert captured["payload"]["reasoning"]=={"effort":"minimal","exclude":True}
    assert captured["headers"]["Authorization"]=="Bearer secret"

def test_missing_api_key_fails_before_transport(monkeypatch):
    monkeypatch.delenv("MISSING_OPENROUTER_KEY",raising=False)
    called=False
    def transport(*_):
        nonlocal called;called=True
    with pytest.raises(RuntimeError,match="Missing MISSING_OPENROUTER_KEY"):
        OpenRouterCombinedClassifier("model","MISSING_OPENROUTER_KEY",transport=transport).classify("x")
    assert not called

def test_invalid_json_is_retried_once(monkeypatch):
    monkeypatch.setenv("TEST_OPENROUTER_KEY","secret");calls=[]
    def transport(payload,headers):
        calls.append(len(payload["messages"]))
        content="not json" if len(calls)==1 else '{"domain":"math","tier":3}'
        return {"choices":[{"message":{"content":content}}]}
    result=OpenRouterCombinedClassifier("model","TEST_OPENROUTER_KEY",transport=transport).classify("prove it")
    assert result.domain=="math" and result.tier==3 and calls==[1,3]
