from __future__ import annotations

import json
import os
from typing import Callable
from urllib.request import Request,urlopen

from routing.models.small_llm import CombinedSmallLLMClassifier,parse_combined
from routing.prompts.domain_tier_prompt import DOMAINS,build_combined_domain_tier_prompt
from routing.schemas.routing_result import LLMClassification


Transport=Callable[[dict,dict[str,str]],dict]


class OpenRouterCombinedClassifier(CombinedSmallLLMClassifier):
    """Model-agnostic OpenRouter adapter for one constrained domain+tier call."""
    def __init__(self,model:str,api_key_env:str="OPENROUTER_API_KEY",
                 endpoint:str="https://openrouter.ai/api/v1/chat/completions",timeout_seconds:float=120,
                 max_output_tokens:int=40,transport:Transport|None=None):
        if not model:raise ValueError("OpenRouter classifier model must not be empty")
        self.model,self.api_key_env,self.endpoint=model,api_key_env,endpoint
        self.timeout_seconds,self.max_output_tokens=timeout_seconds,max_output_tokens
        self.transport=transport or self._http_transport

    def estimate_tokens(self,text:str)->int:
        # Used only for evaluation accounting/skip policy; OpenRouter tokenizes server-side.
        return max(1,(len(text)+3)//4)

    def _http_transport(self,payload:dict,headers:dict[str,str])->dict:
        request=Request(self.endpoint,data=json.dumps(payload).encode(),headers=headers,method="POST")
        with urlopen(request,timeout=self.timeout_seconds) as response:return json.loads(response.read())

    def classify(self,request_text:str)->LLMClassification:
        api_key=os.environ.get(self.api_key_env)
        if not api_key:raise RuntimeError(f"Missing {self.api_key_env}; no OpenRouter request was sent")
        schema={"name":"tarsiq_domain_tier","strict":True,"schema":{"type":"object","additionalProperties":False,
                "properties":{"domain":{"type":"string","enum":list(DOMAINS)},"tier":{"type":"integer","enum":[1,2,3]}},
                "required":["domain","tier"]}}
        payload={"model":self.model,"messages":[{"role":"user","content":build_combined_domain_tier_prompt(request_text)}],
                 "temperature":0,"max_tokens":self.max_output_tokens,"response_format":{"type":"json_schema","json_schema":schema}}
        response=self.transport(payload,{"Authorization":f"Bearer {api_key}","Content-Type":"application/json"})
        try:content=response["choices"][0]["message"]["content"]
        except (KeyError,IndexError,TypeError) as error:raise RuntimeError(f"Malformed OpenRouter response: {response}") from error
        if not isinstance(content,str):raise RuntimeError("OpenRouter classifier content must be a JSON string")
        parsed=parse_combined(content)
        return LLMClassification(parsed.domain,parsed.tier,content,1)
