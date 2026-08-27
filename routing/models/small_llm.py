from __future__ import annotations
from abc import ABC,abstractmethod
from collections import Counter
import json,re
from threading import Lock
from typing import Any,Callable
from routing.prompts.domain_tier_prompt import DOMAINS,build_domain_prompt,build_tier_prompt
from routing.schemas.routing_result import DomainPrediction,LLMTierEstimate

class SmallLLMDomainClassifier(ABC):
    @abstractmethod
    def classify_domain(self,request_text:str)->DomainPrediction: ...

class SmallLLMTierClassifier(ABC):
    @abstractmethod
    def classify_tier(self,request_text:str)->LLMTierEstimate: ...

class SmallLLMClassifier(SmallLLMDomainClassifier,SmallLLMTierClassifier):
    """Compatibility base; implementations still make two independent calls."""
    def classify(self,request_text:str):
        domain=self.classify_domain(request_text);tier=self.classify_tier(request_text)
        from routing.schemas.routing_result import LLMClassification
        return LLMClassification(domain.domain,tier.tier,chunks_classified=max(domain.chunks_classified,tier.chunks_classified))

class CallableDomainClassifier(SmallLLMDomainClassifier):
    """Adapter for a local model, hosted endpoint, or future domain service."""
    def __init__(self,classifier:Callable[[str],DomainPrediction]):self.classifier=classifier
    def classify_domain(self,request_text:str)->DomainPrediction:return self.classifier(request_text)

class CallableTierClassifier(SmallLLMTierClassifier):
    """Adapter for an independent LLM tier backend without coupling orchestration to Qwen."""
    def __init__(self,classifier:Callable[[str],LLMTierEstimate]):self.classifier=classifier
    def classify_tier(self,request_text:str)->LLMTierEstimate:return self.classifier(request_text)

def _single_json(text:str)->dict:
    matches=re.findall(r"\{[^{}]*\}",text,flags=re.DOTALL)
    if len(matches)!=1:raise ValueError("Expected exactly one JSON object")
    return json.loads(matches[0])

def parse_domain(text:str)->DomainPrediction:
    value=_single_json(text)
    if set(value)!={"domain"} or value["domain"] not in DOMAINS:raise ValueError("Invalid constrained domain output")
    return DomainPrediction(value["domain"],text)

def parse_tier(text:str)->LLMTierEstimate:
    value=_single_json(text)
    if set(value)!={"tier"} or isinstance(value["tier"],bool) or int(value["tier"]) not in {1,2,3}:raise ValueError("Invalid constrained tier output")
    return LLMTierEstimate(int(value["tier"]),text)

class _TransformersQwenBase:
    def __init__(self,model_id:str,revision:str,device:str="auto",dtype:str="float16",max_new_tokens:int=20,
                 chunk_size_tokens:int=24576,chunk_overlap_tokens:int=256,model:Any=None,tokenizer:Any=None,
                 generation_lock:Lock|None=None):
        if chunk_overlap_tokens>=chunk_size_tokens:raise ValueError("chunk overlap must be smaller than chunk size")
        self.model_id,self.revision=model_id,revision;self.max_new_tokens=max_new_tokens
        self.chunk_size,self.overlap=chunk_size_tokens,chunk_overlap_tokens;self.generation_lock=generation_lock
        if model is None or tokenizer is None:
            import torch
            from transformers import AutoModelForCausalLM,AutoTokenizer
            if device=="auto":device="mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
            torch_dtype={"float16":torch.float16,"bfloat16":torch.bfloat16,"float32":torch.float32}[dtype]
            tokenizer=AutoTokenizer.from_pretrained(model_id,revision=revision)
            model=AutoModelForCausalLM.from_pretrained(model_id,revision=revision,torch_dtype=torch_dtype).to(device).eval()
        self.model,self.tokenizer=model,tokenizer
        self.model.generation_config.temperature=None;self.model.generation_config.top_p=None;self.model.generation_config.top_k=None
        self.device=next(model.parameters()).device if hasattr(model,"parameters") else device

    def _chunks(self,text:str)->list[str]:
        ids=self.tokenizer.encode(text,add_special_tokens=False)
        if len(ids)<=self.chunk_size:return [text]
        step=self.chunk_size-self.overlap
        return [self.tokenizer.decode(ids[start:start+self.chunk_size],skip_special_tokens=True) for start in range(0,len(ids),step)]

    def _generate(self,prompt:str,allowed_json:list[str])->str:
        import torch
        inputs=self.tokenizer.apply_chat_template([{"role":"user","content":prompt}],add_generation_prompt=True,tokenize=True,return_dict=True,return_tensors="pt")
        inputs={k:v.to(self.device) for k,v in inputs.items()};prompt_length=inputs["input_ids"].shape[1]
        sequences=[self.tokenizer.encode(value,add_special_tokens=False) for value in allowed_json];eos=self.tokenizer.eos_token_id
        def allowed(_batch_id,input_ids):
            generated=input_ids[prompt_length:].tolist();candidates=[seq for seq in sequences if seq[:len(generated)]==generated]
            if not candidates:return [eos]
            nxt={seq[len(generated)] for seq in candidates if len(seq)>len(generated)}
            return sorted(nxt) if nxt else [eos]
        def run():
            with torch.inference_mode():return self.model.generate(**inputs,max_new_tokens=self.max_new_tokens,do_sample=False,pad_token_id=eos,prefix_allowed_tokens_fn=allowed)
        if self.generation_lock:
            with self.generation_lock:output=run()
        else:output=run()
        return self.tokenizer.decode(output[0,prompt_length:],skip_special_tokens=True).strip()

class TransformersQwenDomainClassifier(_TransformersQwenBase,SmallLLMDomainClassifier):
    def classify_domain(self,request_text:str)->DomainPrediction:
        chunks=self._chunks(request_text);allowed=[json.dumps({"domain":d},separators=(",",":")) for d in DOMAINS]
        predictions=[parse_domain(self._generate(build_domain_prompt(chunk),allowed)) for chunk in chunks]
        if len(predictions)==1:return predictions[0]
        counts=Counter(p.domain for p in predictions);domain=sorted(counts,key=lambda d:(-counts[d],DOMAINS.index(d)))[0]
        return DomainPrediction(domain,json.dumps([p.raw_output for p in predictions]),len(chunks))

class TransformersQwenTierClassifier(_TransformersQwenBase,SmallLLMTierClassifier):
    def classify_tier(self,request_text:str)->LLMTierEstimate:
        chunks=self._chunks(request_text);allowed=[json.dumps({"tier":t},separators=(",",":")) for t in range(1,4)]
        predictions=[parse_tier(self._generate(build_tier_prompt(chunk),allowed)) for chunk in chunks]
        if len(predictions)==1:return predictions[0]
        return LLMTierEstimate(max(p.tier for p in predictions),json.dumps([p.raw_output for p in predictions]),len(chunks))
