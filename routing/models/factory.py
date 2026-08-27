from __future__ import annotations
from threading import Lock
from routing.models.small_llm import TransformersQwenDomainClassifier,TransformersQwenTierClassifier

ARGUMENTS=("model_id","revision","device","dtype","max_new_tokens","chunk_size_tokens","chunk_overlap_tokens")


def _kwargs(config:dict)->dict:return {key:config[key] for key in ARGUMENTS if key in config}


def build_local_llm_signals(domain_config:dict,tier_config:dict,share_model_instance:bool=True):
    """Build independent domain/tier interfaces; optionally share local weights to save Mac memory."""
    if not share_model_instance:
        return TransformersQwenDomainClassifier(**_kwargs(domain_config)),TransformersQwenTierClassifier(**_kwargs(tier_config))
    if (domain_config["model_id"],domain_config["revision"],domain_config.get("device"),domain_config.get("dtype")) != (
        tier_config["model_id"],tier_config["revision"],tier_config.get("device"),tier_config.get("dtype")):
        raise ValueError("Shared model instance requires identical model/revision/device/dtype")
    lock=Lock();domain=TransformersQwenDomainClassifier(**_kwargs(domain_config),generation_lock=lock)
    tier=TransformersQwenTierClassifier(**_kwargs(tier_config),model=domain.model,tokenizer=domain.tokenizer,generation_lock=lock)
    return domain,tier
