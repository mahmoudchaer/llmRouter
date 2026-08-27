from __future__ import annotations
import torch
from transformers import AutoModel
from peft import LoraConfig,get_peft_model
from .tier_router_model import TierRouter

def build_tier_router(cfg,device="cpu",loss_kind="ordinal",structural=True):
    dtype=torch.bfloat16 if device=="cuda" else torch.float32
    enc=AutoModel.from_pretrained(cfg["model"]["model_id"],revision=cfg["model"]["revision"],torch_dtype=dtype)
    enc.config.use_cache=False
    if cfg["training"]["gradient_checkpointing"]:enc.gradient_checkpointing_enable()
    l=cfg["training"]["lora"]
    enc=get_peft_model(enc,LoraConfig(r=l["rank"],lora_alpha=l["alpha"],lora_dropout=l["dropout"],target_modules=l["target_modules"],bias="none",task_type="FEATURE_EXTRACTION"))
    if cfg["training"]["gradient_checkpointing"]:enc.enable_input_require_grads()
    hidden=enc.config.hidden_size
    return TierRouter(enc,hidden,loss_kind,cfg["structural_features"]["dimension"] if structural else 0,num_tiers=cfg.get("num_tiers",4)).to(device=device,dtype=dtype)

def parameter_report(model):
    total=sum(p.numel() for p in model.parameters());trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total":total,"trainable":trainable,"trainable_percent":100*trainable/total}
