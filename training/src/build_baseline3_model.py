from __future__ import annotations
import torch
from transformers import AutoModel
from peft import LoraConfig,get_peft_model
from .hierarchical_router_model import HierarchicalRouter

def build(cfg,device="cpu",aggregation=None,structural=True):
    # MPS inference supports fp16, but embedding/LayerNorm backward is more stable in fp32.
    dtype=torch.bfloat16 if device=="cuda" else torch.float32
    encoder=AutoModel.from_pretrained(cfg["model"]["model_id"],revision=cfg["model"]["revision"],torch_dtype=dtype)
    if cfg["training"]["gradient_checkpointing"]:encoder.gradient_checkpointing_enable()
    encoder.config.use_cache=False
    if cfg["training"]["mode"]=="lora":
        l=cfg["training"]["lora"]
        encoder=get_peft_model(encoder,LoraConfig(r=l["rank"],lora_alpha=l["alpha"],lora_dropout=l["dropout"],
            target_modules=l["target_modules"],bias="none",task_type="FEATURE_EXTRACTION"))
        if cfg["training"]["gradient_checkpointing"]:encoder.enable_input_require_grads()
    elif cfg["training"]["mode"]!="full":raise ValueError("training.mode must be lora or full")
    model=HierarchicalRouter(encoder,cfg["model"]["hidden_size"],aggregation or cfg["aggregation"]["primary"],
                             cfg["structural_features"]["dimension"] if structural else 0,cfg["model"]["dropout"])
    return model.to(device=device,dtype=dtype)

def parameter_report(model):
    total=sum(p.numel() for p in model.parameters());trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total":total,"trainable":trainable,"trainable_percent":100*trainable/total}
