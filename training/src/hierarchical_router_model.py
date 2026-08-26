from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

def last_non_padding(last_hidden,attention_mask):
    lengths=attention_mask.sum(1)-1
    return last_hidden[torch.arange(last_hidden.size(0),device=last_hidden.device),lengths]

class MeanAggregator(nn.Module):
    def forward(self,chunks,mask=None):
        if mask is None:return chunks.mean(0)
        w=mask.to(chunks.dtype).unsqueeze(-1);return (chunks*w).sum(0)/w.sum(0).clamp_min(1)

class AttentionAggregator(nn.Module):
    def __init__(self,dim,hidden=128):
        super().__init__();self.score=nn.Sequential(nn.Linear(dim,hidden),nn.Tanh(),nn.Linear(hidden,1))
    def forward(self,chunks,mask=None):
        logits=self.score(chunks).squeeze(-1)
        if mask is not None:logits=logits.masked_fill(~mask,bool("-inf"))
        return (torch.softmax(logits,dim=0).unsqueeze(-1)*chunks).sum(0)

class HierarchicalRouter(nn.Module):
    def __init__(self,encoder,hidden_size=1024,aggregation="mean",structural_dim=15,dropout=.1):
        super().__init__();self.encoder=encoder;self.structural_dim=structural_dim
        self.aggregator=MeanAggregator() if aggregation=="mean" else AttentionAggregator(hidden_size)
        routing_dim=hidden_size+structural_dim
        self.request_norm=nn.LayerNorm(routing_dim);self.dropout=nn.Dropout(dropout)
        self.domain_head=nn.Linear(routing_dim,7);self.tier_head=nn.Linear(routing_dim,4)

    def encode_chunk_batch(self,input_ids,attention_mask):
        out=self.encoder(input_ids=input_ids,attention_mask=attention_mask,use_cache=False)
        # Qwen3Model output is after its final RMSNorm; deliberately no L2 normalization.
        return last_non_padding(out.last_hidden_state,attention_mask)

    def forward_from_chunk_embeddings(self,chunks,structural=None):
        request=self.aggregator(chunks)
        if self.structural_dim:
            if structural is None:structural=torch.zeros(self.structural_dim,device=request.device,dtype=request.dtype)
            request=torch.cat([request,structural.to(request.dtype)])
        shared=self.dropout(self.request_norm(request))
        return {"domain_logits":self.domain_head(shared),"tier_logits":self.tier_head(shared),"request_vector":shared}

    def forward_request_batch(self,request_vectors,structural=None):
        if self.structural_dim:
            if structural is None:
                structural=torch.zeros((request_vectors.size(0),self.structural_dim),device=request_vectors.device,dtype=request_vectors.dtype)
            request_vectors=torch.cat([request_vectors,structural.to(request_vectors.dtype)],dim=1)
        shared=self.dropout(self.request_norm(request_vectors))
        return {"domain_logits":self.domain_head(shared),"tier_logits":self.tier_head(shared),"request_vector":shared}

def multitask_loss(outputs,domain_label,tier_label,domain_weight=1.,tier_weight=1.):
    domain=F.cross_entropy(outputs["domain_logits"].unsqueeze(0),domain_label.view(1))
    resolved=tier_label.ge(0)
    tier=F.cross_entropy(outputs["tier_logits"].unsqueeze(0),tier_label.clamp_min(0).view(1)) if resolved else domain.new_zeros(())
    return {"loss":domain_weight*domain+tier_weight*tier,"domain_loss":domain,"tier_loss":tier,"tier_mask":resolved}

def batched_multitask_loss(outputs,domain_labels,tier_labels,domain_weight=1.,tier_weight=1.):
    domain=F.cross_entropy(outputs["domain_logits"],domain_labels)
    resolved=tier_labels.ge(0)
    tier=F.cross_entropy(outputs["tier_logits"][resolved],tier_labels[resolved]) if resolved.any() else domain.new_zeros(())
    return {"loss":domain_weight*domain+tier_weight*tier,"domain_loss":domain,"tier_loss":tier,"tier_mask":resolved}
