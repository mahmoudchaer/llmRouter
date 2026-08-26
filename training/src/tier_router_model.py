from __future__ import annotations
import torch
from torch import nn
from .hierarchical_router_model import last_non_padding,MeanAggregator
from .tier_ordinal import MonotonicOrdinalHead,cumulative_ordinal_loss

class TierRouter(nn.Module):
    def __init__(self,encoder,hidden_size,loss_kind="ordinal",structural_dim=15,dropout=.1):
        super().__init__();self.encoder=encoder;self.structural_dim=structural_dim;self.aggregator=MeanAggregator();dim=hidden_size+structural_dim
        self.request_norm=nn.LayerNorm(dim);self.dropout=nn.Dropout(dropout);self.loss_kind=loss_kind
        self.tier_head=MonotonicOrdinalHead(dim) if loss_kind=="ordinal" else nn.Linear(dim,4)
    def encode_chunk_batch(self,input_ids,attention_mask):
        out=self.encoder(input_ids=input_ids,attention_mask=attention_mask,use_cache=False)
        return last_non_padding(out.last_hidden_state,attention_mask)
    def route(self,request_vectors,structural=None):
        if request_vectors.ndim==1:request_vectors=request_vectors.unsqueeze(0)
        if self.structural_dim:
            if structural is None:structural=torch.zeros((len(request_vectors),self.structural_dim),device=request_vectors.device,dtype=request_vectors.dtype)
            if structural.ndim==1:structural=structural.unsqueeze(0)
            request_vectors=torch.cat([request_vectors,structural.to(request_vectors.dtype)],1)
        return self.tier_head(self.dropout(self.request_norm(request_vectors)))
    def route_chunks(self,chunks,structural=None):return self.route(self.aggregator(chunks),structural)
    def loss(self,logits,tiers,under_route_weight=1.):
        return cumulative_ordinal_loss(logits,tiers,under_route_weight) if self.loss_kind=="ordinal" else nn.functional.cross_entropy(logits,tiers)
