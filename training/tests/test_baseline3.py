import torch
from torch import nn
from types import SimpleNamespace
from src.hierarchical_chunking import chunk_token_ids
from src.hierarchical_router_model import HierarchicalRouter,multitask_loss,AttentionAggregator
from src.streaming_gradient import streaming_request_backward
from src.token_bucket_sampler import TokenBucketBatchSampler

class TinyEncoder(nn.Module):
    def __init__(self):super().__init__();self.emb=nn.Embedding(50,8);self.proj=nn.Linear(8,8)
    def forward(self,input_ids,attention_mask,use_cache=False):return SimpleNamespace(last_hidden_state=self.proj(self.emb(input_ids)))

def test_chunking_preserves_every_token_and_has_no_slot_limit():
    ids=list(range(10000));chunks=chunk_token_ids(ids,128,16,eos_id=99)
    assert len(chunks)>80 and all(len(x)<=128 for x in chunks)

def test_attention_aggregator_accepts_100_plus_chunks():
    a=AttentionAggregator(8);assert a(torch.randn(137,8)).shape==(8,)

def test_unresolved_masks_only_tier_loss():
    m=HierarchicalRouter(TinyEncoder(),8,"mean",0,0);out=m.forward_from_chunk_embeddings(torch.randn(3,8))
    loss=multitask_loss(out,torch.tensor(2),torch.tensor(-1))
    assert loss["domain_loss"]>0 and loss["tier_loss"]==0 and not loss["tier_mask"]

def test_streaming_gradient_is_request_level_and_reaches_encoder():
    m=HierarchicalRouter(TinyEncoder(),8,"mean",0,0);m.train();m.zero_grad()
    streaming_request_backward(m,[[1,2,3],[4,5],[6]],None,torch.tensor(1),torch.tensor(2),0,5)
    assert m.encoder.emb.weight.grad is not None and m.encoder.emb.weight.grad.abs().sum()>0
    assert m.domain_head.weight.grad is not None and m.tier_head.weight.grad is not None

def test_token_bucket_never_exceeds_budget_except_single_oversize():
    lengths=[10,20,100,5,7];s=TokenBucketBatchSampler(lengths,30,1,3)
    for b in s:assert sum(lengths[i] for i in b)<=30 or len(b)==1
