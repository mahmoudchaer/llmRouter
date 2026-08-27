import torch
from src.tier_ordinal import MonotonicOrdinalHead,tiers_to_cumulative,cumulative_ordinal_loss,cumulative_probabilities
from src.tier_router_model import TierRouter
from src.tier_streaming_gradient import tier_streaming_backward

def test_cumulative_targets():
    got=tiers_to_cumulative(torch.tensor([0,1,2,3]))
    assert got.tolist()==[[0,0,0],[1,0,0],[1,1,0],[1,1,1]]

def test_three_tier_cumulative_targets():
    got=tiers_to_cumulative(torch.tensor([0,1,2]),3)
    assert got.tolist()==[[0,0],[1,0],[1,1]]

def test_monotonic_logits_and_valid_class_probabilities():
    head=MonotonicOrdinalHead(5);logits=head(torch.randn(12,5))
    assert torch.all(logits[:,0]>=logits[:,1]) and torch.all(logits[:,1]>=logits[:,2])
    p=cumulative_probabilities(logits)
    assert torch.all(p>=0) and torch.allclose(p.sum(1),torch.ones(12),atol=1e-6)

def test_three_tier_ordered_cutpoints_and_probabilities():
    head=MonotonicOrdinalHead(5,3);logits=head(torch.randn(12,5))
    assert logits.shape==(12,2) and torch.all(logits[:,0]>=logits[:,1])
    p=cumulative_probabilities(logits)
    assert p.shape==(12,3) and torch.all(p>=0) and torch.allclose(p.sum(1),torch.ones(12),atol=1e-6)

def test_under_route_weight_increases_false_negative_loss():
    logits=torch.tensor([[-3.,-3.,-3.]])
    tier4=torch.tensor([3])
    assert cumulative_ordinal_loss(logits,tier4,3)>cumulative_ordinal_loss(logits,tier4,1)

def test_three_tier_under_route_weight_increases_false_negative_loss():
    logits=torch.tensor([[-3.,-3.]])
    tier3=torch.tensor([2])
    assert cumulative_ordinal_loss(logits,tier3,3)>cumulative_ordinal_loss(logits,tier3,1)

class ToyEncoder(torch.nn.Module):
    def __init__(self):super().__init__();self.emb=torch.nn.Embedding(20,8)
    def forward(self,input_ids,attention_mask,use_cache=False):
        return type("O",(),{"last_hidden_state":self.emb(input_ids)})

def test_request_level_ordinal_gradient_reaches_encoder():
    model=TierRouter(ToyEncoder(),8,"ordinal",0,0)
    chunks=[]
    for ids in [torch.tensor([[1,2,3]]),torch.tensor([[4,5,6]])]:
        chunks.append(model.encode_chunk_batch(ids,torch.ones_like(ids)))
    logits=model.route_chunks(torch.cat(chunks));model.loss(logits,torch.tensor([3]),3).backward()
    assert model.encoder.emb.weight.grad.abs().sum()>0

def test_streaming_ordinal_gradient_reaches_encoder():
    model=TierRouter(ToyEncoder(),8,"ordinal",0,0)
    loss=tier_streaming_backward(model,[[1,2,3],[4,5,6]],None,torch.tensor(3),0,3,3.,7)
    assert loss>0 and model.encoder.emb.weight.grad.abs().sum()>0
