from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

def tiers_to_cumulative(tiers_zero_based):
    thresholds=torch.arange(1,4,device=tiers_zero_based.device)
    return (tiers_zero_based.unsqueeze(1)>=thresholds).to(torch.float32)

class MonotonicOrdinalHead(nn.Module):
    """CORAL-style shared latent score with strictly ordered learned cutpoints."""
    def __init__(self,input_dim):
        super().__init__();self.score=nn.Linear(input_dim,1);self.first_cutpoint=nn.Parameter(torch.tensor(-1.0));self.cutpoint_deltas=nn.Parameter(torch.tensor([0.5,0.5]))
    def cutpoints(self):
        deltas=F.softplus(self.cutpoint_deltas)+1e-4
        return torch.cat([self.first_cutpoint.view(1),self.first_cutpoint+deltas[:1],self.first_cutpoint+deltas.cumsum(0)[-1:]])
    def forward(self,x):return self.score(x)-self.cutpoints()

def cumulative_ordinal_loss(logits,tiers_zero_based,under_route_weight=1.0):
    targets=tiers_to_cumulative(tiers_zero_based).to(logits.dtype)
    element=F.binary_cross_entropy_with_logits(logits,targets,reduction="none")
    weights=torch.where(targets.bool(),torch.as_tensor(under_route_weight,device=logits.device,dtype=logits.dtype),torch.ones_like(element))
    # Keep the denominator fixed so increasing the positive/false-negative cost
    # actually increases under-routing gradients instead of cancelling itself.
    return (element*weights).mean()

def cumulative_probabilities(logits):
    survival=torch.sigmoid(logits)
    return torch.stack([1-survival[:,0],survival[:,0]-survival[:,1],survival[:,1]-survival[:,2],survival[:,2]],dim=1)

def ordinal_prediction(logits,threshold=.5):return 1+(torch.sigmoid(logits)>=threshold).sum(1)

def routing_cost(true_tier,pred_tier,cost_matrix):
    matrix=torch.as_tensor(cost_matrix,device=true_tier.device,dtype=torch.float32)
    return matrix[true_tier-1,pred_tier-1]/matrix.max()

def selection_score(macro_f1,mae,under_routing,severe_under_routing):
    return .35*(mae/3)+.35*severe_under_routing+.20*under_routing+.10*(1-macro_f1)
