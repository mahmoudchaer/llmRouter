from __future__ import annotations
import torch
from .streaming_gradient import microbatches,pad_chunk_batch

def tier_streaming_backward(model,chunks,structural,tier,pad_id,max_tokens,under_route_weight=1.,seed=0):
    """Exact one-request ordinal gradient while retaining only one chunk microbatch's activations."""
    device=next(model.parameters()).device;detached=[];replay=[]
    for i,batch in enumerate(microbatches(chunks,max_tokens)):
        local_seed=seed+i;torch.manual_seed(local_seed);ids,mask=pad_chunk_batch(batch,pad_id,device)
        with torch.no_grad():vectors=model.encode_chunk_batch(ids,mask)
        detached.append(vectors);replay.append((local_seed,batch))
    leaf=torch.cat(detached).detach().requires_grad_(True)
    logits=model.route_chunks(leaf,structural);loss=model.loss(logits,tier.view(1),under_route_weight);loss.backward()
    vector_grad=leaf.grad.detach();offset=0
    for local_seed,batch in replay:
        torch.manual_seed(local_seed);ids,mask=pad_chunk_batch(batch,pad_id,device);vectors=model.encode_chunk_batch(ids,mask)
        grad=vector_grad[offset:offset+len(batch)];torch.autograd.backward(vectors,grad);offset+=len(batch)
    return float(loss.detach())
