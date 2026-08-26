from __future__ import annotations
import torch
from .hierarchical_router_model import multitask_loss

def pad_chunk_batch(chunks,pad_id,device):
    n=max(map(len,chunks));ids=torch.full((len(chunks),n),pad_id,dtype=torch.long,device=device);mask=torch.zeros_like(ids)
    for i,c in enumerate(chunks):ids[i,:len(c)]=torch.tensor(c,device=device);mask[i,:len(c)]=1
    return ids,mask

def microbatches(chunks,max_tokens):
    batch=[];mx=0
    for c in chunks:
        if batch and max(mx,len(c))*(len(batch)+1)>max_tokens:yield batch;batch=[];mx=0
        batch.append(c);mx=max(mx,len(c))
    if batch:yield batch

def streaming_request_backward(model,chunks,structural,domain_label,tier_label,pad_id,max_tokens,loss_scale=1.,seed=0):
    """Exact request-level gradient via detached-vector loss then encoder VJP replay."""
    device=next(model.parameters()).device;detached=[];seeds=[]
    for i,batch in enumerate(microbatches(chunks,max_tokens)):
        local_seed=seed+i;torch.manual_seed(local_seed);ids,mask=pad_chunk_batch(batch,pad_id,device)
        with torch.no_grad():vec=model.encode_chunk_batch(ids,mask)
        detached.append(vec);seeds.append((local_seed,batch))
    leaf=torch.cat(detached).detach().requires_grad_(True)
    outputs=model.forward_from_chunk_embeddings(leaf,structural)
    losses=multitask_loss(outputs,domain_label,tier_label);(losses["loss"]*loss_scale).backward()
    vector_grad=leaf.grad.detach();offset=0
    for local_seed,batch in seeds:
        torch.manual_seed(local_seed);ids,mask=pad_chunk_batch(batch,pad_id,device);vec=model.encode_chunk_batch(ids,mask)
        g=vector_grad[offset:offset+len(batch)];torch.autograd.backward(vec,g);offset+=len(batch)
    return {k:(float(v.detach()) if v.ndim==0 else v) for k,v in losses.items()}

