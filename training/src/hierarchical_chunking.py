from __future__ import annotations
import math

def chunk_token_ids(ids:list[int],chunk_size:int,overlap:int,eos_id:int|None=None):
    reserve=1 if eos_id is not None else 0;capacity=chunk_size-reserve
    if capacity<=overlap or overlap<0:raise ValueError("chunk capacity must exceed overlap")
    if not ids:ids=[]
    step=capacity-overlap;chunks=[]
    for start in range(0,max(len(ids),1),step):
        content=ids[start:start+capacity];chunk=content+([eos_id] if eos_id is not None else [])
        chunks.append(chunk)
        if start+capacity>=len(ids):break
    covered=set()
    for start in range(0,max(len(ids),1),step):
        covered.update(range(start,min(start+capacity,len(ids))))
        if start+capacity>=len(ids):break
    if covered!=set(range(len(ids))):raise RuntimeError("Chunking dropped tokens")
    return chunks

def chunk_count(n_tokens,chunk_size,overlap,reserve=1):
    capacity=chunk_size-reserve;return max(1,math.ceil(max(n_tokens-overlap,1)/(capacity-overlap)))

