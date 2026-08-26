from __future__ import annotations
import numpy as np

class TokenBucketBatchSampler:
    def __init__(self,lengths,max_tokens,seed=0,bucket_size=256):self.lengths=np.asarray(lengths);self.max_tokens=max_tokens;self.seed=seed;self.bucket_size=bucket_size
    def __iter__(self):
        rng=np.random.default_rng(self.seed);order=np.argsort(self.lengths)
        buckets=[order[i:i+self.bucket_size].copy() for i in range(0,len(order),self.bucket_size)]
        for b in buckets:rng.shuffle(b)
        rng.shuffle(buckets);batch=[];tokens=0
        for i in np.concatenate(buckets):
            n=int(self.lengths[i])
            if batch and tokens+n>self.max_tokens:yield batch;batch=[];tokens=0
            batch.append(int(i));tokens+=n
        if batch:yield batch

