from __future__ import annotations

import json, time
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from .embed_requests import QwenEmbedder, token_chunks, cache_fingerprint, save_cache


def batches(entries, token_budget=2048, max_batch=1):
    batch=[]; max_len=0
    for item in sorted(entries,key=lambda x:x[2]):
        proposed=max(max_len,item[2])
        if batch and (len(batch)>=max_batch or proposed*(len(batch)+1)>token_budget):
            yield batch; batch=[]; max_len=0
        batch.append(item); max_len=max(max_len,item[2])
    if batch: yield batch


def run(config_path="configs/baseline1.yaml"):
    cfg=yaml.safe_load(open(config_path)); root=Path(config_path).resolve().parent.parent
    df=pd.read_parquet((root/cfg["data"]["labeled_dataset"]).resolve(),columns=["prompt_id","prompt"])
    emb=QwenEmbedder(cfg["embedding"]); emb.load(); ecfg=cfg["embedding"]
    entries=[]; token_counts=np.zeros(len(df),dtype=np.int32); chunk_counts=np.zeros(len(df),dtype=np.int16)
    for i,text in enumerate(df.prompt):
        ids=emb.tokenize_text(text); token_counts[i]=len(ids)
        chunks=token_chunks(ids,ecfg["chunk_size_tokens"],ecfg["chunk_overlap_tokens"]); chunk_counts[i]=len(chunks)
        for j,c in enumerate(chunks):
            decoded=emb.decode_tokens(c); formatted=emb.task_input(decoded)
            entries.append((len(entries),i,j,len(c)+32,formatted))
    partial=root/"cache"/"embeddings"/"partial"; partial.mkdir(parents=True,exist_ok=True)
    shape=(len(df),ecfg["dimension"])
    sums=np.memmap(partial/"sums.f32",dtype="float32",mode="r+" if (partial/"sums.f32").exists() else "w+",shape=shape)
    seen=np.memmap(partial/"seen.i16",dtype="int16",mode="r+" if (partial/"seen.i16").exists() else "w+",shape=(len(df),))
    processed=np.memmap(partial/"processed.u8",dtype="uint8",mode="r+" if (partial/"processed.u8").exists() else "w+",shape=(len(entries),))
    pending=[x for x in entries if not processed[x[0]]]
    start=time.perf_counter(); done=0
    for batch in batches(pending):
        vec=emb.encode([x[4] for x in batch])
        for item,v in zip(batch,vec): sums[item[1]]+=v; seen[item[1]]+=1; processed[item[0]]=1
        done+=len(batch)
        if done%100< len(batch): sums.flush(); seen.flush(); processed.flush()
        if done%500< len(batch): print(f"new_chunks {done}/{len(pending)} total={int(processed.sum())}/{len(entries)} elapsed={time.perf_counter()-start:.1f}s",flush=True)
    sums.flush(); seen.flush(); processed.flush()
    if not np.array_equal(seen,chunk_counts): raise RuntimeError("Chunk accounting mismatch")
    sums/=seen[:,None]; sums/=np.maximum(np.linalg.norm(sums,axis=1,keepdims=True),1e-12)
    fp=cache_fingerprint(ecfg); out=root/"cache"/"embeddings"/f"qwen3_0.6b_{fp}.npz"
    import torch, transformers
    meta={"model_id":ecfg["model_id"],"revision":ecfg["revision"],"fingerprint":fp,
          "feature_builder_version":cfg["features"]["builder_version"],"runtime":emb.runtime,
          "torch_version":torch.__version__,"transformers_version":transformers.__version__,
          "rows":len(df),"chunks":len(entries),"elapsed_seconds_current_run":time.perf_counter()-start,
          "chunk_size_tokens":ecfg["chunk_size_tokens"],"chunk_overlap_tokens":ecfg["chunk_overlap_tokens"]}
    save_cache(out,df.prompt_id.to_numpy(dtype=str),sums,meta)
    feature_dir=root/"cache"/"features"; feature_dir.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({"prompt_id":df.prompt_id,"token_count":token_counts,"chunk_count":chunk_counts}).to_parquet(
        feature_dir/"request_counts.parquet",index=False)
    print(json.dumps(meta,indent=2))


if __name__=="__main__": run()
