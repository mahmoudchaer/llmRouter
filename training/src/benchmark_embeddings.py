from __future__ import annotations

import json, resource, time
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from .embed_requests import QwenEmbedder, token_chunks


def run(config_path="configs/baseline1.yaml", sample_size=200):
    cfg=yaml.safe_load(open(config_path)); root=Path(config_path).resolve().parent.parent
    data=(root/cfg["data"]["labeled_dataset"]).resolve()
    df=pd.read_parquet(data,columns=["prompt_id","prompt"])
    embedder=QwenEmbedder(cfg["embedding"]); embedder.load()
    token_ids=[]
    for text in df.prompt: token_ids.append(embedder.tokenize_text(text))
    lengths=np.array(list(map(len,token_ids)))
    order=np.argsort(lengths)
    # 80% short, 15% medium, 5% longest; deterministic and length-representative.
    rng=np.random.default_rng(20260826)
    short=order[:int(.90*len(order))]; medium=order[int(.90*len(order)):int(.99*len(order))]; longest=order[int(.99*len(order)):]
    counts=(int(.80*sample_size),int(.15*sample_size)); counts+=(sample_size-sum(counts),)
    selected=np.concatenate([rng.choice(pool,min(n,len(pool)),replace=False) for pool,n in zip((short,medium,longest),counts)])
    records=[]; start=time.perf_counter(); total_tokens=0; failures=[]
    for rank,i in enumerate(selected):
        ids=token_ids[i]
        chunks=token_chunks(ids,cfg["embedding"]["chunk_size_tokens"],cfg["embedding"]["chunk_overlap_tokens"])
        cstart=time.perf_counter(); ok=True; error=None
        try:
            texts=[embedder.task_input(embedder.decode_tokens(c)) for c in chunks]
            vectors=[]
            for text in texts: vectors.append(embedder.encode([text])[0])
            vector=np.mean(vectors,axis=0); vector/=max(np.linalg.norm(vector),1e-12)
        except Exception as exc:
            ok=False; error=f"{type(exc).__name__}: {exc}"; failures.append(error)
        elapsed=time.perf_counter()-cstart; total_tokens+=len(ids)
        records.append({"prompt_id":df.prompt_id.iloc[i],"tokens":len(ids),"chunks":len(chunks),"seconds":elapsed,"ok":ok,"error":error})
        print(f"{rank+1}/{len(selected)} tokens={len(ids)} chunks={len(chunks)} seconds={elapsed:.2f} ok={ok}",flush=True)
    elapsed=time.perf_counter()-start
    report={"sample_size":len(selected),"sampling":{"short":counts[0],"medium":counts[1],"longest":counts[2]},
            "device":embedder.runtime,"elapsed_seconds":elapsed,"prompts_per_second":len(selected)/elapsed,
            "tokens_per_second":total_tokens/elapsed,"total_tokens":total_tokens,"failures":len(failures),
            "peak_process_rss_mb":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024/1024,
            "token_length_quantiles":{str(q):float(np.quantile(lengths,q)) for q in (0,.5,.9,.95,.99,.999,1)},
            "multi_chunk_total":int(sum(len(token_chunks(x,cfg['embedding']['chunk_size_tokens'],cfg['embedding']['chunk_overlap_tokens']))>1 for x in token_ids)),
            "records":records}
    out=root/"reports"/"baseline1_embedding_benchmark.json"; out.write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!="records"},indent=2))


if __name__=="__main__": run()
