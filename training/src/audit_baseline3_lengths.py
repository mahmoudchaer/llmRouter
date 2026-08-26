from __future__ import annotations
import json
from pathlib import Path
import numpy as np,pandas as pd
from .hierarchical_chunking import chunk_count

def run():
    root=Path(__file__).resolve().parent.parent;x=pd.read_parquet(root/"cache"/"features"/"request_counts.parquet");t=x.token_count.to_numpy()
    candidates={1024:128,2048:128,4096:256,8192:256};report={"token_quantiles":{str(q):float(np.quantile(t,q)) for q in (.5,.75,.9,.95,.99,.999,1)},"candidates":{}}
    for size,overlap in candidates.items():
        c=np.array([chunk_count(int(n),size,overlap) for n in t])
        report["candidates"][str(size)]={"overlap":overlap,"average_chunks":float(c.mean()),"p95_chunks":float(np.quantile(c,.95)),
          "p99_chunks":float(np.quantile(c,.99)),"maximum_chunks":int(c.max()),"multi_chunk_count":int((c>1).sum()),
          "multi_chunk_percent":float(100*(c>1).mean()),"total_chunks":int(c.sum())}
    (root/"reports"/"baseline3_length_chunk_audit.json").write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=="__main__":run()

