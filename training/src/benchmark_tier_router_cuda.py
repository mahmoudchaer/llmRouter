from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np,pandas as pd,torch,yaml
from transformers import AutoTokenizer
from .build_tier_router import build_tier_router,parameter_report
from .hierarchical_chunking import chunk_token_ids
from .streaming_gradient import pad_chunk_batch
from .tier_streaming_gradient import tier_streaming_backward

def batches(rows,budget):
    rows=sorted(rows,key=lambda r:len(r["chunks"][0]));batch=[];longest=0
    for r in rows:
        n=len(r["chunks"][0])
        if batch and max(longest,n)*(len(batch)+1)>budget:yield batch;batch=[];longest=0
        batch.append(r);longest=max(longest,n)
    if batch:yield batch

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--budget",type=int,required=True);ap.add_argument("--steps",type=int,default=12);ap.add_argument("--output",required=True);args=ap.parse_args()
    cfg=yaml.safe_load(open("configs/tier_router_v1.yaml"));tok=AutoTokenizer.from_pretrained(cfg["model"]["model_id"],revision=cfg["model"]["revision"])
    df=pd.read_parquet("data/final_labeled_dataset.parquet");split=json.load(open("splits/grouped_split_17.json"))["datasets"]
    df=df[df.resolved & df.source_dataset.map(split).eq("train")].copy();eos=tok.eos_token_id;rows=[]
    for row in df.itertuples():
        ids=tok.encode(str(row.prompt),add_special_tokens=False);cs=chunk_token_ids(ids,2048,128,eos)
        rows.append({"chunks":cs,"tier":int(row.tier)-1,"structural":np.zeros(15,np.float32)})
    single=[r for r in rows if len(r["chunks"])==1];multi=max(rows,key=lambda r:sum(map(len,r["chunks"])))
    # Quantile-stratified ordering keeps benchmark batches representative by length.
    picks=[];ordered=sorted(single,key=lambda r:len(r["chunks"][0]));stride=max(1,len(ordered)//2048)
    for i in range(0,len(ordered),stride):picks.append(ordered[i])
    model=build_tier_router(cfg,"cuda","ordinal",True);model.train();optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=2e-5)
    pad=tok.pad_token_id or eos;torch.cuda.reset_peak_memory_stats();total_real=total_padded=0;started=time.perf_counter();used=0
    for batch in batches(picks,args.budget):
        if used>=args.steps:break
        ids,mask=pad_chunk_batch([r["chunks"][0] for r in batch],pad,"cuda");vectors=model.encode_chunk_batch(ids,mask);struct=torch.zeros((len(batch),15),device="cuda",dtype=vectors.dtype)
        logits=model.route(vectors,struct);tiers=torch.tensor([r["tier"] for r in batch],device="cuda");loss=model.loss(logits,tiers,2.);loss.backward();optimizer.step();optimizer.zero_grad(set_to_none=True)
        total_real+=int(mask.sum());total_padded+=mask.numel();used+=1
    torch.cuda.synchronize();elapsed=time.perf_counter()-started
    # Verify the longest training request through the dedicated request-level replay path.
    struct=torch.zeros(15,device="cuda",dtype=next(model.parameters()).dtype);optimizer.zero_grad(set_to_none=True);long_start=time.perf_counter()
    long_loss=tier_streaming_backward(model,multi["chunks"],struct,torch.tensor(multi["tier"],device="cuda"),pad,2048,2.,991);torch.cuda.synchronize();long_elapsed=time.perf_counter()-long_start
    report={"budget":args.budget,"steps":used,"parameters":parameter_report(model),"batch_elapsed_seconds":elapsed,"real_tokens_per_second":total_real/elapsed,"padded_tokens_per_second":total_padded/elapsed,"padding_efficiency":total_real/total_padded,"peak_vram_bytes":torch.cuda.max_memory_allocated(),"longest_train_request_chunks":len(multi["chunks"]),"longest_request_seconds":long_elapsed,"longest_request_loss":long_loss}
    Path(args.output).parent.mkdir(parents=True,exist_ok=True);Path(args.output).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=="__main__":main()
