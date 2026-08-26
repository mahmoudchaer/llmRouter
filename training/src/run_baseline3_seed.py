from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import time
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .build_baseline3_model import build, parameter_report
from .calibration_analysis import confidence_bins
from .evaluate import domain_metrics, tier_metrics
from .hierarchical_chunking import chunk_token_ids
from .hierarchical_router_model import batched_multitask_loss
from .streaming_gradient import microbatches, pad_chunk_batch, streaming_request_backward

DOMAIN_CLASSES = ["affective", "code", "instruction_following", "knowledge", "logic", "math", "tool_use"]


def jsonable(value):
    if isinstance(value, dict): return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [jsonable(v) for v in value]
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return float(value)
    if isinstance(value, np.ndarray): return value.tolist()
    return value


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def gpu_utilization():
    try:
        out = subprocess.check_output([
            "nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader,nounits"], text=True).strip().split(",")
        return {"gpu_utilization_percent": float(out[0]), "gpu_memory_used_mib": float(out[1])}
    except Exception:
        return {"gpu_utilization_percent": None, "gpu_memory_used_mib": None}


def structural_raw(token_count, char_count, chunk_count):
    return np.asarray([
        math.log1p(token_count), math.log1p(char_count), math.log1p(1), math.log1p(chunk_count),
        0, math.log1p(token_count), 0, 0, 0, 0, 0, 0, 1, 0, 0,
    ], dtype=np.float32)


def prepare(df, tokenizer, split, chunk_size, overlap, cache_path=None):
    if cache_path and Path(cache_path).exists():
        with Path(cache_path).open("rb") as f:return pickle.load(f)
    df = df.copy()
    df["partition"] = df.source_dataset.map(split)
    if df.partition.isna().any(): raise ValueError("Some source datasets are absent from split definition")
    df["semantic_domain"] = df.domain.replace({"logic_arc_agi": "logic", "logic_classic": "logic"})
    unknown = set(df.semantic_domain) - set(DOMAIN_CLASSES)
    if unknown: raise ValueError(f"Unknown domains: {unknown}")
    token_ids = [tokenizer.encode(str(x), add_special_tokens=False) for x in df.prompt]
    eos = tokenizer.eos_token_id
    chunks = [chunk_token_ids(ids, chunk_size, overlap, eos) for ids in token_ids]
    raw_struct = np.stack([structural_raw(len(ids), len(str(text)), len(cs)) for ids, text, cs in zip(token_ids, df.prompt, chunks)])
    scaler = StandardScaler().fit(raw_struct[df.partition.to_numpy() == "train"])
    scaled = scaler.transform(raw_struct).astype(np.float32)
    domain_to_id = {x: i for i, x in enumerate(DOMAIN_CLASSES)}
    records = []
    for pos, (_, row) in enumerate(df.iterrows()):
        records.append({
            "prompt_id": str(row.prompt_id), "source_dataset": str(row.source_dataset),
            "partition": row.partition, "chunks": chunks[pos], "structural": scaled[pos],
            "domain": domain_to_id[row.semantic_domain],
            "tier": int(row.tier) - 1 if bool(row.resolved) else -1,
            "resolved": bool(row.resolved), "raw_tokens": len(token_ids[pos]),
        })
    result=(records,scaler)
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True,exist_ok=True)
        with Path(cache_path).open("wb") as f:pickle.dump(result,f,pickle.HIGHEST_PROTOCOL)
    return result


def single_chunk_batches(records,max_padded_tokens,seed):
    rng=random.Random(seed);ordered=sorted(records,key=lambda r:len(r["chunks"][0]))
    buckets=[ordered[i:i+512] for i in range(0,len(ordered),512)]
    for bucket in buckets:rng.shuffle(bucket)
    rng.shuffle(buckets);batch=[];longest=0
    for record in [r for bucket in buckets for r in bucket]:
        length=len(record["chunks"][0])
        if batch and max(longest,length)*(len(batch)+1)>max_padded_tokens:
            yield batch;batch=[];longest=0
        batch.append(record);longest=max(longest,length)
    if batch:yield batch


def train_single_batch(model,batch,pad_id,device):
    ids,mask=pad_chunk_batch([r["chunks"][0] for r in batch],pad_id,device)
    vectors=model.encode_chunk_batch(ids,mask)
    structural=torch.tensor(np.stack([r["structural"] for r in batch]),device=device,dtype=vectors.dtype)
    outputs=model.forward_request_batch(vectors,structural)
    domains=torch.tensor([r["domain"] for r in batch],device=device)
    tiers=torch.tensor([r["tier"] for r in batch],device=device)
    losses=batched_multitask_loss(outputs,domains,tiers);losses["loss"].backward()
    return {k:(float(v.detach()) if v.ndim==0 else v) for k,v in losses.items()}


@torch.no_grad()
def predict_one(model, record, pad_id, max_tokens, device):
    vectors = []
    for batch in microbatches(record["chunks"], max_tokens):
        ids, mask = pad_chunk_batch(batch, pad_id, device)
        vectors.append(model.encode_chunk_batch(ids, mask))
    vectors = torch.cat(vectors)
    structural = torch.tensor(record["structural"], device=device, dtype=next(model.parameters()).dtype)
    out = model.forward_from_chunk_embeddings(vectors, structural)
    return out["domain_logits"].float().cpu(), out["tier_logits"].float().cpu()


def evaluate_partition(model, records, pad_id, max_tokens, device, include_predictions=False):
    model.eval(); domain_y=[]; domain_logits=[]; tier_y=[]; tier_logits=[]; prediction_rows=[]
    domain_loss_sum=tier_loss_sum=0.0; domain_n=tier_n=0
    started=time.perf_counter()
    for r in records:
        dl, tl = predict_one(model, r, pad_id, max_tokens, device)
        domain_y.append(r["domain"]); domain_logits.append(dl.numpy())
        domain_loss_sum += torch.nn.functional.cross_entropy(dl[None], torch.tensor([r["domain"]])).item(); domain_n += 1
        if r["resolved"]:
            tier_y.append(r["tier"] + 1); tier_logits.append(tl.numpy())
            tier_loss_sum += torch.nn.functional.cross_entropy(tl[None], torch.tensor([r["tier"]])).item(); tier_n += 1
        if include_predictions:
            prediction_rows.append({"prompt_id":r["prompt_id"],"source_dataset":r["source_dataset"],"true_domain":r["domain"],"true_tier":r["tier"]+1 if r["resolved"] else None,"domain_logits":dl.numpy(),"tier_logits":tl.numpy()})
    dp=torch.softmax(torch.tensor(np.stack(domain_logits)),dim=1).numpy(); dy=np.asarray(domain_y); d_pred=dp.argmax(1)
    result={"validation_loss":domain_loss_sum/domain_n + (tier_loss_sum/tier_n if tier_n else 0),
            "domain_loss":domain_loss_sum/domain_n,"tier_loss":tier_loss_sum/tier_n if tier_n else None,
            "domain_accuracy":float(accuracy_score(dy,d_pred)),"domain_macro_f1":float(f1_score(dy,d_pred,average="macro",zero_division=0)),
            "seconds":time.perf_counter()-started,"n":len(records),"tier_n":tier_n}
    if tier_n:
        tp=torch.softmax(torch.tensor(np.stack(tier_logits)),dim=1).numpy(); ty=np.asarray(tier_y); t_pred=tp.argmax(1)+1
        result.update({"tier_exact_accuracy":float(accuracy_score(ty,t_pred)),"tier_macro_f1":float(f1_score(ty,t_pred,average="macro",zero_division=0)),
                       "tier_mae":float(np.abs(ty-t_pred).mean()),"under_routing":float((t_pred<ty).mean()),
                       "severe_under_routing":float(((ty-t_pred)>=2).mean())})
    return result, prediction_rows


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, step, best_metric, cfg):
    path.mkdir(parents=True, exist_ok=True)
    torch.save({"peft":get_peft_model_state_dict(model.encoder),
                "request_norm":model.request_norm.state_dict(),"domain_head":model.domain_head.state_dict(),
                "tier_head":model.tier_head.state_dict(),"aggregator":model.aggregator.state_dict()}, path/"model.pt")
    torch.save({"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),"epoch":epoch,
                "step":step,"best_metric":best_metric}, path/"trainer_state.pt")
    np.savez(path/"structural_scaler.npz", mean=scaler.mean_, scale=scaler.scale_, var=scaler.var_)
    (path/"training_config.json").write_text(json.dumps(jsonable(cfg),indent=2))


def load_model_checkpoint(path, model):
    state=torch.load(path/"model.pt",map_location="cpu",weights_only=True)
    set_peft_model_state_dict(model.encoder,state["peft"])
    model.request_norm.load_state_dict(state["request_norm"]);model.domain_head.load_state_dict(state["domain_head"])
    model.tier_head.load_state_dict(state["tier_head"]);model.aggregator.load_state_dict(state["aggregator"])


def final_test_report(model, records, pad_id, max_tokens, device, output_dir):
    metrics, rows=evaluate_partition(model,records,pad_id,max_tokens,device,include_predictions=True)
    dy=np.asarray([r["true_domain"] for r in rows]); dl=np.stack([r["domain_logits"] for r in rows]); dp=torch.softmax(torch.tensor(dl),1).numpy(); d_pred=dp.argmax(1)
    resolved=[r for r in rows if r["true_tier"] is not None]; ty=np.asarray([r["true_tier"] for r in resolved]); tl=np.stack([r["tier_logits"] for r in resolved]); tp=torch.softmax(torch.tensor(tl),1).numpy(); t_pred=tp.argmax(1)+1
    report={"summary":metrics,"domain":domain_metrics(dy,d_pred,dp,list(range(7))),"domain_confidence_bins":confidence_bins(dy,d_pred,dp)}
    if len(ty): report.update({"tier":tier_metrics(ty,t_pred,tp),"tier_confidence_bins":confidence_bins(ty,t_pred,tp)})
    pred=pd.DataFrame({"prompt_id":[r["prompt_id"] for r in rows],"source_dataset":[r["source_dataset"] for r in rows],
                       "true_domain":dy,"predicted_domain":d_pred,"domain_probabilities":[x.tolist() for x in dp]})
    tier_map={r["prompt_id"]:(r["true_tier"],int(p),probs.tolist()) for r,p,probs in zip(resolved,t_pred,tp)}
    pred["true_tier"]=pred.prompt_id.map(lambda x:tier_map.get(x,(None,None,None))[0]); pred["predicted_tier"]=pred.prompt_id.map(lambda x:tier_map.get(x,(None,None,None))[1]); pred["tier_probabilities"]=pred.prompt_id.map(lambda x:tier_map.get(x,(None,None,None))[2])
    pred.to_parquet(output_dir/"seed17_test_predictions.parquet",index=False)
    (output_dir/"seed17_test_metrics.json").write_text(json.dumps(jsonable(report),indent=2))
    return report


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--config",default="configs/baseline3.yaml");ap.add_argument("--data",default="data/final_labeled_dataset.parquet")
    ap.add_argument("--split",default="splits/grouped_split_17.json");ap.add_argument("--output",default="output/baseline3a_seed17")
    ap.add_argument("--max-train",type=int);ap.add_argument("--max-validation",type=int);ap.add_argument("--epochs",type=int);ap.add_argument("--skip-test",action="store_true")
    ap.add_argument("--batch-token-budget",type=int,default=32768);ap.add_argument("--token-cache",default="cache/baseline3_tokens_seed17.pkl")
    args=ap.parse_args();cfg=yaml.safe_load(Path(args.config).read_text());seed=17;seed_everything(seed);device="cuda"
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True);(output/"logs").mkdir(exist_ok=True)
    tokenizer=AutoTokenizer.from_pretrained(cfg["model"]["model_id"],revision=cfg["model"]["revision"])
    df=pd.read_parquet(args.data);split=json.loads(Path(args.split).read_text())["datasets"]
    records,scaler=prepare(df,tokenizer,split,2048,128,args.token_cache);train=[r for r in records if r["partition"]=="train"];val=[r for r in records if r["partition"]=="validation"];test=[r for r in records if r["partition"]=="test"]
    if args.max_train:train=train[:args.max_train]
    if args.max_validation:val=val[:args.max_validation]
    model=build(cfg,device=device,aggregation="mean",structural=True);model.train();print(json.dumps(parameter_report(model)),flush=True)
    encoder_params=[];head_params=[]
    for name,p in model.named_parameters():
        if p.requires_grad:(encoder_params if name.startswith("encoder.") else head_params).append(p)
    optimizer=torch.optim.AdamW([{"params":encoder_params,"lr":cfg["training"]["encoder_learning_rate"]},{"params":head_params,"lr":cfg["training"]["head_learning_rate"]}],weight_decay=cfg["training"]["weight_decay"])
    epochs=args.epochs or cfg["training"]["epochs"]
    single=[r for r in train if len(r["chunks"])==1];multi=[r for r in train if len(r["chunks"])>1]
    steps_per_epoch=len(list(single_chunk_batches(single,args.batch_token_budget,seed)))+len(multi);total_steps=steps_per_epoch*epochs
    scheduler=get_linear_schedule_with_warmup(optimizer,int(total_steps*cfg["training"]["warmup_ratio"]),total_steps)
    pad=tokenizer.pad_token_id or tokenizer.eos_token_id;max_tokens=cfg["training"]["max_tokens_per_chunk_microbatch"]
    history=[];best=float("inf");global_step=0;torch.cuda.reset_peak_memory_stats();run_start=time.perf_counter()
    for epoch in range(1,epochs+1):
        model.train();random.Random(seed+epoch).shuffle(multi);optimizer.zero_grad(set_to_none=True);epoch_start=time.perf_counter();losses=[];util=[];tokens=0;seen=0
        work=[("single",batch) for batch in single_chunk_batches(single,args.batch_token_budget,seed+epoch)]+[("multi",r) for r in multi]
        random.Random(seed+epoch).shuffle(work)
        for work_i,(kind,item) in enumerate(work,1):
            if kind=="single":
                out=train_single_batch(model,item,pad,device);batch_n=len(item);tokens+=sum(len(r["chunks"][0]) for r in item)
            else:
                r=item;structural=torch.tensor(r["structural"],device=device,dtype=next(model.parameters()).dtype)
                out=streaming_request_backward(model,r["chunks"],structural,torch.tensor(r["domain"],device=device),torch.tensor(r["tier"],device=device),pad,max_tokens,seed=seed*100000+epoch*20000+work_i);batch_n=1;tokens+=2*sum(map(len,r["chunks"]))
            losses.append(out["loss"]);seen+=batch_n
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],cfg["training"]["gradient_clip_norm"]);optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True);global_step+=1
            if work_i%25==0 or work_i==len(work):
                u=gpu_utilization();util.append(u["gpu_utilization_percent"] or 0)
                event={"epoch":epoch,"requests_seen":seen,"work_items":work_i,"single_batch_token_budget":args.batch_token_budget,"global_step":global_step,"train_loss_recent":float(np.mean(losses[-25:])),"learning_rates":[g["lr"] for g in optimizer.param_groups],"elapsed_seconds":time.perf_counter()-epoch_start,"tokens_per_second":tokens/(time.perf_counter()-epoch_start),"peak_vram_bytes":torch.cuda.max_memory_allocated(),**u}
                with (output/"logs"/"train.jsonl").open("a") as f:f.write(json.dumps(event)+"\n")
                print(json.dumps(event),flush=True)
        val_metrics,_=evaluate_partition(model,val,pad,max_tokens,device);epoch_record={"epoch":epoch,"train_loss":float(np.mean(losses)),"epoch_seconds":time.perf_counter()-epoch_start,"train_tokens_per_second":tokens/(time.perf_counter()-epoch_start),"average_sampled_gpu_utilization":float(np.mean(util)) if util else None,"peak_vram_bytes":torch.cuda.max_memory_allocated(),"validation":val_metrics}
        history.append(epoch_record);(output/"history.json").write_text(json.dumps(jsonable(history),indent=2))
        save_checkpoint(output/"latest",model,optimizer,scheduler,scaler,epoch,global_step,best,cfg)
        if val_metrics["validation_loss"]<best:best=val_metrics["validation_loss"];save_checkpoint(output/"best",model,optimizer,scheduler,scaler,epoch,global_step,best,cfg)
        print(json.dumps(epoch_record),flush=True)
    if not args.skip_test:
        load_model_checkpoint(output/"best",model);model.to(device);report=final_test_report(model,test,pad,max_tokens,device,output);print(json.dumps(jsonable(report["summary"])),flush=True)
    (output/"run_complete.json").write_text(json.dumps({"completed":True,"seconds":time.perf_counter()-run_start,"best_validation_loss":best,"test_evaluated":not args.skip_test},indent=2))


if __name__=="__main__":main()
