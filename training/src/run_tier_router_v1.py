from __future__ import annotations
import argparse,json,math,os,pickle,random,subprocess,time
from pathlib import Path
import numpy as np,pandas as pd,torch,yaml
from peft import get_peft_model_state_dict,set_peft_model_state_dict
from sklearn.metrics import accuracy_score,f1_score,log_loss,confusion_matrix
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer,get_linear_schedule_with_warmup
from .build_tier_router import build_tier_router,parameter_report
from .hierarchical_chunking import chunk_token_ids
from .streaming_gradient import pad_chunk_batch,microbatches
from .tier_ordinal import cumulative_probabilities,ordinal_prediction,routing_cost,selection_score
from .tier_streaming_gradient import tier_streaming_backward

def seed_all(seed):random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
def structural_raw(tokens,chars,chunks):return np.asarray([math.log1p(tokens),math.log1p(chars),math.log1p(1),math.log1p(chunks),0,math.log1p(tokens),0,0,0,0,0,0,1,0,0],np.float32)

def prepare(cfg,data_path,split_path,cache_path):
    if Path(cache_path).exists():
        with open(cache_path,"rb") as f:return pickle.load(f)
    tok=AutoTokenizer.from_pretrained(cfg["model"]["model_id"],revision=cfg["model"]["revision"]);df=pd.read_parquet(data_path);split=json.load(open(split_path))["datasets"]
    df=df[df.resolved].copy();df["partition"]=df.source_dataset.map(split)
    if df.partition.isna().any():raise ValueError("Unmapped source in split")
    chunk_size=cfg["request"]["chunk_size_tokens"];overlap=cfg["request"]["chunk_overlap_tokens"]
    ids=[tok.encode(str(x),add_special_tokens=False) for x in df.prompt];chunks=[chunk_token_ids(x,chunk_size,overlap,tok.eos_token_id) for x in ids]
    raw=np.stack([structural_raw(len(x),len(str(text)),len(c)) for x,text,c in zip(ids,df.prompt,chunks)]);scaler=StandardScaler().fit(raw[df.partition.eq("train")]);scaled=scaler.transform(raw).astype(np.float32)
    records=[]
    for pos,row in enumerate(df.itertuples()):records.append({"prompt_id":str(row.prompt_id),"source_dataset":str(row.source_dataset),"partition":row.partition,"tier":int(row.tier)-1,"chunks":chunks[pos],"structural":scaled[pos]})
    result=(records,scaler,tok.pad_token_id or tok.eos_token_id);Path(cache_path).parent.mkdir(parents=True,exist_ok=True)
    with open(cache_path,"wb") as f:pickle.dump(result,f,pickle.HIGHEST_PROTOCOL)
    return result

def token_batches(records,budget,seed,shuffle=True):
    rng=random.Random(seed);ordered=sorted(records,key=lambda r:len(r["chunks"][0]));buckets=[ordered[i:i+512] for i in range(0,len(ordered),512)]
    if shuffle:
        for b in buckets:rng.shuffle(b)
        rng.shuffle(buckets)
    batch=[];mx=0
    for r in [x for b in buckets for x in b]:
        n=len(r["chunks"][0])
        if batch and max(mx,n)*(len(batch)+1)>budget:yield batch;batch=[];mx=0
        batch.append(r);mx=max(mx,n)
    if batch:yield batch

def probabilities(model,logits):return torch.softmax(logits.float(),1) if model.loss_kind=="ce" else cumulative_probabilities(logits.float())
def predictions(model,logits):return logits.argmax(1)+1 if model.loss_kind=="ce" else ordinal_prediction(logits)

def train_batch(model,batch,pad,structural,weight):
    ids,mask=pad_chunk_batch([r["chunks"][0] for r in batch],pad,"cuda");vectors=model.encode_chunk_batch(ids,mask);features=torch.tensor(np.stack([r["structural"] for r in batch]),device="cuda",dtype=vectors.dtype) if structural else None
    logits=model.route(vectors,features);tiers=torch.tensor([r["tier"] for r in batch],device="cuda");loss=model.loss(logits,tiers,weight);loss.backward();return float(loss.detach()),int(mask.sum())

@torch.no_grad()
def infer_batch(model,batch,pad,structural):
    ids,mask=pad_chunk_batch([r["chunks"][0] for r in batch],pad,"cuda");vectors=model.encode_chunk_batch(ids,mask);features=torch.tensor(np.stack([r["structural"] for r in batch]),device="cuda",dtype=vectors.dtype) if structural else None
    return model.route(vectors,features)

@torch.no_grad()
def infer_multi(model,r,pad,structural):
    vectors=[]
    for b in microbatches(r["chunks"],2048):
        ids,mask=pad_chunk_batch(b,pad,"cuda");vectors.append(model.encode_chunk_batch(ids,mask))
    feature=torch.tensor(r["structural"],device="cuda",dtype=next(model.parameters()).dtype) if structural else None
    return model.route_chunks(torch.cat(vectors),feature)

def metrics(y,pred,prob,cost_matrix):
    y=np.asarray(y);pred=np.asarray(pred);delta=pred-y;num_tiers=prob.shape[1];onehot=np.eye(num_tiers)[y-1]
    f1=float(f1_score(y,pred,average="macro",zero_division=0));mae=float(np.abs(delta).mean());under=float((delta<0).mean());severe=float((delta<=-2).mean())
    costs=np.asarray(cost_matrix)[y-1,pred-1]/np.asarray(cost_matrix).max()
    result={"exact_accuracy":float(accuracy_score(y,pred)),"macro_f1":f1,"mae":mae,"within_one_tier":float((np.abs(delta)<=1).mean()),"under_routing":under,"over_routing":float((delta>0).mean()),"severe_under_routing":severe,"routing_cost":float(costs.mean()),"selection_score":selection_score(f1,mae,under,severe,num_tiers-1),"log_loss":float(log_loss(y,prob,labels=list(range(1,num_tiers+1)))),"brier":float(((prob-onehot)**2).sum(1).mean()),"confusion_matrix":confusion_matrix(y,pred,labels=list(range(1,num_tiers+1))).tolist()}
    for true_tier in range(2,num_tiers+1):
        for predicted_tier in range(1,true_tier):result[f"T{true_tier}_to_T{predicted_tier}"]=int(((y==true_tier)&(pred==predicted_tier)).sum())
    return result

def evaluate(model,records,pad,budget,structural,cost_matrix,prediction_path=None):
    model.eval();single=[r for r in records if len(r["chunks"])==1];multi=[r for r in records if len(r["chunks"])>1];ys=[];ps=[];probs=[];ids=[];sources=[];loss_sum=0;n=0;started=time.perf_counter()
    for batch in token_batches(single,budget,0,False):
        logits=infer_batch(model,batch,pad,structural);tiers=torch.tensor([r["tier"] for r in batch],device="cuda");loss_sum+=float(model.loss(logits,tiers,1).detach())*len(batch);n+=len(batch);pr=probabilities(model,logits).cpu().numpy();pd_=predictions(model,logits).cpu().numpy();ys.extend(tiers.cpu().numpy()+1);ps.extend(pd_);probs.extend(pr);ids.extend(r["prompt_id"] for r in batch);sources.extend(r["source_dataset"] for r in batch)
    for r in multi:
        logits=infer_multi(model,r,pad,structural);tier=torch.tensor([r["tier"]],device="cuda");loss_sum+=float(model.loss(logits,tier,1).detach());n+=1;pr=probabilities(model,logits).cpu().numpy();pd_=predictions(model,logits).cpu().numpy();ys.append(r["tier"]+1);ps.append(int(pd_[0]));probs.append(pr[0]);ids.append(r["prompt_id"]);sources.append(r["source_dataset"])
    result=metrics(ys,ps,np.asarray(probs),cost_matrix);result.update({"loss":loss_sum/n,"seconds":time.perf_counter()-started,"n":n})
    if prediction_path:
        pd.DataFrame({"prompt_id":ids,"source_dataset":sources,"true_tier":ys,"predicted_tier":ps,"probabilities":[x.tolist() for x in probs]}).to_parquet(prediction_path,index=False)
    return result

def save_checkpoint(path,model,optimizer,scheduler,scaler,state,run_cfg):
    path.mkdir(parents=True,exist_ok=True);torch.save({"peft":get_peft_model_state_dict(model.encoder),"request_norm":model.request_norm.state_dict(),"tier_head":model.tier_head.state_dict()},path/"model.pt");torch.save({"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),**state},path/"trainer_state.pt");np.savez(path/"structural_scaler.npz",mean=scaler.mean_,scale=scaler.scale_,var=scaler.var_);(path/"run_config.json").write_text(json.dumps(run_cfg,indent=2))
def load_checkpoint(path,model):
    s=torch.load(path/"model.pt",map_location="cpu",weights_only=True);set_peft_model_state_dict(model.encoder,s["peft"]);model.request_norm.load_state_dict(s["request_norm"]);model.tier_head.load_state_dict(s["tier_head"])

def main(default_config="configs/tier_router_v1.yaml"):
    p=argparse.ArgumentParser();p.add_argument("--name",required=True);p.add_argument("--loss",choices=["ce","ordinal"],required=True);p.add_argument("--under-weight",type=float,default=1);p.add_argument("--no-struct",action="store_true");p.add_argument("--eval-test",action="store_true");p.add_argument("--checkpoint");p.add_argument("--config",default=default_config);p.add_argument("--output-root");p.add_argument("--max-train",type=int);p.add_argument("--max-validation",type=int);p.add_argument("--epochs",type=int);args=p.parse_args()
    cfg=yaml.safe_load(open(args.config));seed_all(17);budget=cfg["training"]["single_chunk_batch_token_budget"];artifacts=cfg.get("artifacts",{});data_path=artifacts.get("dataset","data/final_labeled_dataset.parquet");cache_path=artifacts.get("cache","cache/tier_router_v1_seed17.pkl");output_root=args.output_root or artifacts.get("output_root","output/tier_router_v1");records,scaler,pad=prepare(cfg,data_path,"splits/grouped_split_17.json",cache_path);parts={x:[r for r in records if r["partition"]==x] for x in ["train","validation","test"]};out=Path(output_root)/args.name;out.mkdir(parents=True,exist_ok=True);structural=not args.no_struct
    model=build_tier_router(cfg,"cuda",args.loss,structural);cost=cfg["evaluation_cost_matrix"]["matrix"]
    run_cfg={"name":args.name,"loss":args.loss,"under_weight":args.under_weight,"structural":structural,"seed":17,"git_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),"version":cfg["version"],"num_tiers":cfg.get("num_tiers",4),"label_policy":cfg.get("label_policy","legacy_four_tier"),"model":cfg["model"],"training":cfg["training"],"selection":cfg["selection"],"evaluation_cost_matrix":cfg["evaluation_cost_matrix"]}
    if args.eval_test:
        if not args.checkpoint:raise ValueError("--checkpoint required")
        load_checkpoint(Path(args.checkpoint),model);result=evaluate(model,parts["test"],pad,budget,structural,cost,out/"test_predictions.parquet");(out/"test_metrics.json").write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));return
    train=parts["train"][:args.max_train] if args.max_train else parts["train"];parts["validation"]=parts["validation"][:args.max_validation] if args.max_validation else parts["validation"];single=[r for r in train if len(r["chunks"])==1];multi=[r for r in train if len(r["chunks"])>1];epochs=args.epochs or cfg["training"]["epochs"];steps=(len(list(token_batches(single,budget,17)))+len(multi))*epochs
    enc=[];head=[]
    for n,p_ in model.named_parameters():
        if p_.requires_grad:(enc if n.startswith("encoder.") else head).append(p_)
    opt=torch.optim.AdamW([{"params":enc,"lr":cfg["training"]["encoder_learning_rate"]},{"params":head,"lr":cfg["training"]["head_learning_rate"]}],weight_decay=cfg["training"]["weight_decay"]);sched=get_linear_schedule_with_warmup(opt,int(steps*cfg["training"]["warmup_ratio"]),steps);history=[];best=float("inf");global_step=0;torch.cuda.reset_peak_memory_stats();run_start=time.perf_counter()
    for epoch in range(1,epochs+1):
        model.train();work=[("single",x) for x in token_batches(single,budget,17+epoch)]+[("multi",x) for x in multi];random.Random(17+epoch).shuffle(work);losses=[];tokens=0;start=time.perf_counter()
        for kind,item in work:
            opt.zero_grad(set_to_none=True)
            if kind=="single":loss,tok=train_batch(model,item,pad,structural,args.under_weight);tokens+=tok
            else:
                feature=torch.tensor(item["structural"],device="cuda",dtype=next(model.parameters()).dtype) if structural else None;loss=tier_streaming_backward(model,item["chunks"],feature,torch.tensor(item["tier"],device="cuda"),pad,cfg["training"]["chunk_microbatch_tokens"],args.under_weight,170000+epoch*1000+global_step);tokens+=2*sum(map(len,item["chunks"]))
            torch.nn.utils.clip_grad_norm_([p_ for p_ in model.parameters() if p_.requires_grad],cfg["training"]["gradient_clip_norm"]);opt.step();sched.step();global_step+=1;losses.append(loss)
        val=evaluate(model,parts["validation"],pad,budget,structural,cost);row={"epoch":epoch,"train_loss":float(np.mean(losses)),"training_seconds":time.perf_counter()-start,"tokens_per_second":tokens/(time.perf_counter()-start),"peak_vram_bytes":torch.cuda.max_memory_allocated(),"validation":val};history.append(row);(out/"history.json").write_text(json.dumps(history,indent=2));state={"epoch":epoch,"global_step":global_step,"best_selection_score":best};save_checkpoint(out/"latest",model,opt,sched,scaler,state,run_cfg)
        if val["selection_score"]<best:best=val["selection_score"];state["best_selection_score"]=best;save_checkpoint(out/"best",model,opt,sched,scaler,state,run_cfg)
        print(json.dumps(row),flush=True)
    summary={"completed":True,"best_selection_score":best,"runtime_seconds":time.perf_counter()-run_start,"parameters":parameter_report(model),"run_config":run_cfg};(out/"summary.json").write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
if __name__=="__main__":main()
