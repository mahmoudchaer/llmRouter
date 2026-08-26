from __future__ import annotations
import argparse,json,re,time
from pathlib import Path
import numpy as np,pandas as pd,torch,yaml
from huggingface_hub import hf_hub_download
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,f1_score
from transformers import AutoModel,AutoModelForCausalLM,AutoTokenizer

def metrics(df,labels):
    mapped=df[df.mapped_label.notna()];coverage=len(mapped)/len(df);report=classification_report(mapped.true_domain,mapped.mapped_label,labels=labels,output_dict=True,zero_division=0)
    return {"n":len(df),"mapped_n":len(mapped),"coverage":coverage,"unmapped_n":len(df)-len(mapped),"accuracy_on_mapped":accuracy_score(mapped.true_domain,mapped.mapped_label) if len(mapped) else None,"macro_f1_on_mapped":f1_score(mapped.true_domain,mapped.mapped_label,labels=labels,average="macro",zero_division=0) if len(mapped) else None,"effective_accuracy_all":float((df.true_domain==df.mapped_label).sum()/len(df)),"classification_report":report,"confusion_matrix":confusion_matrix(mapped.true_domain,mapped.mapped_label,labels=labels).tolist(),"latency_ms_mean":df.latency_ms.mean(),"latency_ms_p50":df.latency_ms.median(),"latency_ms_p95":df.latency_ms.quantile(.95),"throughput_prompts_per_second":1000/df.latency_ms.mean()}

def zen(df,cfg,device):
    z=cfg["zen"];tok=AutoTokenizer.from_pretrained(z["backbone_id"],revision=z["backbone_revision"]);model=AutoModel.from_pretrained(z["backbone_id"],revision=z["backbone_revision"],torch_dtype=torch.float16 if device=="mps" else torch.float32).to(device).eval();state=torch.load(hf_hub_download(z["router_id"],"zen-router.pt",revision=z["router_revision"]),map_location="cpu",weights_only=True);W=state["task_head.weight"].to(device=device,dtype=next(model.parameters()).dtype);b=state["task_head.bias"].to(device=device,dtype=W.dtype);rows=[]
    for r in df.itertuples():
        start=time.perf_counter();enc=tok(str(r.prompt),return_tensors="pt",add_special_tokens=True);n=enc.input_ids.shape[1]
        if n>32768: native=None;reason="input_over_native_context"
        else:
            enc={k:v.to(device) for k,v in enc.items()};
            with torch.inference_mode():h=model(**enc).last_hidden_state[0,-1];native=z["native_labels"][int((W@h+b).argmax())]
            reason=None
        latency=(time.perf_counter()-start)*1000;mapping=z["mapping"].get(native,{"common":None,"status":"unmapped"}) if native else {"common":None,"status":"unmapped"};rows.append({"prompt_id":r.prompt_id,"true_domain":r.domain,"source_dataset":r.source_dataset,"native_label":native,"mapped_label":mapping["common"],"mapping_status":mapping["status"],"unmapped_reason":reason or (None if mapping["common"] else "native_label_unmapped"),"input_tokens":n,"latency_ms":latency})
    return pd.DataFrame(rows)

def arch_prompt(routes,text):
    instruction="""You are a helpful assistant designed to find the best suited route.\nYou are provided with route description within <routes></routes> XML tags:\n<routes>\n{routes}\n</routes>\n<conversation>\n{conversation}\n</conversation>\n\nYour task is to decide which route best suits the latest user intent. If no route matches, respond {\"route\": \"other\"}. Respond only as JSON using the exact route name: {\"route\": \"route_name\"}."""
    return instruction.format(routes=json.dumps(routes),conversation=json.dumps([{"role":"user","content":text}]))
def arch(df,cfg,device):
    a=cfg["arch"];tok=AutoTokenizer.from_pretrained(a["model_id"],revision=a["revision"]);model=AutoModelForCausalLM.from_pretrained(a["model_id"],revision=a["revision"],torch_dtype=torch.float16 if device=="mps" else torch.float32).to(device).eval();valid=set(cfg["common_taxonomy"]);rows=[]
    for r in df.itertuples():
        start=time.perf_counter();prompt=arch_prompt(a["routes"],str(r.prompt));messages=[{"role":"user","content":prompt}];ids=tok.apply_chat_template(messages,add_generation_prompt=True,return_tensors="pt");n=ids.shape[1]
        if n>model.config.max_position_embeddings:native=None;raw=None;reason="input_over_native_context"
        else:
            with torch.inference_mode():out=model.generate(ids.to(device),max_new_tokens=24,do_sample=False,pad_token_id=tok.eos_token_id)
            raw=tok.decode(out[0,n:],skip_special_tokens=True).strip();m=re.search(r'"route"\s*:\s*"([^"]+)"',raw);native=m.group(1) if m else None;reason=None if native in valid else ("native_other" if native=="other" else "invalid_generation")
        rows.append({"prompt_id":r.prompt_id,"true_domain":r.domain,"source_dataset":r.source_dataset,"native_label":native,"mapped_label":native if native in valid else None,"mapping_status":"configured_identity" if native in valid else "unmapped","unmapped_reason":reason,"raw_output":raw,"input_tokens":n,"latency_ms":(time.perf_counter()-start)*1000})
    return pd.DataFrame(rows)
def main():
    p=argparse.ArgumentParser();p.add_argument("--router",choices=["zen","arch"],required=True);args=p.parse_args();cfg=yaml.safe_load(open("configs/domain_router_eval.yaml"));df=pd.read_parquet("reports/domain_router_eval/subset.parquet");device="mps" if torch.backends.mps.is_available() else "cpu";started=time.perf_counter();result=zen(df,cfg,device) if args.router=="zen" else arch(df,cfg,device);out=Path("reports/domain_router_eval");result.to_parquet(out/f"{args.router}_predictions.parquet",index=False);summary=metrics(result,cfg["common_taxonomy"]);summary.update({"router":args.router,"device":device,"wall_seconds":time.perf_counter()-started,"taxonomy":cfg[args.router]});(out/f"{args.router}_metrics.json").write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k not in ["classification_report","confusion_matrix","taxonomy"]},indent=2))
if __name__=="__main__":main()
