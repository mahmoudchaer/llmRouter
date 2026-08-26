from __future__ import annotations
import argparse,json,resource,time
from pathlib import Path
import torch,yaml
from .build_baseline3_model import build,parameter_report
from .streaming_gradient import streaming_request_backward

def run(size,config="configs/baseline3.yaml"):
    cfg=yaml.safe_load(open(config));device="mps" if torch.backends.mps.is_available() else "cpu"
    model=build(cfg,device=device,aggregation="mean",structural=True);model.train();report=parameter_report(model)
    content=min(size-1,4095);chunks=[list(range(1,content+1)) + [151645]]
    structural=torch.zeros(15,device=device,dtype=next(model.parameters()).dtype)
    model.zero_grad(set_to_none=True);start=time.perf_counter();error=None
    try:
        losses=streaming_request_backward(model,chunks,structural,torch.tensor(1,device=device),torch.tensor(2,device=device),0,size,seed=44)
        torch.mps.synchronize() if device=="mps" else None
    except Exception as exc:error=f"{type(exc).__name__}: {exc}";losses={}
    elapsed=time.perf_counter()-start
    encoder_grad=sum(float(p.grad.detach().abs().sum()) for n,p in model.encoder.named_parameters() if p.requires_grad and p.grad is not None)
    result={"chunk_size":size,"device":device,"elapsed_forward_backward_seconds":elapsed,"tokens_per_second":content/elapsed,
            "peak_process_rss_mb":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024/1024,"error":error,"parameters":report,"losses":losses}
    result["encoder_trainable_gradient_abs_sum"]=encoder_grad
    out=Path(config).resolve().parent.parent/"reports"/f"baseline3_training_benchmark_{size}.json";out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--size",type=int,required=True);p.add_argument("--config",default="configs/baseline3.yaml");a=p.parse_args();run(a.size,a.config)
