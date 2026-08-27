from __future__ import annotations

import json
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
import os
from pathlib import Path
import sys

import yaml

ROOT=Path(__file__).resolve().parents[1]
STATIC=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))

from routing.mock_registry import build_mock_registry
from routing.models.factory import build_mvp_classifier
from routing.pipeline.router import LLMOnlyRuntimeRouter
from routing.schemas.request import CustomerPriceCeiling,HardRequirements,RoutingRequest
from routing.selection.model_selector import ModelSelector


_router=None


def load_root_env()->None:
    path=ROOT/".env"
    if not path.exists():return
    for raw in path.read_text().splitlines():
        line=raw.strip()
        if not line or line.startswith("#") or "=" not in line:continue
        key,value=line.split("=",1);key=key.strip();value=value.strip().strip('"').strip("'")
        if key:os.environ.setdefault(key,value)


def get_router():
    global _router
    if _router is None:
        config=yaml.safe_load((ROOT/"routing/config/routing.yaml").read_text())
        _router=LLMOnlyRuntimeRouter(build_mvp_classifier(config),ModelSelector(),build_mock_registry())
    return _router


def predict(payload:dict)->dict:
    prompt=str(payload.get("prompt","")).strip()
    if not prompt:raise ValueError("Prompt is required")
    ceiling=CustomerPriceCeiling(float(payload.get("max_input_price",1)),float(payload.get("max_output_price",3)))
    requirements=HardRequirements(context_tokens=max(1,(len(prompt)+3)//4),requires_tools=bool(payload.get("requires_tools",False)),
        requires_structured_output=bool(payload.get("requires_structured_output",False)))
    request=RoutingRequest("frontend-preview",prompt,ceiling,requirements,expected_output_tokens=int(payload.get("expected_output_tokens",512)))
    result=get_router().route(request)
    return {"domain":result.domain,"tier":result.final_tier,"selected_model":result.selected_model,
            "provider":result.selected_provider,"reason":result.reason_for_selection,"audit":result.audit}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self,*args,**kwargs):super().__init__(*args,directory=str(STATIC),**kwargs)

    def do_POST(self):
        if self.path!="/api/predict":self.send_error(404);return
        try:
            length=int(self.headers.get("Content-Length","0"));payload=json.loads(self.rfile.read(length) or b"{}")
            body=json.dumps(predict(payload)).encode();status=200
        except Exception as error:
            body=json.dumps({"error":str(error)}).encode();status=400
        self.send_response(status);self.send_header("Content-Type","application/json");self.send_header("Content-Length",str(len(body)));self.end_headers();self.wfile.write(body)


if __name__=="__main__":
    load_root_env()
    host,port="127.0.0.1",8787
    print(f"Tarsiq preview: http://{host}:{port}")
    ThreadingHTTPServer((host,port),Handler).serve_forever()
