import numpy as np
from src.aggregate_context import aggregate_context
from src.build_features import semantic_feature
from src.embed_requests import token_chunks, cache_fingerprint
from src.request_representation import Request
from src.structural_features import extract_structural, FEATURE_NAMES
from src.train_models import hybrid_oof_fold_ids
from src.build_grouped_splits import prompt_stratified_split
import pandas as pd


def test_chunking_has_overlap_and_no_truncation():
    chunks=token_chunks(list(range(10)),4,1)
    assert chunks==[[0,1,2,3],[3,4,5,6],[6,7,8,9],[9]]
    assert sorted(set(sum(chunks,[])))==list(range(10))


def test_short_prompt_does_not_duplicate_embedding():
    task=np.array([1.,0.,0.],dtype=np.float32)
    got=semantic_feature(task,np.empty((0,3),dtype=np.float32))
    np.testing.assert_array_equal(got,np.array([1,0,0,0,0,0,0,0,0],dtype=np.float32))


def test_relevance_weights_prefer_similar_chunk():
    task=np.array([1.,0.]); chunks=np.array([[1.,0.],[0.,1.]])
    _,_,w=aggregate_context(task,chunks,.1)
    assert w[0]>.999*w.sum()


def test_structural_schema_stable():
    x=extract_structural(Request.benchmark("p","hello"),1,1)
    assert x.shape==(len(FEATURE_NAMES),) and np.isfinite(x).all()


def test_cache_fingerprint_changes_with_revision():
    base=dict(model_id="m",revision="a",pooling="last_token",normalize=True,safe_max_tokens=10,
              chunk_size_tokens=8,chunk_overlap_tokens=1,task_instruction="x")
    a=cache_fingerprint(base); base["revision"]="b"
    assert a!=cache_fingerprint(base)


def test_hybrid_oof_keeps_every_class_in_training_and_groups_multi_source():
    y=np.array(["tool"]*10+["code"]*8)
    sources=np.array(["tau2"]*10+["c1"]*4+["c2"]*4)
    folds,methods=hybrid_oof_fold_ids(y,sources,n_splits=5,seed=1)
    assert methods["tool"]["method"]=="stratified_prompt_fallback"
    assert methods["code"]["method"]=="source_grouped"
    assert len(set(folds[sources=="c1"]))==1 and len(set(folds[sources=="c2"]))==1
    for k in range(5): assert set(y[folds!=k])=={"tool","code"}


def test_secondary_split_preserves_small_strata():
    df=pd.DataFrame({"component":["tool_use"]*20,"source_dataset":["tau2"]*20,
                     "tier":[1,2,3,4]*5})
    split=prompt_stratified_split(df,3)
    assert set(split)=={"train","validation","test"}
