import numpy as np
from src.xgb_models import deterministic_search_configs,balanced_weights,domain_specific_eligible

CFG={"search_seed":1,"configurations":16,"n_estimators":[300,600,1000],"max_depth":[3,5,7],
"learning_rate":[.03,.07,.1],"subsample":[.8,1.],"colsample_bytree":[.7,1.],"min_child_weight":[1,5],
"reg_lambda":[1,5],"reg_alpha":[0,.1]}

def test_search_is_deterministic_and_balanced_pairs():
    a=deterministic_search_configs(CFG);b=deterministic_search_configs(CFG)
    assert a==b and len(a)==16
    assert sum(x["weighting"]=="none" for x in a)==8
    assert sum(x["weighting"]=="balanced" for x in a)==8

def test_balanced_weights_equalize_class_mass():
    y=np.array([0,0,0,1]);w=balanced_weights(y)
    assert np.isclose(w[y==0].sum(),w[y==1].sum())

def test_domain_specific_eligibility():
    cfg={"minimum_resolved_train_rows":500,"minimum_rows_per_tier":20,"require_all_four_tiers":True}
    assert domain_specific_eligible(np.tile(np.arange(4),125),cfg)[0]
    assert not domain_specific_eligible(np.tile(np.arange(4),40),cfg)[0]
