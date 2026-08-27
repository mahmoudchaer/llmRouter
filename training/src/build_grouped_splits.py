from __future__ import annotations

import json
import numpy as np
import pandas as pd


def semantic_domain(df: pd.DataFrame) -> pd.Series:
    return df["component"].replace({"logic_arc_agi": "logic", "logic_classic": "logic"})


def _score_assignment(summary, assignment, total, global_domain, num_tiers, target=(.70, .15, .15)):
    loss = 0.0
    for split, frac in zip(("train", "validation", "test"), target):
        names=[g for g,b in assignment.items() if b==split]
        part=summary.loc[names].sum(axis=0)
        n=part["n"]
        loss += 4 * abs(n / total - frac)
        part_domain=part[global_domain.index]/max(n,1)
        loss += (global_domain-part_domain).abs().mean()
        if split=="train":
            if (part[[f"tier_{i}" for i in range(1,num_tiers+1)]]>0).sum()<num_tiers: return np.inf
            if (part[global_domain.index]>0).sum()<7: return np.inf
    return float(loss)


def controlled_group_split(df: pd.DataFrame, seed: int, iterations: int = 5000):
    rng = np.random.default_rng(seed)
    groups = df.source_dataset.unique()
    sizes = df.source_dataset.value_counts()
    domains=sorted(semantic_domain(df).unique())
    temp=df.assign(semantic_domain=semantic_domain(df))
    summary=pd.DataFrame(index=groups)
    summary["n"]=sizes
    for d in domains: summary[d]=temp[temp.semantic_domain==d].source_dataset.value_counts().reindex(groups,fill_value=0)
    num_tiers=int(temp.loc[temp.resolved,"tier"].max())
    for i in range(1,num_tiers+1): summary[f"tier_{i}"]=temp[temp.resolved & (temp.tier==i)].source_dataset.value_counts().reindex(groups,fill_value=0)
    global_domain=temp.semantic_domain.value_counts(normalize=True).reindex(domains)
    candidates = []
    for _ in range(iterations):
        order = list(rng.permutation(groups))
        jitter = {g: rng.uniform(.65, 1.35) for g in groups}
        buckets = {"train": [], "validation": [], "test": []}
        counts = dict.fromkeys(buckets, 0)
        targets = {"train": .70*len(df), "validation": .15*len(df), "test": .15*len(df)}
        # Noisy largest-first packing yields genuinely different, still controlled,
        # source-held-out configurations across seeds.
        for g in sorted(order, key=lambda x: (-sizes[x] * jitter[x], order.index(x))):
            bucket_candidates = sorted(buckets, key=lambda b: counts[b] / targets[b])
            b = bucket_candidates[0]; buckets[b].append(g); counts[b] += sizes[g]
        assignment = {g: b for b, gs in buckets.items() for g in gs}
        score = _score_assignment(summary, assignment, len(df), global_domain, num_tiers)
        if np.isfinite(score): candidates.append((score, assignment))
    if not candidates: raise RuntimeError("No viable source-isolated split found")
    unique={tuple(sorted(a.items())):(s,a) for s,a in candidates}
    candidates=sorted(unique.values(),key=lambda x:x[0])
    best_score=candidates[0][0]
    near=[x for x in candidates if x[0] <= best_score + .025]
    # Select reproducibly among near-optimal layouts. Robustness needs genuinely
    # different held-out datasets, not five rediscoveries of one optimum.
    chosen=near[seed % len(near)]
    return chosen[1], chosen[0]


def save_split(path, seed, assignment, score):
    path.write_text(json.dumps({"seed": seed, "objective": score,
                                "datasets": assignment}, indent=2, sort_keys=True))


def prompt_stratified_split(df: pd.DataFrame, seed: int, target=(.70,.15,.15)) -> pd.Series:
    """Secondary evaluation only; preserves classes but not source isolation."""
    rng=np.random.default_rng(seed)
    domain=semantic_domain(df)
    tier=df.tier.fillna(0).astype(int).astype(str)
    strata=domain.astype(str)+"|"+tier
    result=pd.Series(index=df.index,dtype="object")
    names=("train","validation","test")
    for _,idx in strata.groupby(strata).groups.items():
        idx=np.asarray(list(idx)); rng.shuffle(idx); n=len(idx)
        n_train=int(np.floor(target[0]*n))
        n_val=int(round(target[1]*n))
        if n>=3:
            n_train=max(1,min(n-2,n_train)); n_val=max(1,min(n-n_train-1,n_val))
        cuts=(n_train,n_train+n_val)
        for name,part in zip(names,np.split(idx,cuts)): result.loc[part]=name
    if result.isna().any(): raise RuntimeError("Secondary split assignment incomplete")
    return result
