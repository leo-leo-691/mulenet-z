"""Tabular baselines in both settings (Role 3, Niyati).

    from mulenet.baselines import run_baseline
    silo = run_baseline("xgb", "silo", sample=True)
    comb = run_baseline("xgb", "combined", sample=True)

Same rules as the GNN (playbook definitions 5-7): silo trains each member only
on its own train accounts with features from its own view, and scores only its
own accounts; combined trains one model on all train accounts.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from mulenet.common import INSTS, load_nodes, load_view


def _fit_predict(kind, X, y, tr, va, seed=42, threads=3):
    pos = max(int(y[tr].sum()), 1)
    if kind == "lr":
        from sklearn.linear_model import LogisticRegression
        m = LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0)
        m.fit(X[tr], y[tr])
        return m.predict_proba(X)[:, 1]
    if kind == "xgb":
        from xgboost import XGBClassifier
        m = XGBClassifier(n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.8,
                          colsample_bytree=0.8, tree_method="hist", n_jobs=threads,
                          scale_pos_weight=(tr.sum() - pos) / pos, eval_metric="aucpr",
                          early_stopping_rounds=40, random_state=seed)
        m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
        return m.predict_proba(X)[:, 1]
    raise ValueError("kind must be 'lr' or 'xgb'")


def run_baseline(kind: str, setting: str, sample: bool = False, out_dir: str | None = None) -> pd.DataFrame:
    nodes = load_nodes(sample)
    frames = []
    views = ["combined"] if setting == "combined" else INSTS
    for v in views:
        d = load_view(v, sample=sample)
        X, y = d.x.numpy(), d.y.numpy()
        home = d.home_mask.numpy()
        tr = d.train_mask.numpy() & home
        va = d.val_mask.numpy() & home
        p = _fit_predict(kind, X, y, tr, va)
        df = nodes.loc[home, ["acct_key", "inst", "split"]].copy()
        df["score"] = p[home]
        frames.append(df)
        print(f"[{kind}/{setting}] {v}: trained on {int(tr.sum()):,} accounts ({int(y[tr].sum())} mules)")
    out = pd.concat(frames, ignore_index=True)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{kind}_{setting}.csv")
        out.to_csv(path, index=False)
        print("wrote", path)
    return out
