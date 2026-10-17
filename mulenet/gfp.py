"""Graph-pattern features with IBM Snap ML's GraphFeaturePreprocessor (Role 2).

Snap ML's GFP mines AML patterns on the transaction graph: fan-in / fan-out,
degree, scatter-gather and temporal cycles. It is IBM technology built for
financial-crime detection and runs on IBM Z / LinuxONE.

Why it matters for the experiment: a cycle or scatter-gather that passes
through another bank is INVISIBLE in a bank's silo view, so even a bank's own
accounts get different pattern counts in the silo and the combined view. That
is exactly the "the joined graph reveals what one bank cannot see" claim.

    from mulenet.gfp import smoke_test, run_gfp_xgb
    smoke_test(sample=True)                                  # 1 minute, checks Snap ML works
    run_gfp_xgb("silo", sample=True, out_dir=...)            # gfpxgb_silo.csv
    run_gfp_xgb("combined", sample=True, out_dir=...)        # gfpxgb_combined.csv
    run_gfp_xgb("combined", sample=True, out_dir=..., label_scope="own")   # graph effect only
"""
from __future__ import annotations

import hashlib
import json
import os
import time

import numpy as np
import pandas as pd

from mulenet.common import INSTS, _view_mask, data_dir, load_edges, load_nodes, load_view

# time unit = minutes since the first transaction
GFP_PARAMS = {
    "num_threads": 3,
    "time_window": -1,            # = the largest pattern window below
    "vertex_stats": False,        # per-account amount stats already exist in common.py
    "fan": True, "fan_tw": 1440, "fan_bins": [2, 3, 4, 6, 10],
    "degree": True, "degree_tw": 1440, "degree_bins": [2, 3, 4, 6, 10],
    # laundering spreads over days: 3-day windows for the cross-bank patterns
    "scatter-gather": True, "scatter-gather_tw": 4320, "scatter-gather_bins": [2, 3, 4, 6],
    "temp-cycle": True, "temp-cycle_tw": 4320, "temp-cycle_bins": [2, 3, 4, 6],
    "lc-cycle": False,
}

# Full-data setting: the 8 GB LinuxONE VM ran out of memory on temporal-cycle
# search over all 5M transactions, so cycles are switched off. Every view
# (silo AND combined) uses this same setting, keeping the comparison fair.
GFP_PARAMS_FULL = {**GFP_PARAMS, "temp-cycle": False}


def _gfp(params: dict):
    from snapml import GraphFeaturePreprocessor
    gp = GraphFeaturePreprocessor()
    p = gp.get_params()
    p.update(params)
    gp.set_params(p)
    return gp


def _edge_array(E: dict, m: np.ndarray, t0: int) -> tuple[np.ndarray, np.ndarray]:
    """Rows: edge_id, src, dst, minutes since t0 - sorted by time (Snap ML wants that)."""
    idx = np.flatnonzero(m)
    order = np.argsort(E["ts"][idx], kind="stable")
    idx = idx[order]
    X = np.column_stack([np.arange(len(idx)), E["src"][idx], E["dst"][idx],
                         (E["ts"][idx] - t0) / 60.0]).astype(np.float64)
    return X, idx


def gfp_node_features(E: dict, m: np.ndarray, n: int, params: dict = GFP_PARAMS, gp_factory=_gfp):
    """Run Snap ML GFP on the selected edges, then sum each edge pattern feature
    onto its sender (out_*) and receiver (in_*) account. Returns (n x 2k array, names)."""
    X, _ = _edge_array(E, m, int(E["ts"].min()))
    gp = gp_factory(params)
    out = gp.fit_transform(X)
    del gp
    feats = np.array(out[:, X.shape[1]:], dtype=np.float32)    # appended graph features only
    del out
    src, dst = X[:, 1].astype(np.int64), X[:, 2].astype(np.int64)
    del X
    k = feats.shape[1]
    cols, names = [], []
    for j in range(k):
        cols.append(np.bincount(src, weights=feats[:, j], minlength=n))
        names.append(f"gfp_out_{j}")
    for j in range(k):
        cols.append(np.bincount(dst, weights=feats[:, j], minlength=n))
        names.append(f"gfp_in_{j}")
    return np.log1p(np.column_stack(cols)), names


def gfp_node_features_stream(E: dict, m: np.ndarray, n: int, params: dict = GFP_PARAMS,
                             chunk: int = 250_000, gp_factory=_gfp, verbose: bool = True,
                             mode: str = "transform"):
    """Same features as gfp_node_features, but edges are streamed to Snap ML in
    time order, chunk by chunk (Snap ML's dynamic-graph mode). Snap ML only keeps
    edges inside the time window, so memory stays bounded by ~one window of
    transactions instead of the whole dataset. Node features are accumulated
    chunk by chunk, so the full edge x feature matrix is never held in memory.

    Snap ML's transform() inserts the chunk into the graph and then computes its
    features, and old edges fall out of the time window. So each transaction's
    features use only transactions up to that moment, the way a live system
    would score it. (The one-shot fit_transform builds the whole graph first,
    so its features also see later transactions inside the window.)
    mode="partial_fit" is kept only for diagnostics: it inserts every edge twice."""
    X, _ = _edge_array(E, m, int(E["ts"].min()))
    gp = gp_factory(params)
    acc_out = acc_in = None
    t = time.perf_counter()
    for i, start in enumerate(range(0, len(X), chunk)):
        part = X[start:start + chunk]
        if i == 0:
            out = gp.fit_transform(part)
        else:
            if mode == "partial_fit":
                gp.partial_fit(part)
            out = gp.transform(part)
        f = np.asarray(out[:, part.shape[1]:], dtype=np.float64)
        del out
        if acc_out is None:
            acc_out = np.zeros((n, f.shape[1]))
            acc_in = np.zeros((n, f.shape[1]))
        src, dst = part[:, 1].astype(np.int64), part[:, 2].astype(np.int64)
        for j in range(f.shape[1]):
            acc_out[:, j] += np.bincount(src, weights=f[:, j], minlength=n)
            acc_in[:, j] += np.bincount(dst, weights=f[:, j], minlength=n)
        del f, part
        if verbose and (i % 4 == 0):
            print(f"    streamed {min(start + chunk, len(X)):,}/{len(X):,} edges "
                  f"({time.perf_counter() - t:.0f}s)", flush=True)
    k = acc_out.shape[1]
    names = [f"gfp_out_{j}" for j in range(k)] + [f"gfp_in_{j}" for j in range(k)]
    return np.log1p(np.hstack([acc_out, acc_in])), names


def _gfp_cache_path(view, sample, params, stream):
    tag = hashlib.sha1((json.dumps(params, sort_keys=True) + ("stream-causal" if stream else "")).encode()).hexdigest()[:8]
    return data_dir(sample) / f"x_gfp_{view}_{tag}.npy"   # new settings -> new cache file


def load_view_gfp(view: str, sample: bool = False, cache: bool = True, params: dict | None = None,
                  stream: bool = False):
    """common.load_view + Snap ML graph-pattern features computed inside the same view."""
    params = params or GFP_PARAMS
    d = load_view(view, sample=sample)
    nodes = load_nodes(sample)
    path = _gfp_cache_path(view, sample, params, stream)
    if cache and path.exists() and path.stat().st_mtime > (data_dir(sample) / "edges.npz").stat().st_mtime:
        G = np.load(path)
    else:
        E = load_edges(sample)
        m = _view_mask(E, nodes["inst"].to_numpy(), view)
        t = time.perf_counter()
        fn = gfp_node_features_stream if stream else gfp_node_features
        G, _ = fn(E, m, len(nodes), params=params)
        del E
        print(f"  Snap ML GFP on {view}: {int(m.sum()):,} edges in {time.perf_counter() - t:.1f}s")
        if cache:
            np.save(path, G.astype(np.float32))
    home = d.home_mask.numpy()
    tr = d.train_mask.numpy() & home
    mu, sd = G[tr].mean(0), G[tr].std(0)
    sd[sd == 0] = 1
    import torch
    d.x = torch.cat([d.x, torch.from_numpy(((G - mu) / sd).astype(np.float32))], 1)
    return d


def run_gfp_xgb(setting: str, sample: bool = False, out_dir: str | None = None,
                label_scope: str = "all", params: dict | None = None, suffix: str = "",
                stream: bool = False) -> pd.DataFrame:
    """XGBoost on base + graph-pattern features, same fairness rules as everything else.

    silo:               per member, its own view, its own train labels, scores its accounts
    combined / all:     joined view, all members' train labels
    combined / own:     joined view, but each member trains only on its own labels
                        -> isolates the effect of the joined GRAPH from label sharing
    flproxy:            federated-learning proxy: ONE model on all members' labels, but every
                        account's features come only from its own member's silo view
                        -> label sharing without the joined graph (upper bound of horizontal FL)
    """
    from mulenet.baselines import _fit_predict
    nodes = load_nodes(sample)
    if setting == "flproxy":
        return _run_flproxy(nodes, sample, out_dir, params or GFP_PARAMS, suffix, stream, _fit_predict)
    frames = []
    if setting == "combined" and label_scope == "all":
        jobs = [("combined", None)]
    elif setting == "combined":
        jobs = [("combined", b) for b in INSTS]
    else:
        jobs = [(b, b) for b in INSTS]
    cache = {}
    for view, member in jobs:
        if view not in cache:
            cache = {}
            cache = {view: load_view_gfp(view, sample, params=params, stream=stream)}
        d = cache[view]
        X, y = d.x.numpy(), d.y.numpy()
        mine = np.ones(len(y), bool) if member is None else (nodes["inst"].to_numpy() == member)
        tr, va = d.train_mask.numpy() & mine, d.val_mask.numpy() & mine
        p = _fit_predict("xgb", X, y, tr, va)
        df = nodes.loc[mine, ["acct_key", "inst", "split"]].copy()
        df["score"] = p[mine]
        frames.append(df)
        print(f"[gfp-xgb/{setting}/{label_scope}] view {view}, labels {member or 'all'}: "
              f"{int(tr.sum()):,} train accounts, {X.shape[1]} features")
    out = pd.concat(frames, ignore_index=True)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        name = f"gfpxgb{suffix}_{setting}" + ("_ownlabels" if setting == "combined" and label_scope == "own" else "")
        out.to_csv(os.path.join(out_dir, name + ".csv"), index=False)
        print("wrote", os.path.join(out_dir, name + ".csv"))
    return out


def _run_flproxy(nodes, sample, out_dir, params, suffix, stream, _fit_predict):
    """Stack each member's home accounts with features from that member's silo view only.
    Features are left unstandardised: each silo view standardises with its own statistics,
    which would put the same raw value at different points per member. Trees are invariant
    to one monotone transform per feature, so raw features match what the joined run sees."""
    from mulenet.common import account_features
    E = load_edges(sample)
    inst = nodes["inst"].to_numpy()
    n = len(nodes)
    X = None
    for b in INSTS:
        path = _gfp_cache_path(b, sample, params, stream)
        if not path.exists():
            load_view_gfp(b, sample, params=params, stream=stream)        # builds the cache
        G = np.load(path)
        F = account_features(E, _view_mask(E, inst, b), inst, n).fillna(0).to_numpy(np.float32)
        Xb = np.hstack([F, G]).astype(np.float32)
        if X is None:
            X = np.zeros_like(Xb)
        home = inst == b
        X[home] = Xb[home]                                               # each row from its own silo
        print(f"[gfp-xgb/flproxy] {b}: {int(home.sum()):,} accounts, features from silo view {b}", flush=True)
    del E
    y = nodes["label"].to_numpy()
    split = nodes["split"].to_numpy()
    p = _fit_predict("xgb", X, y, split == "train", split == "val")       # same model, params, seed
    out = nodes[["acct_key", "inst", "split"]].assign(score=p)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        out.to_csv(os.path.join(out_dir, f"gfpxgb{suffix}_flproxy.csv"), index=False)
        print("wrote", os.path.join(out_dir, f"gfpxgb{suffix}_flproxy.csv"))
    return out


def smoke_test(sample: bool = True, n_edges: int = 200_000):
    """Check that Snap ML runs here and show what it returns, on a small slice."""
    from snapml import GraphFeaturePreprocessor
    E = load_edges(sample)
    nodes = load_nodes(sample)
    m = np.zeros(len(E["src"]), bool)
    m[np.argsort(E["ts"], kind="stable")[:n_edges]] = True
    X, _ = _edge_array(E, m, int(E["ts"].min()))
    gp = _gfp(GFP_PARAMS)
    t = time.perf_counter()
    out = gp.fit_transform(X)
    dt = time.perf_counter() - t
    print("params accepted:", {k: v for k, v in gp.get_params().items() if k in GFP_PARAMS})
    print(f"input {X.shape} -> output {out.shape} in {dt:.2f}s ({len(X) / dt:,.0f} edges/s)")
    extra = out[:, X.shape[1]:]
    print("non-zero share per graph feature:", np.round((extra != 0).mean(0), 3).tolist())
    G, names = gfp_node_features(E, m, len(nodes))
    print("node features:", G.shape)
