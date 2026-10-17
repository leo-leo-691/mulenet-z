"""Toy cross-bank transaction graph that follows the playbook's data contract.

Use it to test the GNN pipeline before Role 1's real files exist:
    from mulenet.toy import make_toy
    nodes, load_view = make_toy()
`load_view(view, sample=False)` returns the same PyG Data fields as
mulenet/common.py: x, edge_index (both directions), edge_index_dir, y,
train/val/test masks, home_mask. Features are recomputed inside each view
(definition 6).

Planted laundering: victims -> collector -> a chain of hops at different
banks -> fan-out mules -> cash-out. Normal traffic includes busy merchants
(high fan-in) and payroll accounts (high fan-out), so single-account
statistics alone are ambiguous.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

INSTS = ["B1", "B2", "B3", "B4", "B5"]


def _features(src, dst, amt, ts, inst_of, n):
    """Account features from only the edges given (one view)."""
    f = {}
    cnt_in = np.bincount(dst, minlength=n).astype(float)
    cnt_out = np.bincount(src, minlength=n).astype(float)
    amt_in = np.bincount(dst, weights=amt, minlength=n)
    amt_out = np.bincount(src, weights=amt, minlength=n)
    pairs_in = np.unique(np.stack([dst, src]), axis=1)
    pairs_out = np.unique(np.stack([src, dst]), axis=1)
    f["cnt_in"], f["cnt_out"] = cnt_in, cnt_out
    f["uniq_in"] = np.bincount(pairs_in[0], minlength=n).astype(float)
    f["uniq_out"] = np.bincount(pairs_out[0], minlength=n).astype(float)
    f["amt_in"], f["amt_out"] = amt_in, amt_out
    f["mean_in"] = amt_in / np.maximum(cnt_in, 1)
    f["mean_out"] = amt_out / np.maximum(cnt_out, 1)
    f["pass_through"] = np.where(amt_in > 0, amt_out / np.maximum(amt_in, 1e-9), 0.0).clip(0, 5)
    cross = (inst_of[src] != inst_of[dst]).astype(float)
    f["cross_share"] = (np.bincount(src, weights=cross, minlength=n)
                        + np.bincount(dst, weights=cross, minlength=n)) / np.maximum(cnt_in + cnt_out, 1)
    first_in = np.full(n, np.inf)
    np.minimum.at(first_in, dst, ts)
    last_out = np.full(n, -np.inf)
    np.maximum.at(last_out, src, ts)
    span = last_out - first_in
    f["in_to_out_hours"] = np.where(np.isfinite(span), np.clip(span, 0, 24 * 30), 24 * 30)
    f["in_view"] = ((cnt_in + cnt_out) > 0).astype(float)
    X = pd.DataFrame(f)
    for c in X.columns:
        if c not in ("pass_through", "cross_share", "in_view"):
            X[c] = np.log1p(X[c])
    return X


def make_toy(n_accounts: int = 6000, n_chains: int = 80, seed: int = 42):
    rng = np.random.default_rng(seed)
    n = n_accounts
    inst_of = rng.choice(5, size=n, p=[0.3, 0.25, 0.2, 0.15, 0.1])
    by_inst = [np.where(inst_of == i)[0] for i in range(5)]

    src, dst, amt, ts, laund = [], [], [], [], []

    def add(s, d, a, t, l=0):
        src.append(s); dst.append(d); amt.append(a); ts.append(t); laund.append(l)

    # normal traffic: mostly same-bank, some cross-bank
    for s in range(n):
        for _ in range(rng.poisson(4)):
            same = rng.random() < 0.6
            pool = by_inst[inst_of[s]] if same else np.arange(n)
            d = int(rng.choice(pool))
            if d != s:
                add(s, d, rng.lognormal(8, 1), rng.uniform(0, 24 * 30))
    # busy merchants (fan-in) and payroll accounts (fan-out): legit look-alikes
    for m in rng.choice(n, 60, replace=False):
        for v in rng.choice(n, rng.integers(15, 40), replace=False):
            add(int(v), int(m), rng.lognormal(7, 1), rng.uniform(0, 24 * 30))
    for p in rng.choice(n, 40, replace=False):
        for v in rng.choice(n, rng.integers(10, 30), replace=False):
            add(int(p), int(v), rng.lognormal(9, 0.5), rng.uniform(0, 24 * 30))

    # laundering chains hopping across banks
    mule = np.zeros(n, dtype=int)
    for _ in range(n_chains):
        t = rng.uniform(0, 24 * 28)
        total = rng.lognormal(12, 0.5)
        hop_insts = rng.permutation(5)[: rng.integers(2, 5)]
        hops = [int(rng.choice(by_inst[i])) for i in hop_insts]
        for v in rng.choice(n, rng.integers(4, 9), replace=False):   # victims
            add(int(v), hops[0], total / 6, t + rng.uniform(0, 2), 1)
        for a, b in zip(hops[:-1], hops[1:]):                        # fast pass-through
            t += rng.uniform(0.1, 1.0)
            add(a, b, total * 0.95, t, 1)
        outs = [int(x) for x in rng.choice(n, rng.integers(3, 7), replace=False)]
        for o in outs:                                               # fan-out
            add(hops[-1], o, total * 0.9 / len(outs), t + rng.uniform(0.1, 1), 1)
        mule[hops] = 1
        mule[outs] = 1
        for o in outs:                                               # cash-out sinks
            add(o, int(rng.integers(0, n)), total * 0.8 / len(outs), t + rng.uniform(1, 3), 1)

    src, dst = np.array(src), np.array(dst)
    amt, ts, laund = np.array(amt), np.array(ts), np.array(laund)

    # contract: nodes table with stratified 60/20/20 split, seed 42
    split = np.empty(n, dtype=object)
    for lab in (0, 1):
        idx = rng.permutation(np.where(mule == lab)[0])
        a, b = int(0.6 * len(idx)), int(0.8 * len(idx))
        split[idx[:a]], split[idx[a:b]], split[idx[b:]] = "train", "val", "test"
    nodes = pd.DataFrame({"node_idx": np.arange(n),
                          "acct_key": [f"{inst_of[i]}_{i:06d}" for i in range(n)],
                          "bank": inst_of, "inst": [INSTS[i] for i in inst_of],
                          "label": mule, "split": split})

    def load_view(view: str, sample: bool = False) -> Data:
        if view == "combined":
            m = np.ones(len(src), dtype=bool)
            home = np.ones(n, dtype=bool)
        else:
            b = INSTS.index(view)
            m = (inst_of[src] == b) | (inst_of[dst] == b)
            home = inst_of == b
        X = _features(src[m], dst[m], amt[m], ts[m], inst_of, n)
        tr = (nodes.split.values == "train") & home
        mu, sd = X[tr].mean(), X[tr].std().replace(0, 1)
        Xs = ((X - mu) / sd).fillna(0).values.astype(np.float32)
        ei_dir = torch.tensor(np.stack([src[m], dst[m]]), dtype=torch.long)
        return Data(x=torch.tensor(Xs), edge_index=torch.cat([ei_dir, ei_dir.flip(0)], 1),
                    edge_index_dir=ei_dir, y=torch.tensor(mule),
                    train_mask=torch.tensor(nodes.split.values == "train"),
                    val_mask=torch.tensor(nodes.split.values == "val"),
                    test_mask=torch.tensor(nodes.split.values == "test"),
                    home_mask=torch.tensor(home), num_nodes=n)

    load_view.edges = pd.DataFrame({"src": src, "dst": dst, "amount": amt, "ts": ts,
                                    "is_laundering": laund})
    return nodes, load_view
