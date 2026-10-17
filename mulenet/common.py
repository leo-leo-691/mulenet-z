"""Shared data loader (playbook contract). Everyone loads data through this.

    from mulenet.common import load_view, load_nodes, load_edges, laundering_chains
    data = load_view("combined", sample=True)   # or "B1".."B5"

load_view returns a PyG Data object with:
    x               float32 [N, F], features computed ONLY from that view's
                    transactions (definition 6), log1p + standardised with the
                    train statistics of that view's own accounts
    edge_index      both directions (for undirected GNNs)
    edge_index_dir  original sender -> receiver, duplicates merged
    y, train_mask, val_mask, test_mask
    home_mask       the member's own accounts (all True for "combined")
    feature_names   list of column names of x

Features are cached as x_<view>.npy (+ feature_names.txt) next to the data,
so the second call is instant. Delete the x_*.npy files to rebuild.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

SHARED = Path(__file__).resolve().parents[1]
INSTS = ["B1", "B2", "B3", "B4", "B5"]
LOG_COLS = {"in_count", "out_count", "uniq_in", "uniq_out", "amt_in", "amt_out",
            "mean_in", "mean_out", "max_in", "max_out", "active_days", "n_formats",
            "first_in_to_last_out_h"}


def data_dir(sample: bool) -> Path:
    return SHARED / "data" / ("processed_sample" if sample else "processed")


def load_nodes(sample: bool = False) -> pd.DataFrame:
    nodes = pd.read_csv(data_dir(sample) / "nodes.csv", dtype={"acct_key": str})
    nodes = nodes.rename(columns={"node_id": "node_idx", "institution": "inst"})
    need = {"node_idx", "acct_key", "bank", "inst", "label", "split"}
    missing = need - set(nodes.columns)
    if missing:
        raise ValueError(f"nodes.csv is missing {sorted(missing)}; rebuild it with mulenet.build")
    nodes = nodes.sort_values("node_idx").reset_index(drop=True)
    assert (nodes.node_idx.values == np.arange(len(nodes))).all(), "node_idx must be 0..N-1"
    return nodes


def load_edges(sample: bool = False) -> dict:
    with np.load(data_dir(sample) / "edges.npz", allow_pickle=False) as z:
        E = {k: z[k] for k in z.files}
    need = {"src", "dst", "amount", "ts", "fmt", "is_laundering"}
    missing = need - set(E)
    if missing:
        raise ValueError(f"edges.npz is missing {sorted(missing)}; rebuild it with mulenet.build")
    return E


def _view_mask(E: dict, node_inst: np.ndarray, view: str) -> np.ndarray:
    if view == "combined":
        return np.ones(len(E["src"]), dtype=bool)
    if view not in INSTS:
        raise ValueError(f"view must be 'combined' or one of {INSTS}")
    return (node_inst[E["src"]] == view) | (node_inst[E["dst"]] == view)


def account_features(E: dict, m: np.ndarray, node_inst: np.ndarray, n: int) -> pd.DataFrame:
    """Every shared-list feature, computed from only the edges selected by m."""
    src, dst = E["src"][m].astype(np.int64), E["dst"][m].astype(np.int64)
    amt, ts, fmt = E["amount"][m].astype(np.float64), E["ts"][m], E["fmt"][m]
    names = [str(x) for x in E.get("fmt_names", [])]
    f = {}
    f["in_count"] = np.bincount(dst, minlength=n).astype(float)
    f["out_count"] = np.bincount(src, minlength=n).astype(float)
    pair_in = np.unique(dst * n + src)
    pair_out = np.unique(src * n + dst)
    f["uniq_in"] = np.bincount(pair_in // n, minlength=n).astype(float)
    f["uniq_out"] = np.bincount(pair_out // n, minlength=n).astype(float)
    f["amt_in"] = np.bincount(dst, weights=amt, minlength=n)
    f["amt_out"] = np.bincount(src, weights=amt, minlength=n)
    f["mean_in"] = f["amt_in"] / np.maximum(f["in_count"], 1)
    f["mean_out"] = f["amt_out"] / np.maximum(f["out_count"], 1)
    mx_in, mx_out = np.zeros(n), np.zeros(n)
    np.maximum.at(mx_in, dst, amt)
    np.maximum.at(mx_out, src, amt)
    f["max_in"], f["max_out"] = mx_in, mx_out
    f["pass_through"] = np.clip(f["amt_out"] / np.maximum(f["amt_in"], 1e-9), 0, 10) * (f["amt_in"] > 0)
    # timing: hours from first money in to last money out (fast pass-through = small)
    first_in = np.full(n, np.inf)
    np.minimum.at(first_in, dst, ts.astype(float))
    last_out = np.full(n, -np.inf)
    np.maximum.at(last_out, src, ts.astype(float))
    span = (last_out - first_in) / 3600.0
    f["first_in_to_last_out_h"] = np.where(np.isfinite(span) & (span >= 0), span, 24 * 30)
    f["receives_then_sends"] = (np.isfinite(span) & (span >= 0)).astype(float)
    day = ts // 86400
    act = np.unique(np.concatenate([src * 100_000 + day, dst * 100_000 + day]))
    f["active_days"] = np.bincount(act // 100_000, minlength=n).astype(float)
    fm = np.unique(np.concatenate([src * 16 + fmt, dst * 16 + fmt]))
    f["n_formats"] = np.bincount(fm // 16, minlength=n).astype(float)
    total = np.maximum(f["in_count"] + f["out_count"], 1)
    cross = (node_inst[src] != node_inst[dst]).astype(float)
    f["cross_member_share"] = (np.bincount(src, weights=cross, minlength=n)
                               + np.bincount(dst, weights=cross, minlength=n)) / total
    for name in ("ACH", "Cash"):
        if name in names:
            hit = (fmt == names.index(name)).astype(float)
            f[f"share_{name.lower()}"] = (np.bincount(src, weights=hit, minlength=n)
                                          + np.bincount(dst, weights=hit, minlength=n)) / total
    f["in_view"] = ((f["in_count"] + f["out_count"]) > 0).astype(float)
    X = pd.DataFrame(f)
    for c in X.columns:
        if c in LOG_COLS:
            X[c] = np.log1p(X[c])
    return X


def load_view(view: str = "combined", sample: bool = False, cache: bool = True):
    from torch_geometric.data import Data

    d = data_dir(sample)
    nodes = load_nodes(sample)
    E = load_edges(sample)
    n = len(nodes)
    node_inst = nodes["inst"].to_numpy()
    m = _view_mask(E, node_inst, view)

    cache_x, cache_names = d / f"x_{view}.npy", d / "feature_names.txt"
    fresh = cache and cache_x.exists() and cache_x.stat().st_mtime > (d / "edges.npz").stat().st_mtime
    if fresh:
        X = np.load(cache_x)
        names = cache_names.read_text().split()
    else:
        F = account_features(E, m, node_inst, n)
        home = np.ones(n, bool) if view == "combined" else node_inst == view
        tr = (nodes["split"].to_numpy() == "train") & home
        mu, sd = F[tr].mean(), F[tr].std().replace(0, 1)
        X = ((F - mu) / sd).fillna(0).to_numpy(np.float32)
        names = list(F.columns)
        if cache:
            np.save(cache_x, X)
            cache_names.write_text("\n".join(names))

    pairs = np.unique(E["src"][m].astype(np.int64) * n + E["dst"][m].astype(np.int64))
    ei_dir = torch.from_numpy(np.stack([pairs // n, pairs % n]))
    split = nodes["split"].to_numpy()
    home = np.ones(n, bool) if view == "combined" else node_inst == view
    return Data(x=torch.from_numpy(X), edge_index=torch.cat([ei_dir, ei_dir.flip(0)], 1),
                edge_index_dir=ei_dir, y=torch.from_numpy(nodes["label"].to_numpy(np.int64).copy()),
                train_mask=torch.from_numpy((split == "train").copy()),
                val_mask=torch.from_numpy((split == "val").copy()),
                test_mask=torch.from_numpy((split == "test").copy()),
                home_mask=torch.from_numpy(home), num_nodes=n, feature_names=names)


def laundering_chains(sample: bool = False) -> pd.DataFrame:
    """One row per account in a laundering chain (definition 9).

    chain = weakly connected component of the laundering transactions.
    Columns: node_idx, acct_key, inst, split, chain_id, chain_size, chain_members
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    nodes, E = load_nodes(sample), load_edges(sample)
    lw = E["is_laundering"] == 1
    s, t = E["src"][lw], E["dst"][lw]
    acc = np.unique(np.concatenate([s, t]))
    pos = {a: i for i, a in enumerate(acc)}
    si = np.array([pos[a] for a in s])
    ti = np.array([pos[a] for a in t])
    g = coo_matrix((np.ones(len(si)), (si, ti)), shape=(len(acc), len(acc)))
    _, comp = connected_components(g, directed=True, connection="weak")
    df = nodes.iloc[acc][["node_idx", "acct_key", "inst", "split"]].copy()
    df["chain_id"] = comp
    df["chain_size"] = df.groupby("chain_id")["node_idx"].transform("size")
    df["chain_members"] = df.groupby("chain_id")["inst"].transform("nunique")
    return df.reset_index(drop=True)
