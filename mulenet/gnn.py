"""MuleNet-Z GNN (Role 2, Arindam).

Direction-aware GraphSAGE built on sparse mean-adjacency matrices, plus a
SIGN-style fallback (precomputed propagation + MLP).

Why sparse matrices instead of PyG message passing: plain message passing
materialises one vector per edge (about 2.5 GB per layer at width 64 on the
5M-edge IBM AML graph), which does not fit in the 8 GB LinuxONE server.
torch.sparse.mm never builds per-edge messages, so memory stays O(nodes).

Works with the playbook's data contract:
    data = load_view(view, sample=...)   # view = "combined" or "B1".."B5"
    data.x, data.edge_index, data.y, data.train_mask, data.val_mask,
    data.test_mask, data.home_mask, and optionally data.edge_index_dir
If data.edge_index_dir (original sender -> receiver edges) is present, the
model keeps incoming and outgoing money separately (fan-in vs fan-out).
Otherwise it treats data.edge_index as undirected.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score


# --------------------------------------------------------------------------
# Graph -> sparse mean-adjacency matrices
# --------------------------------------------------------------------------
def mean_adj(src: torch.Tensor, dst: torch.Tensor, n: int) -> torch.Tensor:
    """Row-normalised sparse matrix A with A[v, u] = (#edges u->v) / indeg(v).

    A @ x gives, for every node v, the transaction-weighted mean of the
    features of the nodes that sent to v. Duplicate edges are summed by
    coalesce(), so repeated transfers between a pair count proportionally.
    """
    deg = torch.bincount(dst, minlength=n).clamp(min=1).to(torch.float32)
    val = 1.0 / deg[dst]
    A = torch.sparse_coo_tensor(torch.stack([dst, src]), val, (n, n))
    return A.coalesce()


def build_adjs(data) -> list[torch.Tensor]:
    """Return [A_in, A_out] when directions are known, else [A_undirected]."""
    n = data.num_nodes
    ei = getattr(data, "edge_index_dir", None)
    if ei is not None:
        src, dst = ei[0].long(), ei[1].long()
        return [mean_adj(src, dst, n),   # money received: from senders
                mean_adj(dst, src, n)]   # money sent: from receivers
    src, dst = data.edge_index[0].long(), data.edge_index[1].long()
    return [mean_adj(src, dst, n)]


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
class DirSAGELayer(nn.Module):
    """h_v = W_self x_v + sum_r W_r * mean_{u in N_r(v)} x_u  (one W per direction)."""

    def __init__(self, d_in: int, d_out: int, n_rel: int):
        super().__init__()
        self.lin_self = nn.Linear(d_in, d_out)
        self.lin_rel = nn.ModuleList(nn.Linear(d_in, d_out, bias=False) for _ in range(n_rel))

    def forward(self, x, adjs):
        h = self.lin_self(x)
        for A, lin in zip(adjs, self.lin_rel):
            h = h + lin(torch.sparse.mm(A, x))
        return h


class MuleSAGE(nn.Module):
    """2-layer direction-aware GraphSAGE -> one mule logit per account."""

    def __init__(self, d_in: int, n_rel: int, hidden: int = 64, dropout: float = 0.3):
        super().__init__()
        self.l1 = DirSAGELayer(d_in, hidden, n_rel)
        self.l2 = DirSAGELayer(hidden, hidden, n_rel)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.out = nn.Linear(hidden, 1)
        self.dropout = dropout

    def forward(self, x, adjs):
        h = F.relu(self.norm1(self.l1(x, adjs)))
        h = F.dropout(h, self.dropout, self.training)
        h = F.relu(self.norm2(self.l2(h, adjs)))
        h = F.dropout(h, self.dropout, self.training)
        return self.out(h).squeeze(-1)


def sign_features(x: torch.Tensor, adjs: list[torch.Tensor], hops: int = 2) -> torch.Tensor:
    """SIGN: concatenate x with every directed propagation path up to `hops`."""
    feats, frontier = [x], [x]
    with torch.no_grad():
        for _ in range(hops):
            nxt = [torch.sparse.mm(A, f) for f in frontier for A in adjs]
            feats.extend(nxt)
            frontier = nxt
    return torch.cat(feats, dim=1)


class MuleSIGN(nn.Module):
    """MLP on precomputed SIGN features (the guaranteed-to-fit fallback)."""

    def __init__(self, d_in: int, hidden: int = 64, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, adjs=None):
        return self.net(x).squeeze(-1)


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def _pr_auc(y, p):
    return float(average_precision_score(y, p)) if y.sum() > 0 else float("nan")


def train_model(data, train_mask, val_mask, kind: str = "sage", hidden: int = 64,
                lr: float = 0.005, weight_decay: float = 5e-4, epochs: int = 200,
                patience: int = 20, seed: int = 42, threads: int = 3, verbose: bool = True):
    """Full-batch training with early stopping on validation PR-AUC.

    Returns (model, probs for every node from the best epoch, info dict).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(threads)

    adjs = build_adjs(data)
    x = data.x.float()
    if kind == "sign":
        x = sign_features(x, adjs)
        model = MuleSIGN(x.shape[1], hidden)
    elif kind == "sage":
        model = MuleSAGE(x.shape[1], len(adjs), hidden)
    else:
        raise ValueError("kind must be 'sage' or 'sign'")

    y = data.y.float()
    tr, va = train_mask.bool(), val_mask.bool()
    n_pos = int(y[tr].sum())
    if n_pos == 0:
        raise ValueError("No mule accounts in the training mask; cannot train.")
    pos_weight = torch.tensor((tr.sum().item() - n_pos) / n_pos)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    y_va = y[va].numpy()

    best = (-1.0, None, 0)  # (val PR-AUC, state_dict, epoch)
    t0, wait = time.perf_counter(), 0
    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        logits = model(x, adjs)
        loss = F.binary_cross_entropy_with_logits(logits[tr], y[tr], pos_weight=pos_weight)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            p = torch.sigmoid(model(x, adjs))
        ap = _pr_auc(y_va, p[va].numpy())
        if ap > best[0]:
            best = (ap, {k: v.clone() for k, v in model.state_dict().items()}, epoch)
            wait = 0
        else:
            wait += 1
        if verbose and (epoch == 1 or epoch % 10 == 0):
            print(f"  epoch {epoch:3d}  loss {loss.item():.4f}  val PR-AUC {ap:.4f}")
        if wait >= patience:
            break

    model.load_state_dict(best[1])
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(x, adjs)).numpy()
    info = {"kind": kind, "best_epoch": best[2], "val_pr_auc": best[0],
            "epochs_run": epoch, "train_seconds": time.perf_counter() - t0,
            "n_train": int(tr.sum()), "n_train_pos": n_pos}
    if verbose:
        print(f"  best epoch {best[2]}  val PR-AUC {best[0]:.4f}  ({info['train_seconds']:.1f}s)")
    return model, probs, info


# --------------------------------------------------------------------------
# Silo vs combined experiment (playbook definitions 5-7)
# --------------------------------------------------------------------------
def run_setting(load_view, nodes: pd.DataFrame, setting: str, kind: str = "sage",
                insts=("B1", "B2", "B3", "B4", "B5"), sample: bool = False,
                label_scope: str = "all", **train_kw) -> tuple[pd.DataFrame, list]:
    """Train and score in one setting.

    setting = "silo":     per member b, view b, train on b's own train accounts,
                          score b's own accounts.
    setting = "combined": one model on the joined graph, scores everyone.
        label_scope = "all": trains on every member's train labels (default).
        label_scope = "own": per member, joined graph but only own labels
                             (ablation: separates the graph effect from the
                             extra-labels effect).
    Returns (scores DataFrame [acct_key, inst, split, score], info list).
    """
    frames, infos = [], []

    def collect(probs, mask, tag):
        idx = np.where(mask.numpy())[0]
        df = nodes.iloc[idx][["acct_key", "inst", "split"]].copy()
        df["score"] = probs[idx]
        frames.append(df)

    if setting == "combined" and label_scope == "all":
        print("[combined] training on all members' labels")
        data = load_view("combined", sample=sample)
        _, probs, info = train_model(data, data.train_mask, data.val_mask, kind=kind, **train_kw)
        collect(probs, torch.ones(data.num_nodes, dtype=torch.bool), "combined")
        infos.append({"setting": "combined", **info})
    else:
        joined = load_view("combined", sample=sample) if setting == "combined" else None
        for b in insts:
            data = joined if joined is not None else load_view(b, sample=sample)
            home = torch.as_tensor((nodes["inst"] == b).values)
            print(f"[{setting}/{label_scope}] member {b}: "
                  f"{int((data.train_mask & home).sum())} own train accounts")
            _, probs, info = train_model(data, data.train_mask & home, data.val_mask & home,
                                         kind=kind, **train_kw)
            collect(probs, home, b)
            infos.append({"setting": setting, "member": b, **info})
    return pd.concat(frames, ignore_index=True), infos


def save_scores(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"wrote {path} ({len(df)} rows)")


# --------------------------------------------------------------------------
# Quick self-check (Niyati's evaluator is the official one)
# --------------------------------------------------------------------------
def quick_report(scores: pd.DataFrame, nodes: pd.DataFrame, budget: float = 0.01) -> pd.DataFrame:
    """Test-split PR-AUC and recall at the per-member alert budget."""
    df = scores.merge(nodes[["acct_key", "label"]], on="acct_key")
    df = df[df["split"] == "test"]
    rows = []
    for b, g in df.groupby("inst"):
        k = max(1, int(round(budget * len(g))))
        top = g.nlargest(k, "score")
        rows.append({"inst": b, "test_accts": len(g), "mules": int(g.label.sum()),
                     "alerts": k, "caught": int(top.label.sum()),
                     "recall_at_budget": top.label.sum() / max(1, g.label.sum()),
                     "pr_auc": _pr_auc(g.label.values, g.score.values)})
    out = pd.DataFrame(rows)
    tot = {"inst": "ALL", "test_accts": out.test_accts.sum(), "mules": out.mules.sum(),
           "alerts": out.alerts.sum(), "caught": out.caught.sum(),
           "recall_at_budget": out.caught.sum() / max(1, out.mules.sum()),
           "pr_auc": _pr_auc(df.label.values, df.score.values)}
    return pd.concat([out, pd.DataFrame([tot])], ignore_index=True)


# --------------------------------------------------------------------------
# Inference benchmark for the IBM Z slide
# --------------------------------------------------------------------------
def benchmark_inference(model, data, kind: str = "sage", runs: int = 5, threads: int = 3) -> dict:
    """Median wall-clock time to score every account (graph already loaded)."""
    torch.set_num_threads(threads)
    adjs = build_adjs(data)
    x = data.x.float()
    if kind == "sign":
        x = sign_features(x, adjs)
    model.eval()
    times = []
    with torch.no_grad():
        for _ in range(runs):
            t = time.perf_counter()
            model(x, adjs)
            times.append(time.perf_counter() - t)
    med = float(np.median(times))
    return {"accounts": data.num_nodes, "median_seconds": med,
            "accounts_per_second": data.num_nodes / med, "threads": threads}
