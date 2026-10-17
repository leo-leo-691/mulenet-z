"""Why was this account flagged? Plain-language reasons for every alert (full data).

Re-trains the final combined model (Snap ML graph patterns + XGBoost, same
settings and seed as gfpxgb_stream_combined.csv), then uses XGBoost's exact
per-feature contributions (TreeSHAP, pred_contribs) for each alerted test
account. The top contributions become reasons an investigator can read.

Writes:
  results/alerts_explained.csv       one row per alert: bank, rank, score, 3 reasons
  results/alert_drivers.csv          what drives alerts overall (mean contribution by family)
  results/figures/alert_drivers.png  chart of the above, for the deck
"""
import os
import sys
import time

SH = os.environ.get("MULENET_SHARED", "/home/ibm-user/shared")
sys.path.insert(0, SH)

import numpy as np
import pandas as pd

from mulenet.common import INSTS, _view_mask, account_features, load_edges, load_nodes, LOG_COLS
from mulenet.gfp import GFP_PARAMS, load_view_gfp, data_dir
from mulenet.evaluate import alerts

t0 = time.time()
log = lambda m: print(f"[{time.time() - t0:5.0f}s] {m}", flush=True)
R = os.path.join(SH, "results")
os.makedirs(os.path.join(R, "figures"), exist_ok=True)

# ------------------------------------------------------------ data + model
nodes = load_nodes(False)
d = load_view_gfp("combined", sample=False, params=GFP_PARAMS, stream=True)   # cached features
X, y = d.x.numpy(), d.y.numpy()
base_names = list(d.feature_names)
k = (X.shape[1] - len(base_names)) // 2
# Snap ML GFP output order: fan-in, fan-out, degree-in, degree-out, scatter-gather, temporal cycle
fams = (["fan_in"] * 5 + ["fan_out"] * 5 + ["deg_in"] * 5 + ["deg_out"] * 5
        + ["scatter_gather"] * 4 + ["temp_cycle"] * 4)
assert len(fams) == k, f"expected 28 graph features per side, got {k}"
names = base_names + [f"gfp_out_{j}" for j in range(k)] + [f"gfp_in_{j}" for j in range(k)]
fam_of = {f"gfp_out_{j}": fams[j] for j in range(k)} | {f"gfp_in_{j}": fams[j] for j in range(k)}
log(f"features: {len(base_names)} account + {2 * k} Snap ML graph-pattern")

tr, va = d.train_mask.numpy(), d.val_mask.numpy()
from xgboost import XGBClassifier, DMatrix
pos = int(y[tr].sum())
m = XGBClassifier(n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.8,
                  colsample_bytree=0.8, tree_method="hist", n_jobs=3,
                  scale_pos_weight=(tr.sum() - pos) / pos, eval_metric="aucpr",
                  early_stopping_rounds=40, random_state=42)
m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
p = m.predict_proba(X)[:, 1]
log(f"model trained ({m.best_iteration + 1} trees)")

saved = pd.read_csv(os.path.join(R, "scores", "gfpxgb_stream_combined.csv"), dtype={"acct_key": str})
same = nodes[["acct_key"]].assign(p=p).merge(saved, on="acct_key")
log(f"check vs final scores file: rank correlation {same.p.rank().corr(same.score.rank()):.4f}")

# ------------------------------------------------------------ alerts + contributions
scores = nodes[["acct_key", "inst", "split"]].assign(score=p)
al = alerts(scores, nodes)
al = al[al.alerted].merge(nodes[["acct_key", "node_idx"]], on="acct_key")
idx = al.node_idx.to_numpy()
contrib = m.get_booster().predict(DMatrix(X[idx]), pred_contribs=True)[:, :-1]   # drop bias
log(f"{len(idx):,} alerts explained")

# raw values for readable evidence (same transactions the model saw)
E = load_edges(False)
raw = account_features(E, _view_mask(E, nodes["inst"].to_numpy(), "combined"),
                       nodes["inst"].to_numpy(), len(nodes))
for c in raw.columns:
    if c in LOG_COLS:
        raw[c] = np.expm1(raw[c])
inst = nodes["inst"].to_numpy()
src, dst = E["src"], E["dst"]
# counterparties at OTHER member banks: only visible in the joined graph
other = inst[src] != inst[dst]
banks_in = pd.Series(inst[src[other]]).groupby(dst[other]).nunique()
banks_out = pd.Series(inst[dst[other]]).groupby(src[other]).nunique()

FAMILY = {"fan_in": "fan-in (many senders converge)", "fan_out": "fan-out (money spread to many)",
          "deg_in": "many incoming links", "deg_out": "many outgoing links",
          "scatter_gather": "scatter-gather (split, then regrouped)", "temp_cycle": "money moving in a cycle"}


def family(n):
    if n in fam_of:
        return "graph: " + FAMILY[fam_of[n]]
    return {"in_count": "incoming volume", "out_count": "outgoing volume", "uniq_in": "many distinct senders",
            "uniq_out": "many distinct receivers", "amt_in": "incoming value", "amt_out": "outgoing value",
            "mean_in": "transaction size", "mean_out": "transaction size", "max_in": "transaction size",
            "max_out": "transaction size", "pass_through": "pass-through", "first_in_to_last_out_h": "speed in to out",
            "receives_then_sends": "pass-through", "active_days": "activity span", "n_formats": "payment formats",
            "cross_member_share": "cross-bank activity", "share_ach": "payment formats",
            "share_cash": "payment formats", "in_view": "activity span"}.get(n, n)


active = (raw["in_count"] + raw["out_count"]) > 0
typical = raw[active].median()                     # the middle active account
typical_nz = raw[active].replace(0, np.nan).median()  # for amounts: middle of accounts that have any
SIZE = {"mean_in": "average incoming transfer", "mean_out": "average outgoing transfer",
        "max_in": "largest incoming transfer", "max_out": "largest outgoing transfer",
        "amt_in": "total received", "amt_out": "total sent"}


def reason(n, i):
    r = raw.iloc[i]
    if n in SIZE:
        x = r[n] / typical_nz[n] if typical_nz[n] else np.nan
        return f"{SIZE[n]} is {x:.1f}x the typical account" if x >= 1 else f"{SIZE[n]} is unusually small ({x:.2f}x typical)"
    if n in fam_of:
        side = "receiving" if n.startswith("gfp_in") else "sending"
        return f"Snap ML found it {side} inside a {FAMILY[fam_of[n]]} pattern"
    v = r.get(n, np.nan)
    t = typical.get(n, np.nan)
    tt = f" (typical account: {t:.0f})" if np.isfinite(t) else ""
    return {
        "uniq_in": f"receives from {v:.0f} different accounts{tt}",
        "uniq_out": f"sends to {v:.0f} different accounts{tt}",
        "in_count": f"{v:.0f} incoming transfers{tt}",
        "out_count": f"{v:.0f} outgoing transfers{tt}",
        "pass_through": f"passes on {min(v, 10) * 100:.0f}% of what it receives",
        "first_in_to_last_out_h": (f"money goes out within {v:.0f} h of the first money in" if v < 48
                                   else f"keeps receiving and sending money over {v / 24:.0f} days"),
        "receives_then_sends": "receives money and then sends it on",
        "cross_member_share": f"{v * 100:.0f}% of its transfers cross to other banks",
        "active_days": f"active on {v:.0f} days{tt}",
        "n_formats": f"uses {v:.0f} payment formats{tt}",
        "share_ach": f"{v * 100:.0f}% of transfers by ACH (typical account: {t * 100:.0f}%)",
        "share_cash": f"{v * 100:.0f}% of transfers in cash (typical account: {t * 100:.0f}%)",
    }.get(n, f"unusual {family(n)}")


rows = []
for a, (i, c) in enumerate(zip(idx, contrib)):
    top, seen = [], set()
    for j in np.argsort(-c):                     # strongest reasons, one per family
        if c[j] <= 0 or len(top) == 3:
            break
        if family(names[j]) not in seen:
            seen.add(family(names[j]))
            top.append(j)
    rec = al.iloc[a]
    rows.append({"bank": rec.inst, "rank": int(rec["rank"]), "acct_key": rec.acct_key,
                 "score": round(float(rec.score), 4), "test_label_mule": int(rec.label),
                 "counterparty_banks_in": int(banks_in.get(i, 0)), "counterparty_banks_out": int(banks_out.get(i, 0)),
                 **{f"reason_{t + 1}": reason(names[j], i) for t, j in enumerate(top)}})
ex = pd.DataFrame(rows).sort_values(["bank", "rank"])
ex.to_csv(os.path.join(R, "alerts_explained.csv"), index=False)

# ------------------------------------------------------------ what drives alerts overall
fam = pd.Series([family(n) for n in names])
drv = (pd.DataFrame(np.abs(contrib), columns=names).T.assign(f=fam.values).groupby("f").sum().sum(axis=1))
drv = (drv / drv.sum()).sort_values(ascending=False).rename("share_of_alert_reasoning")
drv.to_csv(os.path.join(R, "alert_drivers.csv"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
top = drv.head(10)[::-1]
fig, ax = plt.subplots(figsize=(7, 4))
ax.barh(top.index, top.values * 100, color=["#1f6feb" if s.startswith("graph") else "#9aa5b1" for s in top.index])
ax.set_xlabel("Share of the model's reasoning across all alerts (%)")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(R, "figures", "alert_drivers.png"), dpi=200)

# ------------------------------------------------------------ report
graph_share = drv[[i for i in drv.index if i.startswith("graph")]].sum()
print("\nWhat drives alerts (share of reasoning):")
print((drv.head(12) * 100).round(1).to_string())
print(f"\nSnap ML graph patterns: {graph_share * 100:.1f}% of all alert reasoning")
print(f"alerts with a counterparty at another bank: {((ex.counterparty_banks_in + ex.counterparty_banks_out) > 0).mean() * 100:.0f}%")
print("\nGraph-feature check (share of accounts with a non-zero value, by family):")
G = X[:, len(base_names):]          # standardised; a raw 0 sits at each column's minimum
G = G > (G.min(0) + 1e-6)
for f in FAMILY:
    cols = [j for j in range(k) if fams[j] == f] + [k + j for j in range(k) if fams[j] == f]
    nz = G[:, cols].any(1).mean()
    print(f"  {f:15s} {nz * 100:5.1f}%")
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 70)
print("\nExample alerts (top 2 per bank):")
print(ex.groupby("bank").head(2)[["bank", "rank", "score", "test_label_mule", "reason_1", "reason_2", "reason_3"]].to_string(index=False))
log("ALL DONE")
