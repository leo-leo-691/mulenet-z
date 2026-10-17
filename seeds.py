"""Is the result stable across training seeds? (HI-Small, full data, ~10 min)

Retrains the three settings with seeds 1, 2, 3 (the reported run used 42):
  Silo-only  one XGBoost per bank on its own silo view and labels
  FL proxy   one XGBoost on all labels, each account's features from its own bank's view
  Joined     one XGBoost on the joined graph and all labels
Same cached Snap ML features, same XGBoost settings, same test accounts and 1% budget.
"""
import os
import sys
import time

SH = os.environ.get("MULENET_SHARED", "/home/ibm-user/shared")
sys.path.insert(0, SH)

import numpy as np
import pandas as pd
from mulenet.baselines import _fit_predict
from mulenet.common import INSTS, _view_mask, account_features, load_edges, load_nodes
from mulenet.evaluate import alerts
from mulenet.gfp import GFP_PARAMS, _gfp_cache_path, load_view_gfp

SEEDS = [1, 2, 3]
t0 = time.time()
log = lambda m: print(f"[{(time.time() - t0) / 60:4.1f} min] {m}", flush=True)
nodes = load_nodes(False)
inst = nodes["inst"].to_numpy()
y = nodes["label"].to_numpy()
split = nodes["split"].to_numpy()
tr_all, va_all = split == "train", split == "val"


def recall(p):
    a = alerts(nodes[["acct_key", "inst", "split"]].assign(score=p), nodes)
    return a[a.label == 1].alerted.mean()


# features, built once
Xj = load_view_gfp("combined", sample=False, params=GFP_PARAMS, stream=True).x.numpy()
silo = {}
E = load_edges(False)
Xfl = np.zeros((len(nodes), Xj.shape[1]), np.float32)
for b in INSTS:
    d = load_view_gfp(b, sample=False, params=GFP_PARAMS, stream=True)
    home = d.home_mask.numpy()
    silo[b] = (d.x.numpy(), home, d.train_mask.numpy() & home, d.val_mask.numpy() & home)
    G = np.load(_gfp_cache_path(b, False, GFP_PARAMS, True))
    F = account_features(E, _view_mask(E, inst, b), inst, len(nodes)).fillna(0).to_numpy(np.float32)
    Xfl[inst == b] = np.hstack([F, G])[inst == b]
del E
log("features ready")

rows = []
for s in SEEDS:
    p_silo = np.zeros(len(nodes))
    for b, (Xb, home, trb, vab) in silo.items():
        p_silo[home] = _fit_predict("xgb", Xb, y, trb, vab, seed=s)[home]
    p_fl = _fit_predict("xgb", Xfl, y, tr_all, va_all, seed=s)
    p_j = _fit_predict("xgb", Xj, y, tr_all, va_all, seed=s)
    r = {"seed": s, "silo": recall(p_silo), "fl_proxy": recall(p_fl), "joined": recall(p_j)}
    rows.append(r)
    log(f"seed {s}: silo {r['silo'] * 100:.1f}% | FL proxy {r['fl_proxy'] * 100:.1f}% | joined {r['joined'] * 100:.1f}%")

t = pd.DataFrame(rows)
t["joined_minus_silo"] = t.joined - t.silo
t["joined_minus_fl"] = t.joined - t.fl_proxy
t.to_csv(os.path.join(SH, "results", "seeds_hi.csv"), index=False)
print("\nrecall at 1% budget, percentage points (seed 42 reported: silo 19.0, FL 22.6, joined 25.0)")
print((t.set_index("seed") * 100).round(1).to_string())
m, sd = t.mean() * 100, t.std() * 100
print(f"\nmean ± sd over seeds: joined - silo {m.joined_minus_silo:+.1f} ± {sd.joined_minus_silo:.1f} pts | "
      f"joined - FL {m.joined_minus_fl:+.1f} ± {sd.joined_minus_fl:.1f} pts")
