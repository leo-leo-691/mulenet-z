"""CPU vs IBM Z on-chip AI accelerator (NNPA) for the same Snap ML boosting model.

Run once per format, each in a fresh process (a model is converted only once):
    python3 -u nnpa2.py compress_trees     # CPU
    python3 -u nnpa2.py zdnn_tensors       # on-chip AI accelerator
    python3 -u nnpa2.py compare            # same alerts? speed?
Same features, same seed, same training rows for both formats.
"""
import os
import sys
import time

SH = os.environ.get("MULENET_SHARED", "/home/ibm-user/shared")
sys.path.insert(0, SH)
import numpy as np

mode = sys.argv[1]
OUT = os.path.join(SH, "results", "nnpa_{}.npz")

if mode == "compare":
    import pandas as pd
    from mulenet.common import load_nodes
    from mulenet.evaluate import alerts
    nodes = load_nodes(False)
    r = {f: np.load(OUT.format(f)) for f in ("compress_trees", "zdnn_tensors")}
    a = {}
    for f, z in r.items():
        al = alerts(nodes[["acct_key", "inst", "split"]].assign(score=z["p"]), nodes)
        a[f] = set(al.acct_key[al.alerted])
        print(f"{f:16s} scoring {len(z['p']):,} accounts: {float(z['t']):.2f} s "
              f"({len(z['p']) / float(z['t']):,.0f} accounts/s), model trained in {float(z['fit']):.0f} s", flush=True)
    same = len(a["compress_trees"] & a["zdnn_tensors"]) / len(a["compress_trees"])
    d = np.abs(r["compress_trees"]["p"] - r["zdnn_tensors"]["p"])
    print(f"max score difference {d.max():.2e} | mean {d.mean():.2e} | same alerts at 1% budget: {same * 100:.1f}%")
    print(f"speed-up: {float(r['compress_trees']['t']) / float(r['zdnn_tensors']['t']):.2f}x", flush=True)
    sys.exit()

from mulenet.gfp import GFP_PARAMS, load_view_gfp
print("loading features", flush=True)
d = load_view_gfp("combined", sample=False, params=GFP_PARAMS, stream=True)
X = np.ascontiguousarray(d.x.numpy(), dtype=np.float32)
y = d.y.numpy().astype(np.float32)
tr = d.train_mask.numpy()

from snapml import BoostingMachineClassifier
print(f"training Snap ML boosting model on {int(tr.sum()):,} accounts", flush=True)
bm = BoostingMachineClassifier(num_round=300, max_depth=6, learning_rate=0.05, n_jobs=3, random_state=42)
t0 = time.perf_counter()
bm.fit(X[tr], y[tr])
fit = time.perf_counter() - t0
print(f"trained in {fit:.0f} s; converting trees to {mode}", flush=True)
bm.optimize_trees(tree_format=mode)
print("converted; scoring", flush=True)

BATCH = 32768                                      # the accelerator format takes at most 2^15 rows per call
col = lambda r: r[:, 1] if r.ndim == 2 else r
best, p = 1e9, None
for _ in range(3):
    t = time.perf_counter()
    p = np.concatenate([col(np.asarray(bm.predict_proba(X[i:i + BATCH]))) for i in range(0, len(X), BATCH)])
    best = min(best, time.perf_counter() - t)
np.savez(OUT.format(mode), p=p, t=best, fit=fit)
print(f"{mode}: scored {len(p):,} accounts in {best:.2f} s", flush=True)
