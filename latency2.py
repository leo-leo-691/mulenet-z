"""Can MuleNet-Z keep up with live payments? Per-transaction latency on LinuxONE (~5 min).

1. Snap ML graph patterns: the graph is warmed with one day of transactions,
   then NEW transactions are pushed one at a time (and in small batches) through
   transform(), exactly as a live system would. Snap ML updates its graph and returns the
   pattern features of the new transaction.
2. XGBoost: scoring one account, and a batch of 1,000, with the final model.
Run when nothing else heavy is running (seeds.py finished).
"""
import os
import sys
import time

SH = os.environ.get("MULENET_SHARED", "/home/ibm-user/shared")
sys.path.insert(0, SH)
import numpy as np
from mulenet.common import load_edges
from mulenet.gfp import GFP_PARAMS, _edge_array, _gfp

E = load_edges(False)
X, _ = _edge_array(E, np.ones(len(E["src"]), bool), int(E["ts"].min()))
del E
print(f"loaded {len(X):,} transactions", flush=True)
# warm the graph with 1 day of traffic, streamed in chunks as in production (minutes since start)
start = np.searchsorted(X[:, 3], 8 * 1440)
warm_end = np.searchsorted(X[:, 3], 9 * 1440)
gp = _gfp(GFP_PARAMS)
t = time.perf_counter()
for i, s0 in enumerate(range(start, warm_end, 50_000)):
    part = X[s0:min(s0 + 50_000, warm_end)]
    gp.fit_transform(part) if i == 0 else gp.transform(part)
    print(f"  warming: {min(s0 + 50_000, warm_end) - start:,}/{warm_end - start:,} ({time.perf_counter() - t:.0f} s)", flush=True)
print(f"graph warmed with {warm_end - start:,} transactions (1 day) in {time.perf_counter() - t:.0f} s", flush=True)

nxt = X[warm_end:warm_end + 20000]
lat = []
for i in range(500):                                   # one transaction at a time
    t = time.perf_counter()
    gp.transform(nxt[i:i + 1])
    lat.append((time.perf_counter() - t) * 1000)
lat = np.array(lat)
print(f"Snap ML, one transaction at a time (500): median {np.median(lat):.2f} ms, "
      f"p95 {np.percentile(lat, 95):.2f} ms, p99 {np.percentile(lat, 99):.2f} ms", flush=True)
for b in (100, 1000, 10000):
    seg = nxt[500:500 + b] if b < 10000 else nxt[5000:15000]
    t = time.perf_counter()
    gp.transform(seg)
    dt = time.perf_counter() - t
    print(f"Snap ML, batch of {b:,}: {dt * 1000:.0f} ms total, {b / dt:,.0f} transactions/s", flush=True)

import xgboost as xgb
path = os.path.join(SH, "results", "final_model.json")
if os.path.exists(path):
    bst = xgb.Booster()
    bst.load_model(path)
    nf = bst.num_features()
    rng = np.random.default_rng(0)
    one = rng.standard_normal((1, nf)).astype(np.float32)
    many = rng.standard_normal((1000, nf)).astype(np.float32)
    bst.inplace_predict(one)                            # warm up
    l1 = []
    for _ in range(500):
        t = time.perf_counter()
        bst.inplace_predict(one)
        l1.append((time.perf_counter() - t) * 1000)
    t = time.perf_counter()
    bst.inplace_predict(many)
    dt = (time.perf_counter() - t) * 1000
    print(f"XGBoost final model, one account: median {np.median(l1):.2f} ms, p99 {np.percentile(l1, 99):.2f} ms; "
          f"batch of 1,000: {dt:.1f} ms", flush=True)
else:
    print("final_model.json not found (it is written by nnpa_bench.py); skipping the XGBoost part")
print(f"\nfor scale: 5,078,345 transactions over ~18 days is about {5078345 / (18 * 86400):.1f} transactions/s on average")
