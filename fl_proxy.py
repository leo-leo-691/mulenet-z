"""Federated-learning proxy vs joined graph (decision rule fixed before running).

    python fl_proxy.py        # HI-Small (main dataset)
    python fl_proxy.py li     # LI-Small (robustness), needs the LI run's caches

Silo-only : one model per bank, its own silo view, its own labels   (gfpxgb_stream_silo.csv)
FL proxy  : ONE model on all banks' labels, each account's features from its own silo view
Joined    : ONE model on all banks' labels, features from the joined graph (gfpxgb_stream_combined.csv)

Same test accounts, 1% budget per bank, same XGBoost settings and seed, same paired
bootstrap over test mules (1,000 resamples, seed 42) as the main results.

Decision rule: Joined "adds value beyond FL" only if its recall CI does not overlap the
FL proxy's recall CI; otherwise "matches FL". The paired Joined - FL CI is reported too.
"""
import os
import sys

SH = os.environ.get("MULENET_SHARED", "/home/ibm-user/shared")
LI = len(sys.argv) > 1 and sys.argv[1] == "li"
if LI:
    os.environ["MULENET_DATA_DIR"] = os.path.join(SH, "data", "processed_li")
sys.path.insert(0, SH)

import numpy as np
import pandas as pd
from mulenet.common import load_nodes, laundering_chains
from mulenet.evaluate import alerts, evaluate_scores, chain_detection
from mulenet.gfp import GFP_PARAMS, run_gfp_xgb

S = os.path.join(SH, "results", "scores_li" if LI else "scores")
tag = "li" if LI else "hi"
fl_path = os.path.join(S, "gfpxgb_stream_flproxy.csv")
if not os.path.exists(fl_path):
    run_gfp_xgb("flproxy", sample=False, out_dir=S, params=GFP_PARAMS, suffix="_stream", stream=True)

rd = lambda f: pd.read_csv(os.path.join(S, f), dtype={"acct_key": str})
nodes, chains = load_nodes(False), laundering_chains(False)
runs = {"Silo-only": rd("gfpxgb_stream_silo.csv"),
        "FL proxy": rd("gfpxgb_stream_flproxy.csv"),
        "Joined": rd("gfpxgb_stream_combined.csv")}

# one shared set of bootstrap resamples over the same test mules -> paired comparisons
al = {k: alerts(v, nodes).set_index("acct_key") for k, v in runs.items()}
mules = al["Silo-only"].index[al["Silo-only"].label == 1]
hit = {k: a.loc[mules, "alerted"].to_numpy(float) for k, a in al.items()}
rng = np.random.default_rng(42)
idx = rng.integers(0, len(mules), size=(1000, len(mules)))
boot = {k: h[idx].mean(1) for k, h in hit.items()}
ci = lambda b: np.percentile(b, [2.5, 97.5])

rows = []
for k, v in runs.items():
    ev = evaluate_scores(v, nodes).iloc[-1]                    # the ALL row
    ch = chain_detection(v, nodes, chains)
    lo, hi = ci(boot[k])
    rows.append({"setting": k, "recall_at_budget": hit[k].mean(), "ci_low": lo, "ci_high": hi,
                 "pr_auc": ev.pr_auc, "precision_at_budget": ev.precision_at_budget,
                 "alerts": int(ev.alerts), "caught": int(ev.caught),
                 "cross_bank_chains_caught": ch["cross_bank_detected"], "cross_bank_chains": ch["cross_bank_chains"]})
t = pd.DataFrame(rows)
t.to_csv(os.path.join(SH, "results", f"fl_proxy_{tag}.csv"), index=False)

fmt = lambda r: f"{r.recall_at_budget * 100:.1f}% ({r.ci_low * 100:.1f}-{r.ci_high * 100:.1f})"
print(f"\nDataset: {'LI-Small' if LI else 'HI-Small'} | test mules: {len(mules):,} | budget: 1% per bank\n")
print(f"{'':16s}" + "".join(f"{s:>24s}" for s in t.setting))
print(f"{'recall (95% CI)':16s}" + "".join(f"{fmt(r):>24s}" for r in t.itertuples()))
print(f"{'PR-AUC':16s}" + "".join(f"{r.pr_auc:>24.3f}" for r in t.itertuples()))
print(f"{'precision@budg.':16s}" + "".join(f"{r.precision_at_budget * 100:>23.1f}%" for r in t.itertuples()))
print(f"{'chains caught':16s}" + "".join(f"{f'{r.cross_bank_chains_caught} / {r.cross_bank_chains}':>24s}" for r in t.itertuples()))

print("\nPaired differences (same resamples):")
for a, b in [("FL proxy", "Silo-only"), ("Joined", "Silo-only"), ("Joined", "FL proxy")]:
    d = boot[a] - boot[b]
    lo, hi = ci(d)
    print(f"  {a} - {b}: {(hit[a].mean() - hit[b].mean()) * 100:+.1f} pts (95% CI {lo * 100:+.1f} to {hi * 100:+.1f})")

j, f = t.set_index("setting").loc["Joined"], t.set_index("setting").loc["FL proxy"]
verdict = "joined adds value beyond FL" if j.ci_low > f.ci_high else "matches FL"
print(f"\nDecision rule (non-overlapping recall CIs): {verdict}")
