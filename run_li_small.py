"""Robustness check: the whole pipeline on IBM's LI-Small dataset (lower laundering rate).

Same rules as HI-Small: 5 members by greedy volume balancing of real bank IDs,
same labels, same 60/20/20 split (seed 42), same models and settings, same
causal Snap ML streaming. Writes into data/processed_li/ and results/scores_li/,
so nothing from the HI-Small run is touched.
"""
import os
import sys
import time
from collections import Counter
from pathlib import Path

SH = Path(os.environ.get("MULENET_SHARED", "/home/ibm-user/shared"))
OUT = SH / "data" / "processed_li"
os.environ["MULENET_DATA_DIR"] = str(OUT)          # full-data loaders now read LI-Small
sys.path.insert(0, str(SH))

import pandas as pd
from mulenet.build import build_processed
from mulenet.common import INSTS, load_nodes, laundering_chains
from mulenet.baselines import run_baseline
from mulenet.gfp import GFP_PARAMS, run_gfp_xgb
from mulenet.evaluate import results_table, evaluate_scores

SCORES = SH / "results" / "scores_li"
t0 = time.time()
step = lambda msg: print(f"=== {msg} ({(time.time() - t0) / 60:.0f} min)", flush=True)

# 1) raw file
raw = sorted(p for p in SH.rglob("LI-Small_Trans.csv") if "processed" not in str(p))
if not raw:
    sys.exit("LI-Small_Trans.csv not found under shared/ - ask Vishwajeet where he put it")
raw = raw[0]
step(f"raw: {raw}")

# 2) bank -> member, same greedy volume rule as HI-Small
if not (OUT / "edges.npz").exists():
    OUT.mkdir(parents=True, exist_ok=True)
    vol = Counter()
    for ch in pd.read_csv(raw, usecols=["From Bank", "To Bank"], dtype="int64", chunksize=1_000_000):
        vol.update(ch["From Bank"].value_counts().to_dict())
        vol.update(ch["To Bank"].value_counts().to_dict())
    load = {b: 0 for b in INSTS}
    rows = []
    for bank, v in sorted(vol.items(), key=lambda kv: (-kv[1], kv[0])):
        m = min(INSTS, key=lambda b: (load[b], b))
        load[m] += v
        rows.append((bank, m))
    pd.DataFrame(rows, columns=["bank", "inst"]).to_csv(OUT / "bank_to_inst.csv", index=False)
    step(f"bank mapping: {len(rows):,} banks, volume per member {load}")
    build_processed(sample_days=None, raw=raw, out_dir=OUT, bank_map=OUT / "bank_to_inst.csv")
step("data built")

# 3) models (skip any score file that already exists, so a restart resumes)
def done(name):
    return (SCORES / f"{name}.csv").exists()

for s in ("silo", "combined"):
    if not done(f"xgb_{s}"):
        run_baseline("xgb", s, sample=False, out_dir=str(SCORES))
step("xgboost done")
for setting, scope, name in [("combined", "all", "gfpxgb_stream_combined"),
                             ("silo", "all", "gfpxgb_stream_silo"),
                             ("combined", "own", "gfpxgb_stream_combined_ownlabels")]:
    if not done(name):
        run_gfp_xgb(setting, sample=False, out_dir=str(SCORES), label_scope=scope,
                    params=GFP_PARAMS, suffix="_stream", stream=True)
    step(f"{name} done")

# 4) results
nodes, chains = load_nodes(False), laundering_chains(False)
rd = lambda f: pd.read_csv(SCORES / f, dtype={"acct_key": str})
pairs = {
    "Snap ML graph patterns + XGBoost": (rd("gfpxgb_stream_silo.csv"), rd("gfpxgb_stream_combined.csv")),
    "Own labels only (ablation)": (rd("gfpxgb_stream_silo.csv"), rd("gfpxgb_stream_combined_ownlabels.csv")),
    "XGBoost": (rd("xgb_silo.csv"), rd("xgb_combined.csv")),
}
t = results_table(pairs, nodes, chains)
t.to_csv(SH / "results" / "metrics_li.csv", index=False)
s, c = pairs["Snap ML graph patterns + XGBoost"]
pb = evaluate_scores(s, nodes)[["inst", "mules", "recall_at_budget"]].rename(columns={"recall_at_budget": "recall_silo"})
pb["recall_combined"] = evaluate_scores(c, nodes)["recall_at_budget"].values
pb.to_csv(SH / "results" / "metrics_li_per_bank.csv", index=False)

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
cols = ["model", "recall_silo", "recall_combined", "uplift_pts", "uplift_ci_low", "uplift_ci_high",
        "rescued_mules", "pr_auc_silo", "pr_auc_combined", "cross_chains",
        "cross_chains_detected_silo", "cross_chains_detected_combined"]
print(t[cols].round(3).to_string(index=False))
print(pb.round(3).to_string(index=False))
step("ALL DONE")
