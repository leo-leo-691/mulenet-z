"""Official full-data results (Role 3 task, run by Arindam): metrics.csv + recall figure."""
import os, sys
sys.path.insert(0, '/home/ibm-user/shared')
import pandas as pd
from mulenet.common import load_nodes, laundering_chains
from mulenet.evaluate import results_table, plot_recall, evaluate_scores

R = '/home/ibm-user/shared/results/'
S = R + 'scores/'
os.makedirs(R + 'figures', exist_ok=True)
rd = lambda f: pd.read_csv(S + f, dtype={"acct_key": str})

nodes = load_nodes(False)
chains = laundering_chains(False)
pairs = {
    "Snap ML graph patterns + XGBoost": (rd("gfpxgb_stream_silo.csv"), rd("gfpxgb_stream_combined.csv")),
    "Own labels only (ablation)": (rd("gfpxgb_stream_silo.csv"), rd("gfpxgb_stream_combined_ownlabels.csv")),
    "XGBoost": (rd("xgb_silo.csv"), rd("xgb_combined.csv")),
    "Logistic regression": (rd("lr_silo.csv"), rd("lr_combined.csv")),
}
t = results_table(pairs, nodes, chains)
t.to_csv(R + 'metrics.csv', index=False)

fig_t = t[t.model != "Own labels only (ablation)"].copy()
fig_t["model"] = ["Snap ML patterns\n+ XGBoost", "XGBoost", "Logistic\nregression"]
plot_recall(fig_t, R + 'figures/recall_full.png')

# per-bank view of the headline model, for the deck
s, c = pairs["Snap ML graph patterns + XGBoost"]
pb = evaluate_scores(s, nodes)[["inst", "mules", "recall_at_budget"]].rename(columns={"recall_at_budget": "recall_silo"})
pb["recall_combined"] = evaluate_scores(c, nodes)["recall_at_budget"].values
pb.to_csv(R + 'metrics_per_bank.csv', index=False)

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
cols = ["model", "recall_silo", "recall_combined", "uplift_pts", "uplift_ci_low", "uplift_ci_high",
        "rescued_mules", "pr_auc_silo", "pr_auc_combined", "cross_chains_detected_silo", "cross_chains_detected_combined"]
print(t[cols].round(3).to_string(index=False))
print("\nPer bank (Snap ML model):")
print(pb.round(3).to_string(index=False))
print("\nwrote results/metrics.csv, results/metrics_per_bank.csv, results/figures/recall_full.png")
