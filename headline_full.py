"""Quick full-data headline table (Arindam's check). Niyati's notebook makes the official one."""
import sys
sys.path.insert(0, '/home/ibm-user/shared')
import pandas as pd
from mulenet.common import load_nodes, laundering_chains
from mulenet.evaluate import results_table

S = '/home/ibm-user/shared/results/scores/'
rd = lambda f: pd.read_csv(S + f, dtype={"acct_key": str})
nodes = load_nodes(False)
chains = laundering_chains(False)
pairs = {
    "GFP+XGB (Snap ML)": (rd("gfpxgb_stream_silo.csv"), rd("gfpxgb_stream_combined.csv")),
    "GFP+XGB own labels": (rd("gfpxgb_stream_silo.csv"), rd("gfpxgb_stream_combined_ownlabels.csv")),
    "XGBoost": (rd("xgb_silo.csv"), rd("xgb_combined.csv")),
    "LogReg": (rd("lr_silo.csv"), rd("lr_combined.csv")),
}
t = results_table(pairs, nodes, chains)
cols = ["model", "recall_silo", "recall_combined", "uplift_pts", "uplift_ci_low", "uplift_ci_high",
        "rescued_mules", "pr_auc_silo", "pr_auc_combined", "cross_chains",
        "cross_chains_detected_silo", "cross_chains_detected_combined"]
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 20)
print(t[cols].round(3).to_string(index=False))
t.to_csv('/home/ibm-user/shared/results/headline_full_check.csv', index=False)
