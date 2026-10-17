"""Official evaluator (Role 3, Niyati). Every model goes through these functions.

    from mulenet.evaluate import evaluate_scores, results_table, chain_detection, uplift_ci
Score files follow the contract: acct_key, inst, split, score.
Labels come from nodes.csv through mulenet.common.load_nodes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

BUDGET = 0.01  # definition 8: each member alerts on its top 1% of test accounts


def _join(scores: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    need = {"acct_key", "inst", "split", "score"}
    missing = need - set(scores.columns)
    if missing:
        raise ValueError(f"score file is missing {sorted(missing)}")
    test = nodes.loc[nodes.split == "test", ["acct_key", "inst", "label"]]
    s = scores.loc[scores.split == "test", ["acct_key", "score"]]
    df = test.merge(s, on="acct_key", how="left", validate="one_to_one")
    if df.score.isna().any():
        raise ValueError(f"{int(df.score.isna().sum())} test accounts have no score")
    return df


def alerts(scores: pd.DataFrame, nodes: pd.DataFrame, budget: float = BUDGET) -> pd.DataFrame:
    """Test accounts with an alerted flag: top `budget` share per member."""
    df = _join(scores, nodes)
    df["rank"] = df.groupby("inst")["score"].rank(ascending=False, method="first")
    k = df.groupby("inst")["acct_key"].transform("size").mul(budget).round().clip(lower=1)
    df["alerted"] = df["rank"] <= k
    return df


def evaluate_scores(scores, nodes, budget: float = BUDGET) -> pd.DataFrame:
    """Per-member and overall metrics on the test split."""
    df = alerts(scores, nodes, budget)
    rows = []
    for inst, g in list(df.groupby("inst")) + [("ALL", df)]:
        tp = int((g.alerted & (g.label == 1)).sum())
        n_alert, n_mule = int(g.alerted.sum()), int(g.label.sum())
        p = tp / max(n_alert, 1)
        r = tp / max(n_mule, 1)
        rows.append({"inst": inst, "test_accounts": len(g), "mules": n_mule, "alerts": n_alert,
                     "caught": tp, "precision_at_budget": p, "recall_at_budget": r,
                     "f1_at_budget": 2 * p * r / max(p + r, 1e-12),
                     "pr_auc": average_precision_score(g.label, g.score) if n_mule else np.nan,
                     "roc_auc": roc_auc_score(g.label, g.score) if 0 < n_mule < len(g) else np.nan})
    return pd.DataFrame(rows)


def uplift_ci(silo, combined, nodes, budget=BUDGET, n_boot=1000, seed=42):
    """Recall uplift (combined - silo) with a paired 95% bootstrap CI over test mules."""
    a = alerts(silo, nodes, budget).set_index("acct_key")
    b = alerts(combined, nodes, budget).set_index("acct_key")
    mules = a.index[a.label == 1]
    hit_a = a.loc[mules, "alerted"].to_numpy(float)
    hit_b = b.loc[mules, "alerted"].to_numpy(float)
    diff = hit_b.mean() - hit_a.mean()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(mules), size=(n_boot, len(mules)))
    boots = hit_b[idx].mean(1) - hit_a[idx].mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"recall_silo": hit_a.mean(), "recall_combined": hit_b.mean(),
            "uplift_pts": diff, "ci_low": lo, "ci_high": hi,
            "rescued_mules": int(((hit_b == 1) & (hit_a == 0)).sum()),
            "lost_mules": int(((hit_a == 1) & (hit_b == 0)).sum())}


def chain_detection(scores, nodes, chains: pd.DataFrame, budget=BUDGET) -> dict:
    """Definition 9: a chain is detected if any of its test accounts is alerted."""
    al = alerts(scores, nodes, budget)[["acct_key", "alerted"]]
    c = chains[chains.split == "test"].merge(al, on="acct_key")
    per = c.groupby("chain_id").agg(members=("chain_members", "first"),
                                    detected=("alerted", "max"),
                                    share_alerted=("alerted", "mean"))
    cross = per[per.members >= 2]
    return {"chains_with_test_accounts": len(per), "chains_detected": int(per.detected.sum()),
            "cross_bank_chains": len(cross), "cross_bank_detected": int(cross.detected.sum()),
            "mean_share_alerted": float(per.share_alerted.mean())}


def results_table(pairs: dict, nodes, chains=None, budget=BUDGET) -> pd.DataFrame:
    """pairs = {"GNN": (silo_df, combined_df), "XGBoost": (...), ...} -> the 2x2 headline table."""
    rows = []
    for model, (silo, comb) in pairs.items():
        es, ec = evaluate_scores(silo, nodes, budget), evaluate_scores(comb, nodes, budget)
        u = uplift_ci(silo, comb, nodes, budget)
        row = {"model": model,
               "recall_silo": u["recall_silo"], "recall_combined": u["recall_combined"],
               "uplift_pts": u["uplift_pts"], "uplift_ci_low": u["ci_low"], "uplift_ci_high": u["ci_high"],
               "uplift_relative": u["uplift_pts"] / max(u["recall_silo"], 1e-12),
               "rescued_mules": u["rescued_mules"],
               "pr_auc_silo": es.iloc[-1].pr_auc, "pr_auc_combined": ec.iloc[-1].pr_auc}
        if chains is not None:
            cs, cc = chain_detection(silo, nodes, chains, budget), chain_detection(comb, nodes, chains, budget)
            row.update({"cross_chains": cs["cross_bank_chains"],
                        "cross_chains_detected_silo": cs["cross_bank_detected"],
                        "cross_chains_detected_combined": cc["cross_bank_detected"]})
        rows.append(row)
    return pd.DataFrame(rows)


def plot_recall(table: pd.DataFrame, path: str):
    """Grouped bars: recall at budget, silo vs combined, per model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.arange(len(table))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - 0.2, table.recall_silo * 100, 0.4, label="Each bank alone (silo)", color="#9aa5b1")
    ax.bar(x + 0.2, table.recall_combined * 100, 0.4, label="Joined in enclave (combined)", color="#1f6feb")
    for i, r in table.reset_index().iterrows():
        ax.text(i + 0.2, r.recall_combined * 100 + 1, f"{r.uplift_pts*100:+.1f} pts", ha="center", fontsize=9)
    ax.set_xticks(x, table.model)
    ax.set_ylabel("Mules caught at 1% alert budget (%)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
