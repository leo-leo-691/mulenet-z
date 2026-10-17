"""The killer visual: a real cross-bank laundering ring that every bank misses alone
and the joined graph catches (HI-Small full data, final model, 1% budget per bank).

Writes:
  results/figures/ring.png    each bank's partial view vs the joined view
  results/ring_example.json   the ring's accounts and transfers (pseudonymised), for the demo page
and prints how many cross-bank chains are caught ONLY by the joined graph.
"""
import hashlib
import json
import os
import sys

SH = os.environ.get("MULENET_SHARED", "/home/ibm-user/shared")
sys.path.insert(0, SH)

import numpy as np
import pandas as pd
from mulenet.common import load_edges, load_nodes, laundering_chains
from mulenet.evaluate import alerts

S = os.path.join(SH, "results", "scores")
rd = lambda f: pd.read_csv(os.path.join(S, f), dtype={"acct_key": str})
nodes, E, chains = load_nodes(False), load_edges(False), laundering_chains(False)
silo = alerts(rd("gfpxgb_stream_silo.csv"), nodes)
joined = alerts(rd("gfpxgb_stream_combined.csv"), nodes)
hit_s = set(silo.acct_key[silo.alerted])
hit_j = set(joined.acct_key[joined.alerted])

# ---- which cross-bank chains are caught only when joined
t = chains[chains.split == "test"].copy()
t["silo"] = t.acct_key.isin(hit_s)
t["joined"] = t.acct_key.isin(hit_j)
per = t.groupby("chain_id").agg(size=("chain_size", "first"), members=("chain_members", "first"),
                                silo=("silo", "max"), joined=("joined", "max"))
cross = per[per.members >= 2]
only = cross[cross.joined & ~cross.silo]
print(f"cross-bank chains with test accounts: {len(cross)} | caught alone: {int(cross.silo.sum())} | "
      f"caught joined: {int(cross.joined.sum())} | caught ONLY when joined: {len(only)} | lost by joining: "
      f"{int((cross.silo & ~cross.joined).sum())}")

# ---- pick a drawable ring: most banks involved, 5-12 accounts
cand = only[(only["size"] >= 5) & (only["size"] <= 12)].sort_values(["members", "size"], ascending=[False, True])
if cand.empty:
    cand = only.sort_values(["members", "size"], ascending=[False, True])
if cand.empty:                       # only happens on tiny test data
    print("WARNING: no chain is caught only when joined; drawing any cross-bank chain instead")
    cand = cross.sort_values(["members", "size"], ascending=[False, True])
cid = cand.index[0]
ring = chains[chains.chain_id == cid].copy()
idx = ring.node_idx.to_numpy()
lw = (E["is_laundering"] == 1) & np.isin(E["src"], idx) & np.isin(E["dst"], idx)
tx = pd.DataFrame({"src": E["src"][lw], "dst": E["dst"][lw], "ts": E["ts"][lw], "fmt": E["fmt"][lw]}).sort_values("ts")
t0 = tx.ts.min()
tx["hours"] = ((tx.ts - t0) / 3600).round(1)
fmt_names = [str(x) for x in E["fmt_names"]]
tx["format"] = [fmt_names[f] for f in tx.fmt]

inst = nodes.set_index("node_idx")["inst"]
key = nodes.set_index("node_idx")["acct_key"]
order = list(dict.fromkeys(list(tx.src) + list(tx.dst)))          # first appearance in time
pseud = {n: hashlib.sha256(f"demo:{key[n]}".encode()).hexdigest()[:6] for n in order}
acc = [{"id": pseud[n], "bank": inst[n], "split": nodes.split[n],
        "alerted_alone": key[n] in hit_s, "alerted_joined": key[n] in hit_j} for n in order]
print(f"\npattern {cid}: {len(order)} accounts at {ring.inst.nunique()} banks "
      f"({', '.join(sorted(ring.inst.unique()))}), {len(tx)} laundering transfers over {tx.hours.max():.0f} h")
for a in acc:
    print(f"  {a['id']}  {a['bank']}  {a['split']:5s}  alone: {'ALERT' if a['alerted_alone'] else '-':5s}  "
          f"joined: {'ALERT' if a['alerted_joined'] else '-'}")
json.dump({"chain_id": int(cid), "accounts": acc,
           "transfers": [{"from": pseud[s], "to": pseud[d], "hours": float(h), "format": f}
                         for s, d, h, f in zip(tx.src, tx.dst, tx.hours, tx.format)],
           "summary": {"cross_chains": int(len(cross)), "caught_alone": int(cross.silo.sum()),
                       "caught_joined": int(cross.joined.sum()), "only_joined": int(len(only))}},
          open(os.path.join(SH, "results", "ring_example.json"), "w"), indent=1)

# ---- figure: what each bank sees vs the joined view
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

banks = sorted(ring.inst.unique())
cmap = dict(zip(sorted(nodes.inst.unique()), ["#2753d8", "#dc2626", "#15803d", "#7c3aed", "#ca8a04"]))
ang = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, len(order), endpoint=False)
pos = {n: (np.cos(a), np.sin(a)) for n, a in zip(order, ang)}
deg = pd.concat([tx.src, tx.dst]).value_counts()
hub = deg.index[0] if deg.iloc[0] == len(tx) else None     # one account in every transfer -> star layout
if hub is not None:
    rest = [n for n in order if n != hub]
    ang = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, len(rest), endpoint=False)
    pos = {n: (np.cos(a), np.sin(a)) for n, a in zip(rest, ang)} | {hub: (0.0, 0.0)}


def draw(ax, title, visible_edge, visible_node, joined_view):
    ax.set_title(title, fontsize=11, loc="left")
    for s, d in zip(tx.src, tx.dst):
        if not visible_edge(s, d):
            continue
        ax.add_patch(FancyArrowPatch(pos[s], pos[d], arrowstyle="-|>", mutation_scale=10, lw=1.4,
                                     color="#c2410c" if joined_view else "#8b97ab",
                                     shrinkA=9, shrinkB=9, connectionstyle="arc3,rad=0.12"))
    for n in order:
        if not visible_node(n):
            continue
        k = key[n]
        alerted = (k in hit_j) if joined_view else (k in hit_s)
        ax.scatter(*pos[n], s=260, color=cmap[inst[n]], edgecolor="#152033" if alerted else "white",
                   linewidth=3 if alerted else 1, zorder=3)
        ax.text(pos[n][0], pos[n][1], inst[n], ha="center", va="center", fontsize=7, color="white", zorder=4)
    ax.set_xlim(-1.35, 1.35); ax.set_ylim(-1.35, 1.35); ax.set_aspect("equal"); ax.axis("off")


# two panels: the bank whose test account is caught only when joined, vs the joined view
caught = [n for n in order if key[n] in hit_j and key[n] not in hit_s]
focus = inst[caught[0]] if caught else banks[0]
lab = nodes.set_index("node_idx")["label"]
fig, axes = plt.subplots(1, 2, figsize=(9, 4.4))
seen = lambda s, d: inst[s] == focus or inst[d] == focus
nodes_seen = lambda n: any(seen(s, d) and n in (s, d) for s, d in zip(tx.src, tx.dst))
k_seen = sum(seen(s, d) for s, d in zip(tx.src, tx.dst))
draw(axes[0], f"What {focus} sees alone: {k_seen} incoming transfer{'s' if k_seen > 1 else ''}, no alert",
     seen, nodes_seen, False)
draw(axes[1], "Joined in enclave: the whole pattern, alert raised", lambda s, d: True, lambda n: True, True)
if hub is not None and nodes.split[hub] == "train" and lab[hub] == 1 and inst[hub] != focus:
    axes[1].text(0, -1.3, f"Centre: the sender, already a known mule at {inst[hub]}",
                 ha="center", fontsize=9, color="#152033")   # another bank's label, never shown to the focus bank
fig.text(0.01, 0.02, "Real laundering pattern from IBM AML HI-Small. Dark outline = alerted at the 1% budget. "
         "A bank sees only transfers touching its own accounts.", fontsize=8, color="#5b6880")
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(os.path.join(SH, "results", "figures", "ring.png"), dpi=200)
print("\nwrote results/figures/ring.png and results/ring_example.json")
