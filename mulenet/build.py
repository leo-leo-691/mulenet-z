"""Build the processed graph files from the raw IBM AML CSV (Role 1).

One code path for the sample and the full data, so both follow the playbook
contract exactly:

    from mulenet.build import build_processed
    build_processed(sample_days=3)      # -> data/processed_sample/
    build_processed(sample_days=None)   # -> data/processed/

Writes into the output folder:
    nodes.csv   node_idx, acct_key, bank, inst, label, split
    edges.npz   src, dst (int64 node_idx), amount (float32, Amount Paid),
                ts (int64 unix seconds), fmt (int8), is_laundering (int8),
                fmt_names (payment-format name for each fmt code)

Reuses data/processed/bank_to_inst.csv (columns bank + institution or inst);
never regenerates it. Memory-light: reads the CSV in chunks, two passes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SHARED = Path(__file__).resolve().parents[1]
RAW = SHARED / "data" / "HI-Small_Trans.csv"
FMT_NAMES = ["ACH", "Bitcoin", "Cash", "Cheque", "Credit Card", "Reinvestment", "Wire"]
COLS = ["Timestamp", "From Bank", "Account", "To Bank", "Account.1",
        "Amount Paid", "Payment Format", "Is Laundering"]
DTYPES = {"From Bank": "int64", "To Bank": "int64", "Account": "str", "Account.1": "str",
          "Amount Paid": "float64", "Payment Format": "str", "Is Laundering": "int8"}


def _chunks(raw: Path, cutoff):
    for ch in pd.read_csv(raw, usecols=COLS, dtype=DTYPES, chunksize=500_000):
        ts = pd.to_datetime(ch["Timestamp"], format="%Y/%m/%d %H:%M")
        if cutoff is not None:
            keep = ts < cutoff
            ch, ts = ch[keep.values], ts[keep]
        # bank read as int64 -> "010" becomes 10, so keys match everywhere
        ch = ch.assign(src_key=ch["From Bank"].astype(str) + "_" + ch["Account"],
                       dst_key=ch["To Bank"].astype(str) + "_" + ch["Account.1"],
                       # explicit seconds: pandas 3 stores microseconds by default
                       ts=ts.values.astype("datetime64[s]").astype("int64"))
        yield ch


def build_processed(sample_days: int | None = 3, raw: Path = RAW, out_dir: Path | None = None,
                    seed: int = 42, verbose: bool = True) -> Path:
    raw = Path(raw)
    if out_dir is None:
        out_dir = SHARED / "data" / ("processed_sample" if sample_days else "processed")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    b2i = pd.read_csv(SHARED / "data" / "processed" / "bank_to_inst.csv")
    b2i = b2i.rename(columns={"institution": "inst"})
    bank_to_inst = dict(zip(b2i["bank"].astype("int64"), b2i["inst"]))

    cutoff = None
    if sample_days:
        first = pd.read_csv(raw, usecols=["Timestamp"], nrows=1)["Timestamp"].iloc[0]
        cutoff = pd.Timestamp(first[:10].replace("/", "-")) + pd.Timedelta(days=sample_days)

    # pass 1: every account seen, with its bank
    keys, banks = {}, {}
    for ch in _chunks(raw, cutoff):
        for k, b in ((ch.src_key, ch["From Bank"]), (ch.dst_key, ch["To Bank"])):
            u = pd.DataFrame({"k": k.values, "b": b.values}).drop_duplicates("k")
            banks.update(zip(u.k, u.b))
    acct = np.array(sorted(banks))
    idx = {k: i for i, k in enumerate(acct)}
    if verbose:
        print(f"accounts: {len(acct):,}")

    # pass 2: edges
    parts = {c: [] for c in ("src", "dst", "amount", "ts", "fmt", "is_laundering")}
    fmt_code = {n: i for i, n in enumerate(FMT_NAMES)}
    for ch in _chunks(raw, cutoff):
        parts["src"].append(ch.src_key.map(idx).to_numpy(np.int64))
        parts["dst"].append(ch.dst_key.map(idx).to_numpy(np.int64))
        parts["amount"].append(ch["Amount Paid"].to_numpy(np.float32))
        parts["ts"].append(ch.ts.to_numpy(np.int64))
        f = ch["Payment Format"].map(fmt_code)
        if f.isna().any():
            raise ValueError(f"Unknown payment formats: {set(ch['Payment Format'][f.isna()])}")
        parts["fmt"].append(f.to_numpy(np.int8))
        parts["is_laundering"].append(ch["Is Laundering"].to_numpy(np.int8))
    E = {c: np.concatenate(v) for c, v in parts.items()}
    n = len(acct)

    # labels: account touches any laundering transaction (definition 3)
    label = np.zeros(n, dtype=np.int8)
    lw = E["is_laundering"] == 1
    label[E["src"][lw]] = 1
    label[E["dst"][lw]] = 1

    # stratified 60/20/20 split, fixed seed (definition 4)
    rng = np.random.default_rng(seed)
    split = np.empty(n, dtype=object)
    for lab in (0, 1):
        ids = rng.permutation(np.flatnonzero(label == lab))
        a, b = int(0.6 * len(ids)), int(0.8 * len(ids))
        split[ids[:a]], split[ids[a:b]], split[ids[b:]] = "train", "val", "test"

    bank = np.array([banks[k] for k in acct], dtype=np.int64)
    inst = pd.Series(bank).map(bank_to_inst)
    if inst.isna().any():
        raise ValueError(f"{int(inst.isna().sum())} accounts have a bank missing from bank_to_inst.csv")

    nodes = pd.DataFrame({"node_idx": np.arange(n), "acct_key": acct, "bank": bank,
                          "inst": inst.values, "label": label, "split": split})
    nodes.to_csv(out_dir / "nodes.csv", index=False)
    np.savez_compressed(out_dir / "edges.npz", fmt_names=np.array(FMT_NAMES), **E)
    # cached feature files from an older build are now stale
    for old in out_dir.glob("x_*.npy"):
        old.unlink()

    if verbose:
        t = pd.to_datetime(E["ts"], unit="s")
        print(f"edges: {len(E['src']):,} | laundering: {int(lw.sum()):,} | "
              f"mules: {int(label.sum()):,} ({label.mean():.2%})")
        print(f"time range: {t.min()} -> {t.max()}")
        print("accounts per member:", nodes.inst.value_counts().sort_index().to_dict())
        print(pd.crosstab(nodes.split, nodes.label))
        print(f"saved to {out_dir}")
    return out_dir
