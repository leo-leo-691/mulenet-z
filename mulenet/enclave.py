"""MuleNet-Z enclave simulation (Role 4, Vipul). Version 2.

PROTOTYPE: this simulates the trust boundary in software. It is not a
hardware enclave. The production target is IBM Secure Execution for Linux.

Flow (one call each):
    keyring = make_keyring()                      # consortium HMAC key + one key per bank
    subs    = {b: bank_submit(b, keyring, sample=True) for b in INSTS}   # at each bank
    m       = measure_enclave()                   # hash of the enclave code + model
    keys    = release_keys(keyring, m, expected=m)  # banks release keys only to a
                                                    # verified enclave image
    files   = enclave_run(subs, keys, scores_csv=..., out_dir=...)  # join, score, per-bank alerts
    report  = verify_isolation(out_dir, keyring)  # B1 can read only B1's file, etc.

Keys never go to Git: they live in shared/secrets/ (add "secrets/" to .gitignore).
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from mulenet.common import INSTS, SHARED, load_edges, load_nodes

SECRETS = SHARED / "secrets"
BUDGET = 0.01


# ---------------------------------------------------------------- keys
def make_keyring(path: Path = SECRETS / "keyring.json", overwrite: bool = False) -> dict:
    """Consortium pseudonym key (shared by all banks) + one AES-256 key per bank."""
    path = Path(path)
    if path.exists() and not overwrite:
        return load_keyring(path)
    ring = {"hmac": os.urandom(32), "banks": {b: os.urandom(32) for b in INSTS}}
    path.parent.mkdir(parents=True, exist_ok=True)
    enc = {"hmac": base64.b64encode(ring["hmac"]).decode(),
           "banks": {b: base64.b64encode(k).decode() for b, k in ring["banks"].items()}}
    path.write_text(json.dumps(enc))
    os.chmod(path, 0o600)
    return ring


def load_keyring(path: Path = SECRETS / "keyring.json") -> dict:
    enc = json.loads(Path(path).read_text())
    return {"hmac": base64.b64decode(enc["hmac"]),
            "banks": {b: base64.b64decode(k) for b, k in enc["banks"].items()}}


def pseudonym(acct_key: str, hmac_key: bytes) -> str:
    """Same account -> same pseudonym at every bank, so the enclave can join."""
    return hmac.new(hmac_key, acct_key.encode(), hashlib.sha256).hexdigest()[:24]


def _seal(obj, key: bytes, aad: str) -> bytes:
    nonce = os.urandom(12)
    return nonce + AESGCM(key).encrypt(nonce, json.dumps(obj).encode(), aad.encode())


def _open(blob: bytes, key: bytes, aad: str):
    return json.loads(AESGCM(key).decrypt(blob[:12], blob[12:], aad.encode()))


# ---------------------------------------------------------------- bank side
def bank_submit(bank: str, keyring: dict, sample: bool = True, days: float | None = 1.0) -> bytes:
    """What bank `bank` sends: its customers' outgoing transactions (every
    transaction is submitted exactly once, by the sender's bank), its roster of
    own accounts, and the labels it knows for its own train/val accounts.
    Account IDs are pseudonymised; the payload is encrypted with the bank's key.
    days: only the first N days of transactions (keeps the demo light on 8 GB);
    None = everything."""
    nodes, E = load_nodes(sample), load_edges(sample)
    if days:
        keep = E["ts"] < E["ts"].min() + days * 86400
        E = {k: (v[keep] if k != "fmt_names" else v) for k, v in E.items()}
    inst = nodes["inst"].to_numpy()
    ps = np.array([pseudonym(k, keyring["hmac"]) for k in nodes["acct_key"]])
    out = inst[E["src"]] == bank
    home = nodes[nodes.inst == bank]
    known = home[home.split.isin(["train", "val"])]
    payload = {
        "bank": bank,
        "tx": {"src": ps[E["src"][out]].tolist(), "dst": ps[E["dst"][out]].tolist(),
               "amount": E["amount"][out].astype(float).round(2).tolist(),
               "ts": E["ts"][out].tolist(), "fmt": E["fmt"][out].tolist(),
               "dst_member": inst[E["dst"][out]].tolist()},
        "roster": {"p": ps[home.node_idx].tolist(), "split": home.split.tolist()},
        "known_labels": {"p": ps[known.node_idx].tolist(), "label": known.label.astype(int).tolist()},
    }
    return _seal(payload, keyring["banks"][bank], aad=f"submission:{bank}")


# ---------------------------------------------------------------- attestation (simulated)
def measure_enclave(extra_files=()) -> str:
    """Hash of the code (and model files) that will run inside the enclave.
    In IBM Secure Execution the hardware produces this measurement."""
    h = hashlib.sha256()
    for f in [Path(__file__), Path(__file__).with_name("gnn.py"), *map(Path, extra_files)]:
        h.update(f.read_bytes())
    return h.hexdigest()


def release_keys(keyring: dict, measurement: str, expected: str) -> dict:
    """Banks release their data keys only if the enclave image is the one they approved."""
    if not hmac.compare_digest(measurement, expected):
        raise PermissionError("Attestation failed: enclave image does not match the approved measurement")
    return dict(keyring["banks"])


# ---------------------------------------------------------------- inside the enclave
def enclave_run(submissions: dict, bank_keys: dict, scores_csv: str, out_dir: Path,
                hmac_key: bytes, budget: float = BUDGET, split: str = "test") -> dict:
    """Decrypt, join into one graph, score, and write one encrypted alert file per bank.

    Scoring: the joined graph here is exactly the 'combined' view, so we use the
    combined model's scores (scores_csv, keyed by pseudonym inside the enclave).
    """
    t0 = time.perf_counter()
    subs = {b: _open(blob, bank_keys[b], f"submission:{b}") for b, blob in submissions.items()}
    t_dec = time.perf_counter() - t0

    tx = pd.concat([pd.DataFrame(s["tx"]) for s in subs.values()], ignore_index=True)
    roster = pd.concat([pd.DataFrame({**s["roster"], "bank": b}) for b, s in subs.items()], ignore_index=True)

    sc = pd.read_csv(scores_csv, dtype={"acct_key": str})
    sc["p"] = [pseudonym(k, hmac_key) for k in sc["acct_key"]]   # model output keyed by pseudonym
    roster = roster.merge(sc[["p", "score"]], on="p", how="left")

    # evidence from the joined graph (only visible inside the enclave)
    inflow = tx.groupby("dst").agg(amt_in=("amount", "sum"), n_in=("amount", "size"))
    outflow = tx.groupby("src").agg(amt_out=("amount", "sum"), n_out=("amount", "size"),
                                    out_members=("dst_member", "nunique"))
    src_member = tx.merge(roster[["p", "bank"]].rename(columns={"p": "src"}), on="src", how="left")
    in_members = src_member.groupby("dst")["bank"].nunique().rename("in_members")

    os.makedirs(out_dir, exist_ok=True)
    written = {}
    for b, g in roster[roster.split == split].groupby("bank"):
        k = max(1, round(budget * len(g)))
        top = g.nlargest(k, "score").set_index("p")
        top = top.join(inflow).join(outflow).join(in_members).fillna(0)
        alerts = [{"account_pseudonym": p, "score": round(float(r.score), 4), "rank": i + 1,
                   "evidence": {"amount_in": round(r.amt_in, 2), "amount_out": round(r.amt_out, 2),
                                "pass_through": round(r.amt_out / r.amt_in, 3) if r.amt_in else None,
                                "senders_from_members": int(r.in_members),
                                "receivers_at_members": int(r.out_members)}}
                  for i, (p, r) in enumerate(top.iterrows())]
        path = Path(out_dir) / f"{b}_alerts.enc"
        path.write_bytes(_seal({"bank": b, "alerts": alerts}, bank_keys[b], aad=f"alerts:{b}"))
        written[b] = str(path)
    return {"files": written, "joined_transactions": len(tx), "joined_accounts": len(roster),
            "decrypt_seconds": t_dec, "total_seconds": time.perf_counter() - t0}


# ---------------------------------------------------------------- bank side again
def read_my_alerts(bank: str, keyring: dict, out_dir: Path, sample: bool = True) -> pd.DataFrame:
    """A bank decrypts its own file and maps pseudonyms back to its own accounts."""
    data = _open((Path(out_dir) / f"{bank}_alerts.enc").read_bytes(), keyring["banks"][bank], f"alerts:{bank}")
    nodes = load_nodes(sample)
    mine = nodes[nodes.inst == bank]
    back = {pseudonym(k, keyring["hmac"]): k for k in mine["acct_key"]}
    df = pd.json_normalize(data["alerts"])
    df.insert(0, "acct_key", df["account_pseudonym"].map(back))
    return df


def verify_isolation(out_dir: Path, keyring: dict, sample: bool = True) -> pd.DataFrame:
    """Proof for the demo: each bank reads only its own file, every alert is its
    own customer, and no other bank's key can open it."""
    rows = []
    for b in INSTS:
        path = Path(out_dir) / f"{b}_alerts.enc"
        if not path.exists():
            continue
        df = read_my_alerts(b, keyring, out_dir, sample)
        others_open = 0
        for o in INSTS:
            if o == b:
                continue
            try:
                _open(path.read_bytes(), keyring["banks"][o], f"alerts:{b}")
                others_open += 1
            except InvalidTag:
                pass
        rows.append({"bank": b, "alerts": len(df), "all_own_accounts": bool(df.acct_key.notna().all()),
                     "other_banks_could_open": others_open})
    rep = pd.DataFrame(rows)
    assert rep.all_own_accounts.all() and (rep.other_banks_could_open == 0).all(), rep
    return rep
