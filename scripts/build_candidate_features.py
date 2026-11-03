"""
Candidate-level residual features, one row per race (2010-2026):
data/processed/candidate_features.csv

    PYTHONPATH=src python3 scripts/build_candidate_features.py

1. Money (FEC "All candidates" files, data/raw/fec/weball{yy}.txt):
   each party's nominee is matched by last name within the same state/district/office
   (names from the results file for 2010-2024, the incumbency file for 2026); fallback
   = the party's top fundraiser there.
     fec_fund_share     D share of (receipts - self-funding), minus 0.5
     fec_log_ratio      log((D + 50k) / (R + 50k)) of the same
     fec_cash_share     D share of cash on hand at the end of the period, minus 0.5
     fec_self_fund_diff (D self-funding - R self-funding) / (D + R receipts)
   Blank when either nominee can't be found. Timing caveat: past cycles are two-year
   totals, so they include some post-election money; 2026 is as of the latest reports.

2. Past overperformance (prev_overperf): when an incumbent runs, the baseline's
   out-of-fold residual from that same person's previous race for this chamber in
   this state (matched by last name of the previous winner), in D-margin terms.
   Blank for open seats or when no earlier race is found (e.g. 2010 House, first terms
   won in specials). Captures personal brands (Collins, Manchin) the baseline misses.
"""

from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from election_prediction.baseline import baseline_residuals  # noqa: E402
from election_prediction.config import load_config, repo_path  # noqa: E402
from election_prediction.data import KEYS, load_fec_weball, standardize_keys  # noqa: E402
from election_prediction.train import build_training_table, load_baseline_inputs  # noqa: E402

SUFFIXES = {"JR", "SR", "II", "III", "IV", "MD", "DR", "MR", "MRS", "MS", "PHD"}


def last_name(n) -> str:
    """'COLLINS, SUSAN M.' / 'Susan M. Collins' / 'Joe Manchin, III' -> 'COLLINS' / 'MANCHIN'."""
    if not isinstance(n, str) or not n.strip():
        return ""
    s = unicodedata.normalize("NFKD", n).encode("ascii", "ignore").decode().upper()
    s = s.split(";")[0]                                   # several candidates listed: first
    s = re.sub(r"\(.*?\)|\".*?\"", " ", s)
    parts = [p.strip() for p in s.split(",")]
    if len(parts) > 1 and re.sub(r"[^A-Z]", "", parts[1]) not in SUFFIXES and len(parts[0].split()) <= 2:
        cand = parts[0]                                   # "LAST, FIRST" format
    else:
        cand = parts[0]
    toks = [t for t in re.sub(r"[^A-Z ]", " ", cand).split() if t not in SUFFIXES and len(t) > 1]
    if not toks:
        return ""
    return toks[0] if (len(parts) > 1 and len(parts[0].split()) <= 2) else toks[-1]


def money_features(races: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for y in sorted(races["year"].unique()):
        p = ROOT / "data" / "raw" / "fec" / f"weball{y % 100:02d}.txt"
        if p.exists():
            frames.append(load_fec_weball(p, int(y)))
    if not frames:
        return races[KEYS].copy()
    fec = pd.concat(frames, ignore_index=True)
    fec = fec[fec["party"].isin(["D", "R"])].copy()
    fec["last"] = fec["name"].map(last_name)
    for c in ("receipts", "self_funding", "cash_on_hand"):
        fec[c] = pd.to_numeric(fec[c], errors="coerce").fillna(0)
    fec["net"] = (fec["receipts"] - fec["self_funding"]).clip(lower=0)
    grp = {k: g for k, g in fec.groupby(["year", "chamber", "state", "district", "party"])}

    rows = []
    for _, r in races.iterrows():
        rec = {k: r[k] for k in KEYS}
        got = {}
        for party, name in (("D", r.get("dem_name")), ("R", r.get("rep_name"))):
            g = grp.get((r["year"], r["chamber"], r["state"], r["district"], party))
            if g is None or g.empty:
                continue
            ln = last_name(name)
            hit = g[g["last"] == ln] if ln else g.iloc[0:0]
            pick = hit if len(hit) else g
            got[party] = pick.sort_values("receipts", ascending=False).iloc[0]
        if len(got) == 2:
            d, rr = got["D"], got["R"]
            tot = d["net"] + rr["net"]
            cash = d["cash_on_hand"] + rr["cash_on_hand"]
            rec.update(fec_fund_share=(d["net"] / tot - 0.5) if tot > 0 else np.nan,
                       fec_log_ratio=float(np.log((d["net"] + 5e4) / (rr["net"] + 5e4))),
                       fec_cash_share=(d["cash_on_hand"] / cash - 0.5) if cash > 0 else np.nan,
                       fec_self_fund_diff=(d["self_funding"] - rr["self_funding"])
                       / max(d["receipts"] + rr["receipts"], 1.0))
        rows.append(rec)
    return pd.DataFrame(rows)


def prev_overperf(history: pd.DataFrame, races: pd.DataFrame) -> pd.Series:
    """history: past races with margin, residual, dem_name, rep_name. races: rows to fill
    (needs incumbency_signed, dem_name, rep_name)."""
    h = history.copy()
    h["winner_last"] = np.where(h["margin"] > 0, h["dem_name"].map(last_name), h["rep_name"].map(last_name))
    by = {k: g.sort_values("year") for k, g in h.groupby(["chamber", "state"])}
    out = []
    for _, r in races.iterrows():
        inc = r.get("incumbency_signed")
        if not inc or pd.isna(inc) or inc == 0:
            out.append(np.nan)
            continue
        ln = last_name(r["dem_name"] if inc > 0 else r["rep_name"])
        g = by.get((r["chamber"], r["state"]))
        if not ln or g is None:
            out.append(np.nan)
            continue
        prior = g[(g["year"] < r["year"]) & (g["year"] >= r["year"] - 12) & (g["winner_last"] == ln)]
        if r["chamber"] == "House":
            prior = prior[prior["district"] == r["district"]] if (prior["district"] == r["district"]).any() \
                else prior
        out.append(float(prior.iloc[-1]["residual"]) if len(prior) else np.nan)
    return pd.Series(out, index=races.index)


def main():
    cfg = load_config("configs/baseline.yaml")
    feats = list(cfg["baseline"]["features"])
    inputs = load_baseline_inputs(cfg)
    national, incumbency = inputs
    hist = build_training_table(cfg, verbose=False, inputs=inputs).dropna(subset=feats)
    hist = baseline_residuals(hist, feats)                 # out-of-fold residuals, 2010-2024

    # 2026 races with candidate names from the incumbency file
    i26 = incumbency[(incumbency["year"] == 2026)
                     & ((incumbency["chamber"] == "Senate") | ~incumbency["special"])].copy()
    i26 = i26.rename(columns={"dem_candidate": "dem_name", "rep_candidate": "rep_name"})
    races = pd.concat([hist[KEYS + ["dem_name", "rep_name", "incumbency_signed"]],
                       i26[KEYS + ["dem_name", "rep_name", "incumbency_signed"]]], ignore_index=True)

    money = money_features(races)
    races["prev_overperf"] = prev_overperf(hist, races).to_numpy()
    out = races[KEYS + ["prev_overperf"]].merge(money, on=KEYS, how="left")
    out = standardize_keys(out)
    dest = repo_path("data/processed/candidate_features.csv")
    out.to_csv(dest, index=False)
    cov = out.groupby(["year", "chamber"])[["fec_fund_share", "prev_overperf"]].apply(
        lambda d: d.notna().mean().round(2))
    print(f"{len(out)} races -> {dest}\nshare of races with each feature:\n{cov.unstack().to_string()}")
    if "fec_fund_share" in hist.columns:
        pass
    m = out.merge(hist[KEYS + ["residual"]], on=KEYS)
    for c in ("fec_fund_share", "fec_log_ratio", "fec_cash_share", "prev_overperf"):
        ok = m[c].notna()
        print(f"corr({c}, baseline residual) = {np.corrcoef(m.loc[ok, c], m.loc[ok, 'residual'])[0, 1]:.3f} "
              f"(n={ok.sum()})")


if __name__ == "__main__":
    main()
