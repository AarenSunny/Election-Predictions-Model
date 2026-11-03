"""
ACS 5-year files (from "ACS data get.py", folder acs_out/) -> residual-model features,
one row per race: data/processed/residual_features.csv.

    python scripts/build_acs_features.py --acs ~/Downloads/Residual/acs_out \
        --out data/processed/residual_features.csv

Fixes / rules applied (see the data check of 2026-11-02):
  * industry shares recomputed from the raw C24050 columns 002-014 (the pulled ind_*
    columns double-count occupation totals: construction and 'reference' are wrong)
  * income and rent: log value minus that election year's mean across the same
    geography level (House districts / states), so inflation drops out
  * House rows on the wrong district lines are blanked, never guessed:
      2022 (ACS 2016-20 is on the old 116th lines; 2022 used new maps), and in 2026
      the states that redrew (CA TX FL OH TN LA MO AL UT NC).
    (2016 FL/VA/NC, 2018 PA, 2020 NC, 2024 AL/GA/LA/NC/NY have no lean, so those races
    are not in training anyway.)
  * Senate rows = state; DC / Puerto Rico dropped; specials get the same row
  * Elections 2010-2014 have no file -> not in the output (features blank in training)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FIPS = {"01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO", "09": "CT", "10": "DE",
        "12": "FL", "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN", "19": "IA", "20": "KS",
        "21": "KY", "22": "LA", "23": "ME", "24": "MD", "25": "MA", "26": "MI", "27": "MN", "28": "MS",
        "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH", "34": "NJ", "35": "NM", "36": "NY",
        "37": "NC", "38": "ND", "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI", "45": "SC",
        "46": "SD", "47": "TN", "48": "TX", "49": "UT", "50": "VT", "51": "VA", "53": "WA", "54": "WV",
        "55": "WI", "56": "WY"}
AT_LARGE = {"AK", "DE", "ND", "SD", "VT", "WY"}
WRONG_LINES = {2022: "ALL", 2026: {"CA", "TX", "FL", "OH", "TN", "LA", "MO", "AL", "UT", "NC"}}

IND = {"ind_agri_mining": [2], "ind_construction": [3], "ind_manufacturing": [4],
       "ind_prof_finance": [9, 10], "ind_educ_health": [11], "ind_public_admin": [14],
       "ind_other": [5, 6, 7, 8, 12, 13]}
FEATURES = ["acs_rel_log_income", "acs_rel_log_rent", "acs_median_age", "acs_pct_poverty",
            "acs_pct_ba_plus", "acs_pct_less_hs", "acs_eth_nh_black", "acs_eth_hispanic",
            "acs_eth_nh_asian", "acs_eth_other"] + [f"acs_{k}" for k in IND if k != "ind_other"]


def _features(d: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=d.index)
    inc = pd.to_numeric(d["median_hh_income_nominal"], errors="coerce")
    rent = pd.to_numeric(d["median_gross_rent"], errors="coerce")
    out["acs_rel_log_income"] = np.log(inc) - np.log(inc).mean()
    out["acs_rel_log_rent"] = np.log(rent) - np.log(rent).mean()
    out["acs_median_age"] = pd.to_numeric(d["median_age"], errors="coerce")
    for c in ("pct_poverty", "pct_ba_plus", "pct_less_hs", "eth_nh_black", "eth_hispanic",
              "eth_nh_asian", "eth_other"):
        out[f"acs_{c}"] = pd.to_numeric(d[c], errors="coerce")
    table = "C24050" if "C24050_001E" in d else "C24030"
    if table == "C24050":
        tot = pd.to_numeric(d["C24050_001E"], errors="coerce")
        for k, rows in IND.items():
            if k == "ind_other":
                continue
            out[f"acs_{k}"] = sum(pd.to_numeric(d[f"C24050_{r:03d}E"], errors="coerce")
                                  for r in rows) / tot
    else:  # older table: the pulled shares were correct (they sum to 1)
        for k in IND:
            if k != "ind_other":
                out[f"acs_{k}"] = pd.to_numeric(d[k], errors="coerce")
    return out


def build(acs_dir: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(acs_dir.glob("acs_*_house.csv")):
        d = pd.read_csv(f, dtype={"state": str, "congressional district": str}, low_memory=False)
        year = int(d["election_year"].iloc[0])
        d["st"] = d["state"].str.zfill(2).map(FIPS)
        d = d[d["st"].notna()].copy()
        dist = pd.to_numeric(d["congressional district"], errors="coerce").fillna(0).astype(int)
        d["district"] = np.where(d["st"].isin(AT_LARGE) | (dist == 0), 0, dist)
        if year < 2022:   # MT had one seat until 2022
            d.loc[d["st"] == "MT", "district"] = 0
        feats = _features(d)
        bad = WRONG_LINES.get(year)
        if bad == "ALL":
            feats[:] = np.nan
        elif bad:
            feats.loc[d["st"].isin(bad).to_numpy(), :] = np.nan
        rows.append(pd.concat([pd.DataFrame({"year": year, "chamber": "House", "state": d["st"],
                                             "district": d["district"], "special": False}),
                               feats], axis=1))
    for f in sorted(acs_dir.glob("acs_*_senate.csv")):
        if "baseline" in f.name:
            continue
        d = pd.read_csv(f, dtype={"state": str}, low_memory=False)
        year = int(d["election_year"].iloc[0])
        d["st"] = d["state"].str.zfill(2).map(FIPS)
        d = d[d["st"].notna()].copy()
        feats = _features(d)
        base = pd.concat([pd.DataFrame({"year": year, "chamber": "Senate", "state": d["st"],
                                        "district": 0}), feats], axis=1)
        rows.append(base.assign(special=False))
        rows.append(base.assign(special=True))
    out = pd.concat(rows, ignore_index=True)
    out = out[["year", "chamber", "state", "district", "special"] + FEATURES]
    dup = out.duplicated(["year", "chamber", "state", "district", "special"])
    if dup.any():
        raise ValueError(f"duplicate races: {out[dup].head()}")
    return out.sort_values(["year", "chamber", "state", "district", "special"]).reset_index(drop=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--acs", required=True, type=Path)
    ap.add_argument("--out", default=Path("data/processed/residual_features.csv"), type=Path)
    a = ap.parse_args(argv)
    out = build(a.acs)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    filled = out.groupby(["year", "chamber"])["acs_pct_ba_plus"].apply(lambda s: s.notna().sum())
    print(f"{len(out)} rows -> {a.out}\nraces with ACS data:\n{filled.unstack().to_string()}")
    ind = out[[c for c in out if c.startswith("acs_ind_")]].sum(axis=1)
    print(f"industry shares (6 of 7 groups) sum: median {ind.median():.3f}, "
          f"max {ind.max():.3f} (must be < 1)")


if __name__ == "__main__":
    main()
