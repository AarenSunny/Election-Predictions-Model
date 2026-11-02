"""
Build data/raw/lean.csv: partisan lean for every race, in D terms, on the same
MARGIN scale as the model's target ((D - R) / (D + R)):

    lean = average over the 2 presidential elections BEFORE the race year of
           (area's D two-party margin - national D two-party margin)

Scale note: Cook PVI is in vote-SHARE points, half of margin points. Cook D+8
corresponds to lean = +0.16 here. If you use Cook PVI numbers, multiply by 2 / 100.

"Before" is strict: a 2016 race uses 2008 + 2012, never 2016 itself (that result
wasn't known before the election).

Senate: area = state, from 538's presidential results (fivethirtyeight/election-results),
        which start in 2000. For 1998-2002 races pass --pres-mit with MIT Election Lab's
        1976-2020-president.csv (columns year, state_po, party, candidatevotes); only its
        pre-2000 years are used. Checked: national two-party margins 1992 +6.96, 1996 +9.47
        match the official totals, and 2000-2020 state margins match 538's.
House:  area = district on THAT race year's lines. 538 has no district-level
        presidential results, so pass --house-pres with a file built from The
        Downballot's sheets, columns:
            year, state, district, pres_year, dem_votes, rep_votes
        where `year` is the race year whose district lines the numbers use.

Usage:
    python scripts/build_lean.py --e538 <election-results dir> --out data/raw/lean.csv \
        [--pres-mit data/raw/mit/1976-2020-president.csv] \
        [--house-pres data/raw/downballot/house_pres_by_district.csv]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

RACE_YEARS = list(range(1998, 2027, 2))
STATES_50_DC = {"AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA", "HI", "ID", "IL",
                "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE",
                "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD",
                "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"}
PRES_YEARS = [1992, 1996, 2000, 2004, 2008, 2012, 2016, 2020, 2024]


def prior_pres_years(race_year: int, n: int = 2) -> list[int]:
    """The n presidential elections strictly before race_year."""
    return [y for y in PRES_YEARS if y < race_year][-n:]


def _votes_538(e538: Path) -> pd.DataFrame:
    p = pd.read_csv(e538 / "election_results_presidential.csv", low_memory=False)
    # 538 codes Maine/Nebraska district results as M1, M2, N1... : keep real states + DC only
    p = p[(p["stage"] == "general") & p["state_abbrev"].isin(STATES_50_DC)]
    p["party"] = p["ballot_party"].map({"DEM": "D", "DFL": "D", "REP": "R"})
    # fusion (NY): a candidate's votes on minor-party lines count for their party
    p["party"] = p.groupby(["cycle", "state_abbrev", "candidate_id"])["party"].transform("first")
    v = (p.dropna(subset=["party"])
          .groupby(["cycle", "state_abbrev", "party"])["votes"].sum()
          .unstack("party").reset_index())
    v.columns.name = None
    return v


def _votes_mit(path: Path) -> pd.DataFrame:
    m = pd.read_csv(path)
    m["party"] = m["party"].astype(str).str.upper().map({"DEMOCRAT": "D", "REPUBLICAN": "R"})
    m = m[m["state_po"].isin(STATES_50_DC)].dropna(subset=["party"])
    v = (m.groupby(["year", "state_po", "party"])["candidatevotes"].sum()
          .unstack("party").reset_index()
          .rename(columns={"year": "cycle", "state_po": "state_abbrev"}))
    v.columns.name = None
    return v


def state_pres_margins(e538: Path, pres_mit: Path | None = None) -> pd.DataFrame:
    """D two-party margin per state per presidential year, plus the national margin."""
    v = _votes_538(e538)
    if pres_mit is not None:
        old = _votes_mit(pres_mit)
        v = pd.concat([old[old["cycle"] < v["cycle"].min()], v], ignore_index=True)
    nat = v.groupby("cycle")[["D", "R"]].sum()            # national = sum of states + DC
    v["margin"] = (v["D"] - v["R"]) / (v["D"] + v["R"])
    v["nat_margin"] = v["cycle"].map((nat["D"] - nat["R"]) / (nat["D"] + nat["R"]))
    v["rel"] = v["margin"] - v["nat_margin"]
    return v.rename(columns={"cycle": "pres_year", "state_abbrev": "state"})


def senate_lean(margins: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for y in RACE_YEARS:
        yrs = prior_pres_years(y)
        m = margins[margins["pres_year"].isin(yrs)].groupby("state")
        lean = m["rel"].mean()[m.size() == len(yrs)]      # need both elections
        rows.append(pd.DataFrame({"year": y, "chamber": "Senate", "state": lean.index,
                                  "district": 0, "lean": lean.values}))
    out = pd.concat(rows, ignore_index=True)
    out = out[out["state"] != "DC"]
    # a state can have a regular and a special Senate race in the same year: same lean
    both = out.assign(special=True)
    return pd.concat([out.assign(special=False), both], ignore_index=True)


def house_lean(house_pres: pd.DataFrame, margins: pd.DataFrame) -> pd.DataFrame:
    h = house_pres.copy()
    h["margin"] = (h["dem_votes"] - h["rep_votes"]) / (h["dem_votes"] + h["rep_votes"])
    nat = margins.groupby("pres_year")["nat_margin"].first()
    h["rel"] = h["margin"] - h["pres_year"].map(nat)
    rows = []
    for y, g in h.groupby("year"):
        yrs = prior_pres_years(int(y))
        g = g[g["pres_year"].isin(yrs)]
        m = g.groupby(["state", "district"])
        lean = m["rel"].mean()[m.size() == len(yrs)]
        rows.append(lean.reset_index().assign(year=int(y)))
    out = pd.concat(rows, ignore_index=True).rename(columns={"rel": "lean"})
    out["chamber"] = "House"
    return pd.concat([out.assign(special=False), out.assign(special=True)], ignore_index=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--e538", required=True, type=Path)
    ap.add_argument("--pres-mit", type=Path, default=None,
                    help="MIT 1976-2020-president.csv, for presidential years before 2000")
    ap.add_argument("--house-pres", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("data/raw/lean.csv"))
    a = ap.parse_args(argv)

    margins = state_pres_margins(a.e538, a.pres_mit)
    parts = [senate_lean(margins)]
    if a.house_pres:
        parts.append(house_lean(pd.read_csv(a.house_pres), margins))
    else:
        print("no --house-pres file: building Senate lean only")
    lean = pd.concat(parts, ignore_index=True)
    cols = ["year", "chamber", "state", "district", "special", "lean"]
    lean[cols].sort_values(cols[:5]).to_csv(a.out, index=False)
    print(lean.groupby(["year", "chamber"]).size().unstack().to_string())


if __name__ == "__main__":
    main()
