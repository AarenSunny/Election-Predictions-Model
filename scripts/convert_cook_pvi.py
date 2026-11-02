"""
Turn the Cook PVI workbook (Cook_PVI_1997-2026.xlsx) into House rows of data/raw/lean.csv.

Each sheet is one set of district lines. A race in year Y uses the sheet whose lines
match Y's districts AND whose presidential inputs were all known before Y's election
(no same-day results):

    race  sheet                            lines            presidential inputs
    1998  106th (99-00)                    1998 maps        1992, 1996
    2000  106th (99-00)                    1998 maps        1992, 1996
    2002  108th (03-04)                    2002 maps        1996, 2000
    2004  108th (03-04)                    2002 maps        1996, 2000
    2006  110th (07-08)                    2006 maps        2000, 2004
    2008  110th (07-08)                    2006 maps        2000, 2004
    2010  111th (09-10)                    2008 maps        2004, 2008
    2012  Post-2012 redistricting (2012)   2012 maps        2004, 2008
    2014  113th (13-14)                    2012 maps        2008, 2012
    2016  114th (15-16)                    2012 maps        2008, 2012
    2018  115th (17-18)                    2016 maps        2012, 2016
    2020  116th (19-20)                    2018 maps        2012, 2016
    2022  118th (23-24)                    2022 maps        2016, 2020
    2024  118th (23-24)                    2022 maps        2016, 2020
    2026  2026 (post-redistricting)        2026 maps        2020, 2024

(Each odd-year sheet is the Congress elected the November before, scored with the two
presidential elections before that: e.g. 106th = elected 1998, 1992 + 1996 results. 1998 uses
106th rather than 105th because a few states (NC, VA, NY...) redrew between 1996 and 1998
and the inputs are the same.)

States whose lines changed mid-decade have no clean match (the right-lines sheet uses
that year's own presidential result), so those state-years are left out:
    2000 NC   2004 TX   2016 FL, VA, NC   2020 NC   2024 AL, GA, LA, NC, NY
Exception: PA 2018 regular races use the 116th sheet (new court-drawn PA map, same
2012/2016 inputs); PA's 2018 specials (old PA-7, PA-15) ran on the old map, so they keep 115th.
Specials in a redistricting year usually run on the OLD lines, so they use the previous
Congress's sheet (SPECIAL_SHEET): 2002 -> 107th, 2012 -> 112th, 2022 -> 117th. 2006 specials
(e.g. TX-22 on the court-redrawn map) use 110th like the regular races.

Scale: Cook PVI is in vote-share points; lean (margin scale) = PVI x 2 / 100, D positive.
The "Raw" (unrounded) column is used where the sheet has one.

Senate (--senate-xlsx COOK_SENATE_PVI.xlsx): Cook state PVIs published in 1997, 2001,
2005, ..., 2025 (one block of 50 states + DC per year, the year after each presidential
election). A Senate race in year Y uses the latest block published BEFORE Y:
    1997 -> 1998, 2000    2001 -> 2002, 2004    2005 -> 2006, 2008    2009 -> 2010, 2012
    2013 -> 2014, 2016    2017 -> 2018, 2020    2021 -> 2022, 2024    2025 -> 2026
State lines never change, so there are no redistricting gaps. D.C. is dropped (no
Senate seats); state names are cleaned ('Nebraska †' -> NE). The raw (unrounded)
column is used.

Usage:
    python scripts/convert_cook_pvi.py --xlsx data/raw/cook/Cook_PVI_1997-2026.xlsx \
        --senate-xlsx data/raw/cook/COOK_SENATE_PVI.xlsx --out data/raw/lean.csv
(Without --senate-xlsx, --senate-lean <csv> keeps the Senate rows of an existing lean.csv,
e.g. the presidential-results version from scripts/build_lean.py.)
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

SHEET_FOR_YEAR = {
    1998: "106th (99-00)",
    2000: "106th (99-00)",
    2002: "108th (03-04)",
    2004: "108th (03-04)",
    2006: "110th (07-08)",
    2008: "110th (07-08)",
    2010: "111th (09-10)",
    2012: "Post-2012 redistricting (2012)",
    2014: "113th (13-14)",
    2016: "114th (15-16)",
    2018: "115th (17-18)",
    2020: "116th (19-20)",
    2022: "118th (23-24)",
    2024: "118th (23-24)",
    2026: "2026 (post-redistricting)",
}
OVERRIDES = {(2018, "PA"): "116th (19-20)"}
SPECIAL_SHEET = {2002: "107th (01-02)", 2012: "112th (11-12)", 2022: "117th (21-22)"}
SKIP = {2000: {"NC"}, 2004: {"TX"}, 2016: {"FL", "VA", "NC"}, 2020: {"NC"},
        2024: {"AL", "GA", "LA", "NC", "NY"}}

STATE_CODES = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
    "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "Florida": "FL", "Georgia": "GA",
    "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA",
    "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH",
    "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY", "North Carolina": "NC",
    "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA",
    "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN",
    "Texas": "TX", "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
}


def parse_pvi(v) -> float:
    """'R+12.07' -> -12.07, 'D+5' -> 5.0, 'EVEN' -> 0.0 (vote-share points, D positive)."""
    s = str(v).strip().upper().replace(" ", "")
    if s in ("EVEN", "E"):
        return 0.0
    m = re.fullmatch(r"([DR])\+([\d.]+)", s)
    if not m:
        raise ValueError(f"can't read PVI value {v!r}")
    return (1 if m.group(1) == "D" else -1) * float(m.group(2))


def parse_district(v) -> int:
    s = str(v).strip().upper()
    return 0 if s in ("AL", "AT-LARGE", "0", "00") else int(float(s))


def read_sheet(xlsx: pd.ExcelFile, sheet: str) -> pd.DataFrame:
    """Sheet -> state, district, pvi (vote-share points). Handles the two layouts."""
    d = pd.read_excel(xlsx, sheet, dtype=str)
    cols = list(d.columns)
    raw = [c for c in cols if "Raw" in c]
    pvi_col = raw[0] if raw else next(c for c in cols if "PVI" in c)
    if "Dist" in cols:                                  # 2026 layout: "AL-01", "WY-AL"
        parts = d["Dist"].str.split("-", n=1, expand=True)
        state, dist = parts[0], parts[1]
    else:                                               # older layout: State name + number
        state = d["State"].map(STATE_CODES)
        dist = d[cols[1]]
    out = pd.DataFrame({"state": state, "district": dist.map(parse_district),
                        "pvi": d[pvi_col].map(parse_pvi)})
    if out["state"].isna().any():
        raise ValueError(f"{sheet}: unknown state names {d.loc[out['state'].isna(), cols[0]].unique()}")
    # single-district states: always district 0
    n = out.groupby("state")["district"].transform("count")
    out.loc[n == 1, "district"] = 0
    return out


def build_house_lean(xlsx_path: Path) -> pd.DataFrame:
    xlsx = pd.ExcelFile(xlsx_path)
    cache: dict[str, pd.DataFrame] = {}

    def sheet(name: str) -> pd.DataFrame:
        if name not in cache:
            cache[name] = read_sheet(xlsx, name)
        return cache[name]

    rows = []
    for year, base in SHEET_FOR_YEAR.items():
        regular_rows = sheet(base).copy()
        special_rows = sheet(SPECIAL_SHEET.get(year, base)).copy()   # overrides don't apply
        for (y, st), alt in OVERRIDES.items():
            if y == year:
                regular_rows = pd.concat([regular_rows[regular_rows["state"] != st],
                                          sheet(alt)[sheet(alt)["state"] == st]])
        for d, is_special in ((regular_rows, False), (special_rows, True)):
            d = d[~d["state"].isin(SKIP.get(year, set()))].copy()
            d["year"], d["special"] = year, is_special
            rows.append(d)
    h = pd.concat(rows, ignore_index=True)
    h["chamber"] = "House"
    h["lean"] = 2 * h["pvi"] / 100
    return h[["year", "chamber", "state", "district", "special", "lean"]]


def build_senate_lean(xlsx_path: Path, race_years=range(1998, 2027, 2)) -> pd.DataFrame:
    """Cook state PVI blocks -> Senate rows of lean.csv (see module docstring for the year rule)."""
    raw = pd.read_excel(xlsx_path, header=None, dtype=str)
    if not re.fullmatch(r"\d{4}(\.0)?", str(raw.iloc[0, 0]).strip()):   # skip a header row if present
        raw = raw.iloc[1:]
    d = pd.DataFrame({"cook_year": raw.iloc[:, 0].str.strip().str[:4].astype(int),
                      "name": raw.iloc[:, 1].str.replace(r"[^A-Za-z .]", "", regex=True).str.strip(),
                      "pvi": raw.iloc[:, 3].map(parse_pvi)})
    d = d[~d["name"].str.upper().isin(["DC", "D.C.", "DISTRICT OF COLUMBIA"])]
    d["state"] = d["name"].map(STATE_CODES)
    if d["state"].isna().any():
        raise ValueError(f"Senate PVI: unknown state names {d.loc[d['state'].isna(), 'name'].unique()}")
    n = d.groupby("cook_year").size()
    if (n != 50).any():
        raise ValueError(f"Senate PVI: expected 50 states per year, got {n.to_dict()}")
    rows = []
    for y in race_years:
        earlier = [c for c in sorted(d["cook_year"].unique()) if c < y]
        if not earlier:
            continue
        blk = d[d["cook_year"] == earlier[-1]]
        rows.append(pd.DataFrame({"year": y, "chamber": "Senate", "state": blk["state"].values,
                                  "district": 0, "lean": 2 * blk["pvi"].values / 100,
                                  "cook_year": earlier[-1]}))
    s = pd.concat(rows, ignore_index=True)
    s = pd.concat([s.assign(special=False), s.assign(special=True)], ignore_index=True)
    return s[["year", "chamber", "state", "district", "special", "lean"]]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True, type=Path)
    ap.add_argument("--senate-xlsx", type=Path, default=None,
                    help="Cook Senate (state) PVI workbook -> Senate rows")
    ap.add_argument("--senate-lean", type=Path, default=None,
                    help="existing lean.csv whose Senate rows are kept (if no --senate-xlsx)")
    ap.add_argument("--out", type=Path, default=Path("data/raw/lean.csv"))
    a = ap.parse_args(argv)

    house = build_house_lean(a.xlsx)
    parts = [house]
    if a.senate_xlsx:
        parts.append(build_senate_lean(a.senate_xlsx))
    elif a.senate_lean and a.senate_lean.exists():
        old = pd.read_csv(a.senate_lean)
        parts.append(old[old["chamber"] == "Senate"])
    lean = pd.concat(parts, ignore_index=True)
    keys = ["year", "chamber", "state", "district", "special"]
    if lean.duplicated(keys).any():
        raise ValueError(f"duplicate races: {lean[lean.duplicated(keys, keep=False)].head()}")
    lean.sort_values(keys).to_csv(a.out, index=False)
    print(lean[~lean["special"]].groupby(["year", "chamber"]).size().unstack().to_string())
    for y, sts in SKIP.items():
        print(f"left out (lines changed, no clean PVI): {y} {sorted(sts)}")


if __name__ == "__main__":
    main()
