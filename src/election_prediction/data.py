"""
Raw-data loading, key standardization, and validation.

Every table that describes a race uses the same keys:

    year      int   2010, 2012, ..., 2026
    chamber   str   "House" or "Senate"
    state     str   two-letter postal code ("NY")
    district  int   House district number; 0 for at-large seats and all Senate races
    special   bool  True for special elections (lets two Senate races share a state-year)

race_id() turns these into one string, e.g. "2026-H-NY-17" or "2026-S-OH-00-SP".

The column contract for each raw file lives in SCHEMAS (and data/README.md).
validate() checks any table against it and raises one readable error listing
every problem it found.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

KEYS = ["year", "chamber", "state", "district", "special"]

STATE_CODES = {
    "ALABAMA": "AL", "ALASKA": "AK", "ARIZONA": "AZ", "ARKANSAS": "AR", "CALIFORNIA": "CA",
    "COLORADO": "CO", "CONNECTICUT": "CT", "DELAWARE": "DE", "DISTRICT OF COLUMBIA": "DC",
    "FLORIDA": "FL", "GEORGIA": "GA", "HAWAII": "HI", "IDAHO": "ID", "ILLINOIS": "IL",
    "INDIANA": "IN", "IOWA": "IA", "KANSAS": "KS", "KENTUCKY": "KY", "LOUISIANA": "LA",
    "MAINE": "ME", "MARYLAND": "MD", "MASSACHUSETTS": "MA", "MICHIGAN": "MI", "MINNESOTA": "MN",
    "MISSISSIPPI": "MS", "MISSOURI": "MO", "MONTANA": "MT", "NEBRASKA": "NE", "NEVADA": "NV",
    "NEW HAMPSHIRE": "NH", "NEW JERSEY": "NJ", "NEW MEXICO": "NM", "NEW YORK": "NY",
    "NORTH CAROLINA": "NC", "NORTH DAKOTA": "ND", "OHIO": "OH", "OKLAHOMA": "OK", "OREGON": "OR",
    "PENNSYLVANIA": "PA", "RHODE ISLAND": "RI", "SOUTH CAROLINA": "SC", "SOUTH DAKOTA": "SD",
    "TENNESSEE": "TN", "TEXAS": "TX", "UTAH": "UT", "VERMONT": "VT", "VIRGINIA": "VA",
    "WASHINGTON": "WA", "WEST VIRGINIA": "WV", "WISCONSIN": "WI", "WYOMING": "WY",
}
VALID_STATES = set(STATE_CODES.values())

AT_LARGE_STATES = {"AK", "DE", "ND", "SD", "VT", "WY"}   # one House seat, 2010-2026
AT_LARGE_BEFORE_2022 = {"MT"}                              # MT gained a 2nd seat in 2022

TOP_TWO_STATES = {"CA", "WA"}       # general election can be same-party
RANKED_CHOICE_STATES = {"ME", "AK"}  # first-round margin may not decide the winner


# ------------------------------------------------------------------ key cleanup
def standardize_state(s) -> pd.Series:
    """Accept 'NY', 'ny', 'New York', 'NEW YORK' -> 'NY'."""
    s = pd.Series(s, dtype="object").astype(str).str.strip().str.upper()
    return s.map(lambda x: x if x in VALID_STATES else STATE_CODES.get(x, np.nan))


def standardize_chamber(s) -> pd.Series:
    s = pd.Series(s, dtype="object").astype(str).str.strip().str.lower()
    mapping = {"house": "House", "us house": "House", "h": "House",
               "senate": "Senate", "us senate": "Senate", "s": "Senate"}
    return s.map(mapping)


def standardize_district(s) -> pd.Series:
    """'AL', 'at-large', '00', '0', 7.0, '07' -> int. Unparseable -> NaN."""
    def parse(x):
        if pd.isna(x):
            return np.nan
        t = str(x).strip().upper()
        if t in ("AL", "AT-LARGE", "AT LARGE", "ATLARGE"):
            return 0
        try:
            return int(float(t))
        except ValueError:
            return np.nan
    return pd.Series(s, dtype="object").map(parse).astype("Int64")


def standardize_keys(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean whichever key columns exist, in place on a copy:
      * state names/codes -> postal code
      * chamber spellings -> House/Senate
      * district -> int; Senate rows forced to 0
      * at-large seats -> district 0 (sources disagree on 0 vs 1), using the
        explicit lists above, so partial tables (one district of a state) are safe
      * special missing -> False
    """
    out = df.copy()
    if "state" in out:
        out["state"] = standardize_state(out["state"])
    if "chamber" in out:
        out["chamber"] = standardize_chamber(out["chamber"])
    if "district" in out:
        out["district"] = standardize_district(out["district"])
        if "chamber" in out:
            out.loc[out["chamber"] == "Senate", "district"] = 0
        if "state" in out:
            at_large = out["state"].isin(AT_LARGE_STATES)
            if "year" in out:
                yr = pd.to_numeric(out["year"], errors="coerce")
                at_large |= out["state"].isin(AT_LARGE_BEFORE_2022) & (yr < 2022)
            out.loc[at_large & out["district"].isin([0, 1]), "district"] = 0
    if "special" in out:
        out["special"] = out["special"].fillna(False).astype(str).str.upper().isin(
            ["TRUE", "1", "T", "YES", "Y"])
    elif {"year", "chamber", "state"} <= set(out.columns):
        out["special"] = False
    if "year" in out:
        out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    return out


def race_id(df: pd.DataFrame) -> pd.Series:
    """'2026-H-NY-17', '2026-S-OH-00-SP' (special)."""
    ch = df["chamber"].map({"House": "H", "Senate": "S"})
    base = (df["year"].astype(str) + "-" + ch + "-" + df["state"] + "-"
            + df["district"].astype(int).map("{:02d}".format))
    return base + np.where(df.get("special", False), "-SP", "")


def party_code(s) -> pd.Series:
    """Map messy party labels to D / R / O. DFL, Democratic-NPL count as D."""
    t = pd.Series(s, dtype="object").fillna("").astype(str).str.upper()
    return pd.Series(np.select(
        [t.str.contains("DEMOCRAT") | t.isin(["D", "DEM"]),
         t.str.contains("REPUBLICAN") | t.isin(["R", "REP", "GOP"])],
        ["D", "R"], default="O"), index=t.index)


# ------------------------------------------------------------------ schemas
@dataclass
class Schema:
    required: list[str]
    keys: list[str] = field(default_factory=list)          # must be unique together
    ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    allowed: dict[str, set] = field(default_factory=dict)
    no_missing: list[str] = field(default_factory=list)


SCHEMAS: dict[str, Schema] = {
    "races": Schema(
        required=KEYS + ["dem_votes", "rep_votes"],
        keys=KEYS,
        ranges={"dem_votes": (0, 1e9), "rep_votes": (0, 1e9), "year": (1976, 2026)},
        allowed={"chamber": {"House", "Senate"}},
        no_missing=["year", "chamber", "state", "district"],
    ),
    "national": Schema(
        required=["year", "generic_ballot", "net_approval", "pres_party"],
        keys=["year"],
        ranges={"generic_ballot": (-0.3, 0.3), "net_approval": (-0.6, 0.6)},
        allowed={"pres_party": {"D", "R"}},
        no_missing=["year", "generic_ballot", "net_approval", "pres_party"],
    ),
    "lean": Schema(
        required=KEYS + ["lean"],
        keys=KEYS,
        ranges={"lean": (-0.8, 0.8)},
        no_missing=["lean"],
    ),
    "candidates": Schema(
        required=KEYS + ["party", "name"],
        keys=KEYS + ["party"],
        allowed={"party": {"D", "R", "O"}},
        no_missing=["party", "name"],
    ),
    "polls": Schema(
        required=KEYS + ["end_date", "pollster", "sample_size", "dem_pct", "rep_pct"],
        ranges={"dem_pct": (0, 100), "rep_pct": (0, 100), "sample_size": (1, 1e6)},
    ),
    "ratings": Schema(
        required=KEYS + ["rating"],
        keys=KEYS,
        no_missing=["rating"],
    ),
}


class DataValidationError(ValueError):
    pass


def validate(df: pd.DataFrame, schema: str | Schema, name: str | None = None) -> pd.DataFrame:
    """
    Check a table against its schema. Collects ALL problems, then raises once.
    Returns df unchanged if it passes, so it can be chained.
    """
    sch = SCHEMAS[schema] if isinstance(schema, str) else schema
    label = name or (schema if isinstance(schema, str) else "table")
    problems: list[str] = []

    missing_cols = [c for c in sch.required if c not in df.columns]
    if missing_cols:
        problems.append(f"missing columns {missing_cols}")

    for c in sch.no_missing:
        if c in df.columns and df[c].isna().any():
            problems.append(f"{df[c].isna().sum()} blank values in '{c}'")

    if "state" in df.columns:
        bad = sorted(set(df["state"].dropna()) - VALID_STATES)
        if bad:
            problems.append(f"unknown state codes {bad[:10]}")

    for c, (lo, hi) in sch.ranges.items():
        if c in df.columns:
            v = pd.to_numeric(df[c], errors="coerce")
            out = v[(v < lo) | (v > hi)]
            if len(out):
                problems.append(f"{len(out)} values of '{c}' outside [{lo}, {hi}] "
                                f"(e.g. {out.iloc[0]}) -- check units (fractions vs points)")

    for c, ok in sch.allowed.items():
        if c in df.columns:
            bad = sorted(set(df[c].dropna()) - ok)
            if bad:
                problems.append(f"unexpected values in '{c}': {bad[:10]}")

    keys = [k for k in sch.keys if k in df.columns]
    if keys and len(keys) == len(sch.keys):
        dup = df.duplicated(keys, keep=False)
        if dup.any():
            ex = df.loc[dup, keys].head(3).to_dict("records")
            problems.append(f"{dup.sum()} rows share the same {keys} (e.g. {ex})")

    if problems:
        raise DataValidationError(f"[{label}] " + "; ".join(problems))
    return df


def coverage_report(df: pd.DataFrame, by: str = "year") -> pd.DataFrame:
    """Share of non-missing values per column per year. Spot gaps fast."""
    return df.groupby(by).agg(lambda s: s.notna().mean()).round(2)


# ------------------------------------------------------------------ loaders
def load_results(path, rename: dict | None = None) -> pd.DataFrame:
    """Generic loader for an already-tidy races table (one row per race)."""
    df = pd.read_csv(path)
    if rename:
        df = df.rename(columns=rename)
    return standardize_keys(df)


def load_table(path, schema: str, rename: dict | None = None, check: bool = True) -> pd.DataFrame:
    """Read a CSV, rename columns to the contract, standardize keys, validate."""
    df = pd.read_csv(path)
    if rename:
        df = df.rename(columns=rename)
    df = standardize_keys(df)
    return validate(df, schema, name=str(path)) if check else df


def load_mit_results(path, chamber: str, years: tuple[int, int] | None = None) -> pd.DataFrame:
    """
    Turn MIT Election Data + Science Lab candidate-level files (one row per
    candidate per ballot line) into one row per race with dem_votes / rep_votes.

    * keeps general elections only (stage == "GEN"), drops write-ins
    * fusion tickets (e.g. NY): the D candidate's votes are summed across all lines
    * adds flags: same_party (no D or no R in the general), top_two_state, rcv_state
    """
    raw = pd.read_csv(path, low_memory=False)
    raw.columns = [c.lower() for c in raw.columns]
    if "stage" in raw:
        raw = raw[raw["stage"].astype(str).str.upper().isin(["GEN", "GENERAL"])]
    if "writein" in raw:
        raw = raw[~raw["writein"].astype(str).str.upper().isin(["TRUE", "1"])]
    if years:
        raw = raw[raw["year"].between(*years)]

    party_col = "party_simplified" if "party_simplified" in raw else "party"
    raw = raw.assign(chamber=chamber, pcode=party_code(raw[party_col]).values)
    raw["state"] = raw["state_po"] if "state_po" in raw else raw["state"]
    if "district" not in raw or chamber == "Senate":
        raw["district"] = 0
    if "special" not in raw:
        raw["special"] = False
    raw = standardize_keys(raw)

    # Votes per candidate across all ballot lines, then each candidate's main party.
    cand = (raw.groupby(KEYS + ["candidate"], dropna=False)
               .agg(votes=("candidatevotes", "sum"),
                    pcode=("pcode", lambda p: "D" if (p == "D").any()
                           else ("R" if (p == "R").any() else "O")))
               .reset_index())
    # If a race somehow has two D (or two R) candidates, keep the larger vote-getter.
    top = (cand[cand["pcode"].isin(["D", "R"])]
           .sort_values("votes", ascending=False)
           .drop_duplicates(KEYS + ["pcode"]))
    wide = top.pivot_table(index=KEYS, columns="pcode", values="votes", aggfunc="first")
    names = top.pivot_table(index=KEYS, columns="pcode", values="candidate", aggfunc="first")
    races = raw.groupby(KEYS)["totalvotes"].max().to_frame("total_votes")
    races["dem_votes"] = wide.get("D")
    races["rep_votes"] = wide.get("R")
    races["dem_name"] = names.get("D")
    races["rep_name"] = names.get("R")
    races = races.reset_index()
    races["same_party"] = races["dem_votes"].isna() | races["rep_votes"].isna()
    races["top_two_state"] = races["state"].isin(TOP_TWO_STATES)
    races["rcv_state"] = races["state"].isin(RANKED_CHOICE_STATES)
    return races


# Header for the FEC bulk "All candidates" summary file (weball{yy}.txt, pipe-delimited,
# no header row). Check against the FEC's current data dictionary when you download it.
FEC_WEBALL_COLUMNS = [
    "CAND_ID", "CAND_NAME", "CAND_ICI", "PTY_CD", "CAND_PTY_AFFILIATION", "TTL_RECEIPTS",
    "TRANS_FROM_AUTH", "TTL_DISB", "TRANS_TO_AUTH", "COH_BOP", "COH_COP", "CAND_CONTRIB",
    "CAND_LOANS", "OTHER_LOANS", "CAND_LOAN_REPAY", "OTHER_LOAN_REPAY", "DEBTS_OWED_BY",
    "TTL_INDIV_CONTRIB", "CAND_OFFICE_ST", "CAND_OFFICE_DISTRICT", "SPEC_ELECTION",
    "PRIM_ELECTION", "RUN_ELECTION", "GEN_ELECTION", "GEN_ELECTION_PRECENT",
    "OTHER_POL_CMTE_CONTRIB", "POL_PTY_CONTRIB", "CVG_END_DT", "INDIV_REFUNDS", "CMTE_REFUNDS",
]


def load_fec_weball(path, year: int) -> pd.DataFrame:
    """
    Load one cycle's FEC candidate summary into tidy money columns per candidate.

    Timing caveat: these are cycle totals up to CVG_END_DT. For past cycles that
    can include post-election money; for a strict "as of Nov 1" view use the
    pre-general reports instead. Fine for v1, note it in the write-up.
    """
    df = pd.read_csv(path, sep="|", header=None, names=FEC_WEBALL_COLUMNS,
                     dtype={"CAND_OFFICE_DISTRICT": str}, low_memory=False)
    office = df["CAND_ID"].str[0].map({"H": "House", "S": "Senate"})
    out = pd.DataFrame({
        "year": year,
        "chamber": office,
        "state": df["CAND_OFFICE_ST"],
        "district": df["CAND_OFFICE_DISTRICT"],
        "special": False,
        "fec_candidate_id": df["CAND_ID"],
        "name": df["CAND_NAME"],
        "party": party_code(df["CAND_PTY_AFFILIATION"].map(
            {"DEM": "D", "REP": "R", "DFL": "D"}).fillna(df["CAND_PTY_AFFILIATION"])),
        "incumbent_status": df["CAND_ICI"].map({"I": "incumbent", "C": "challenger", "O": "open"}),
        "receipts": df["TTL_RECEIPTS"],
        "self_funding": df["CAND_CONTRIB"].fillna(0) + df["CAND_LOANS"].fillna(0),
        "indiv_contrib": df["TTL_INDIV_CONTRIB"],
        "pac_contrib": df["OTHER_POL_CMTE_CONTRIB"],
        "party_contrib": df["POL_PTY_CONTRIB"],
        "cash_on_hand": df["COH_COP"],
        "coverage_end": df["CVG_END_DT"],
    })
    out = out[out["chamber"].notna()]
    return standardize_keys(out)
