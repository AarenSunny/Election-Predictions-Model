"""
Feature-engineering helpers. Pure functions: DataFrame in, columns out.
Nothing here fits a model.

Candidate features are turned into ONE value per race, usually D minus R
or D's share of the two-candidate total, so every race is a single row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import KEYS


# ------------------------------------------------------------------ basics
def two_party_margin(dem_votes, rep_votes):
    """(D - R) / (D + R) in [-1, 1]. NaN if either side is missing or both are 0."""
    d = pd.to_numeric(pd.Series(dem_votes), errors="coerce")
    r = pd.to_numeric(pd.Series(rep_votes), errors="coerce")
    tot = d + r
    return ((d - r) / tot.where(tot > 0)).to_numpy()


def d_share(d, r):
    """D / (D + R). NaN when both are 0 or missing (e.g. no money raised)."""
    d = pd.to_numeric(pd.Series(d), errors="coerce").astype(float)
    r = pd.to_numeric(pd.Series(r), errors="coerce").astype(float)
    tot = d + r
    return (d / tot.where(tot > 0)).to_numpy()


def d_minus_r(d, r):
    return (pd.to_numeric(pd.Series(d), errors="coerce").astype(float)
            - pd.to_numeric(pd.Series(r), errors="coerce").astype(float)).to_numpy()


def signed_incumbency(incumbent_party):
    """'D' -> +1, 'R' -> -1, anything else (open seat, independent) -> 0."""
    s = pd.Series(incumbent_party).astype(str).str.upper().str.strip()
    return np.select([s == "D", s == "R"], [1, -1], default=0)


def approval_signed(net_approval, pres_party):
    """President's net approval in D terms (a popular R president is bad for Ds)."""
    na = pd.to_numeric(pd.Series(net_approval), errors="coerce").to_numpy()
    p = pd.Series(pres_party).astype(str).str.upper().to_numpy()
    return np.where(p == "D", na, -na)


def midterm_flag(year):
    return (pd.Series(year).astype(int) % 4 == 2).astype(int).to_numpy()


def pres_party_D(pres_party):
    """+1 if the president is a Democrat, -1 if a Republican (in D-candidate terms)."""
    p = pd.Series(pres_party).astype(str).str.upper().str.strip()
    return np.select([p == "D", p == "R"], [1, -1], default=0)


def midterm_pres_penalty(year, pres_party):
    """
    -(midterm_flag x pres_party_D), so positive = good for Ds like every other input:
    -1 in a midterm under a D president (bad for Ds),
    +1 in a midterm under an R president, 0 in presidential years. This is the
    "president's party loses in midterms" effect as one baseline input.
    """
    return -midterm_flag(year) * pres_party_D(pres_party)


# ------------------------------------------------------------------ baseline inputs (v1)
# The team's baseline spec. Linear regression can only ADD columns, never multiply them,
# so every "depends on which party holds the White House" effect is multiplied in here.
# Signs follow the target: positive = good for Democrats.

def approval_x_pres(approve_pct, pres_flag):
    """
    (approval - 50) x president flag (+1 D president, -1 R).
    Approval is in percent; 50% is neutral. A popular D president is positive (helps Ds),
    a popular R president negative. E.g. Bush 2008 at 25%: (25 - 50) x -1 = +25.
    """
    a = pd.to_numeric(pd.Series(approve_pct), errors="coerce").to_numpy(dtype=float)
    return (a - 50.0) * np.asarray(pres_flag, dtype=float)


def approval_x_pres_midterm(approve_pct, pres_flag, midterm):
    """approval_x_pres in midterm years, 0 otherwise ('approval counts extra in midterms')."""
    return approval_x_pres(approve_pct, pres_flag) * np.asarray(midterm, dtype=float)


def midterm_x_pres(midterm, pres_flag):
    """
    midterm flag x president flag: +1 midterm under a D president, -1 under an R
    president, 0 in presidential years. The regression learns the sign (expected
    negative: the president's party loses ground in midterms).
    """
    return np.asarray(midterm, dtype=float) * np.asarray(pres_flag, dtype=float)


BASELINE_INPUTS = ["lean", "generic_ballot", "incumbency_signed", "approval_x_pres",
                   "midterm_x_pres", "approval_x_pres_midterm"]


def add_baseline_inputs(df: pd.DataFrame, national: pd.DataFrame, incumbency: pd.DataFrame,
                        approval_source: str = "reliable_30d") -> pd.DataFrame:
    """
    Add every baseline input column to a race table (one row per race, KEYS + lean).

        national    inputs.build_national(...)   one row per year
        incumbency  inputs.load_incumbency(...)  one row per race
        approval_source  which approval series feeds approval_x_pres*, e.g.
                    'reliable_30d' (sheet: reliable | broad, window: latest | 30d | 90d)

    Adds: generic_ballot, midterm_flag, pres_flag, approval (the chosen series, %),
    approval_x_pres, approval_x_pres_midterm, midterm_x_pres, incumbency_signed.
    Races missing from the incumbency file get NaN (reported by the caller).
    """
    col = f"approval_{approval_source}"
    if col not in national:
        raise KeyError(f"unknown approval_source {approval_source!r}; have "
                       f"{sorted(c[len('approval_'):] for c in national if c.startswith('approval_'))}")
    nat = (national[["year", "generic_ballot", "midterm_flag", "pres_flag", col]]
           .rename(columns={col: "approval"}))
    out = df.drop(columns=[c for c in nat.columns if c != "year" and c in df], errors="ignore")
    out = out.merge(nat, on="year", how="left", validate="many_to_one")
    out["approval_x_pres"] = approval_x_pres(out["approval"], out["pres_flag"])
    out["approval_x_pres_midterm"] = approval_x_pres_midterm(out["approval"], out["pres_flag"],
                                                             out["midterm_flag"])
    out["midterm_x_pres"] = midterm_x_pres(out["midterm_flag"], out["pres_flag"])
    inc = incumbency[KEYS + ["incumbency_signed"]]
    out = out.drop(columns=["incumbency_signed"], errors="ignore").merge(
        inc, on=KEYS, how="left", validate="one_to_one")
    return out


def incumbency_x_lean(incumbent_party, lean):
    """
    signed incumbency x partisan lean (baseline input from the doc). Lets the
    baseline learn that incumbents in seats that lean their way run differently
    from incumbents holding seats that lean against them.
    """
    return signed_incumbency(incumbent_party) * pd.to_numeric(
        pd.Series(lean), errors="coerce").to_numpy()


def election_day(year: int) -> pd.Timestamp:
    """US general election: the Tuesday after the first Monday in November."""
    nov1 = pd.Timestamp(year=int(year), month=11, day=1)
    first_monday = nov1 + pd.Timedelta(days=(0 - nov1.weekday()) % 7)
    return first_monday + pd.Timedelta(days=1)


# ------------------------------------------------------------- candidates
def candidates_to_wide(cands: pd.DataFrame, value_cols: list[str],
                       keys: list[str] = KEYS) -> pd.DataFrame:
    """
    Long candidate table (one row per candidate, with a 'party' column of D/R)
    -> one row per race with D_<col> and R_<col> columns.
    Only the major-party candidates are kept; if a party has two (top-two
    states), the first row wins, so sort beforehand if it matters.
    """
    c = cands[cands["party"].isin(["D", "R"])].drop_duplicates(keys + ["party"])
    wide = c.pivot(index=keys, columns="party", values=value_cols)
    wide.columns = [f"{party}_{col}" for col, party in wide.columns]
    return wide.reset_index()


def moderation(cfscore, party, party_mean):
    """
    How much more moderate a candidate is than their party's average, with the
    sign fixed so POSITIVE ALWAYS MEANS MORE MODERATE for both parties.
    CFscores: liberal < 0 < conservative.
      D: cfscore - party_mean   (less negative than the average D = moderate)
      R: party_mean - cfscore   (less positive than the average R = moderate)
    """
    cf = pd.to_numeric(pd.Series(cfscore), errors="coerce").to_numpy()
    pm = pd.to_numeric(pd.Series(party_mean), errors="coerce").to_numpy()
    p = pd.Series(party).astype(str).str.upper().to_numpy()
    return np.where(p == "D", cf - pm, np.where(p == "R", pm - cf, np.nan))


def party_means(cands: pd.DataFrame, value_col="cfscore",
                by=("year", "chamber", "party")) -> pd.Series:
    """Party average of value_col within year+chamber, aligned to cands' rows."""
    return cands.groupby(list(by))[value_col].transform("mean")


def distance_advantage(d_position, r_position, district_position):
    """
    |R - district| - |D - district|: POSITIVE when the Democrat is closer to the
    district (good for D). All three must be on the same scale, e.g. CFscore
    units with the district placed via a mapping from partisan lean that you fit.
    """
    d = np.asarray(d_position, dtype=float)
    r = np.asarray(r_position, dtype=float)
    x = np.asarray(district_position, dtype=float)
    return np.abs(r - x) - np.abs(d - x)


def issue_gap_composite(cand_pos: np.ndarray, voter_pos: np.ndarray,
                        importance: np.ndarray, p: float = 2.0) -> np.ndarray:
    """
    Sum over issues of importance * |candidate - voter|^p for ONE candidate.
    Arrays are (n_races, n_issues); NaN issues are skipped. Compute it for D and R,
    then use R_gap - D_gap so positive favors D. (v2 feature; data-hungry.)
    """
    gap = importance * np.abs(cand_pos - voter_pos) ** p
    return np.nansum(gap, axis=1)


# ------------------------------------------------------------------ money
def money_features(wide: pd.DataFrame) -> pd.DataFrame:
    """
    From D_/R_ money columns (see data.load_fec_weball + candidates_to_wide) build:
      cash_share          D share of cash on hand ("cash in reserves")
      donor_mix_diff      (small-donor share - PAC share) for D, minus the same for R
                          (needs D_/R_small_donor_contrib = unitemized receipts < $200;
                          NaN if that column isn't present)
      fundraising_share   D share of receipts, excluding self-funding
      pac_tilt            (PAC share of receipts) D minus R
      self_fund_diff      log1p(self funding) D minus R
    The doc's v1 list uses cash_share and donor_mix_diff; the rest are available
    if you decide to test them. Missing inputs give NaN (LightGBM handles NaN).
    """
    out = pd.DataFrame(index=wide.index)
    d_raised = wide["D_receipts"] - wide["D_self_funding"].fillna(0)
    r_raised = wide["R_receipts"] - wide["R_self_funding"].fillna(0)
    out["fundraising_share"] = d_share(d_raised.clip(lower=0), r_raised.clip(lower=0))
    out["cash_share"] = d_share(wide["D_cash_on_hand"].clip(lower=0),
                                wide["R_cash_on_hand"].clip(lower=0))
    d_pac = wide["D_pac_contrib"] / wide["D_receipts"].where(wide["D_receipts"] > 0)
    r_pac = wide["R_pac_contrib"] / wide["R_receipts"].where(wide["R_receipts"] > 0)
    out["pac_tilt"] = d_pac - r_pac
    if {"D_small_donor_contrib", "R_small_donor_contrib"} <= set(wide.columns):
        d_small = wide["D_small_donor_contrib"] / wide["D_receipts"].where(wide["D_receipts"] > 0)
        r_small = wide["R_small_donor_contrib"] / wide["R_receipts"].where(wide["R_receipts"] > 0)
        out["donor_mix_diff"] = (d_small - d_pac) - (r_small - r_pac)
    else:
        out["donor_mix_diff"] = np.nan
    out["self_fund_diff"] = (np.log1p(wide["D_self_funding"].clip(lower=0))
                             - np.log1p(wide["R_self_funding"].clip(lower=0)))
    return out


# ------------------------------------------------------------------ changes
def change_over(df: pd.DataFrame, value_col: str, geo_cols: list[str],
                years_back: int, year_col: str = "year") -> pd.Series:
    """
    value(year) - value(year - years_back) for the same geography. NaN when the
    earlier year isn't present (e.g. after redistricting changed the unit).
    """
    prev = df[geo_cols + [year_col, value_col]].copy()
    prev[year_col] = prev[year_col] + years_back
    prev = prev.rename(columns={value_col: "_prev"})
    merged = df[geo_cols + [year_col, value_col]].merge(
        prev, on=geo_cols + [year_col], how="left")
    return pd.Series((merged[value_col] - merged["_prev"]).to_numpy(), index=df.index)


def add_horizon_changes(df: pd.DataFrame, value_col: str, geo_cols: list[str],
                        horizons: list[int], year_col: str = "year") -> pd.DataFrame:
    """
    Add one change column per horizon: '<value_col>_chg_<h>y' = current minus h
    years earlier, same geography (the doc's "time spans are change from current").
    Horizons per feature live in configs/baseline.yaml under feature_horizons.
    Missing earlier years give NaN (the doc's "Null").
    """
    out = df.copy()
    for h in horizons:
        out[f"{value_col}_chg_{h}y"] = change_over(out, value_col, geo_cols, h, year_col)
    return out


# ------------------------------------------------------------------ missing data
def add_missing_flags(df: pd.DataFrame, groups: dict[str, list[str]]) -> pd.DataFrame:
    """
    One 0/1 flag per GROUP of columns: 1 if every column in the group is missing.
    e.g. groups={"polls_missing": ["poll_gap", "poll_count"]}
    (Needed for neural nets; LightGBM can use NaN directly.)
    """
    out = df.copy()
    for flag, cols in groups.items():
        present = [c for c in cols if c in out]
        out[flag] = out[present].isna().all(axis=1).astype(int) if present else 1
    return out


# ------------------------------------------------------------------ crosswalks
def county_to_district(county_df: pd.DataFrame, crosswalk: pd.DataFrame, value_cols: list[str],
                       county_col="county_fips", weight_col="afact",
                       district_keys=("state", "district")) -> pd.DataFrame:
    """
    Population-weighted average of county values into districts.

    crosswalk: one row per (county, district) piece with weight_col = share of the
    DISTRICT's population living in that county (Geocorr's 'afact' when run
    district <- county). Also returns 'largest_county_share': close to 1 means the
    district sits inside one county, so county data can't tell it apart from
    its neighbors (weak estimate).
    """
    keys = list(district_keys)
    m = crosswalk.merge(county_df[[county_col] + value_cols], on=county_col, how="left")
    rows = []
    for k, g in m.groupby(keys):
        rec = dict(zip(keys, k if isinstance(k, tuple) else (k,), strict=False))
        for c in value_cols:
            ok = g[c].notna()
            w = g.loc[ok, weight_col]
            rec[c] = float(np.average(g.loc[ok, c], weights=w)) if w.sum() > 0 else np.nan
        rec["largest_county_share"] = float(g[weight_col].max())
        rows.append(rec)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ polls
POLL_COLUMNS = ["poll_margin", "poll_count_30d", "days_since_poll", "n_polls_used"]


def poll_average(polls: pd.DataFrame, as_of, half_life_days: float = 12.0,
                 max_age_days: int = 60, internal_discount: float = 0.5,
                 keys: list[str] = KEYS) -> pd.DataFrame:
    """
    Weighted two-party poll margin per race, as in the doc:

        m_i = (D_i - R_i) / (D_i + R_i)          poll i's two-party margin
        w_i = 0.5^(age_i / H) * sqrt(n_i) * rating_i * internal_i
        poll_margin = sum(w_i * m_i) / sum(w_i)

      age_i       days from the poll's field end date to the cutoff (as_of)
      H           half-life (default 12 days)
      n_i         sample size
      rating_i    'pollster_rating' column if present (weight, e.g. 0-1), else 1
      internal_i  internal_discount (0.5) when the 'internal' column is True
                  (internal or sponsor-released poll), else 1

    Only polls that ended on or before as_of and within max_age_days of it count
    (no peeking). Use poll_cutoff(year, days_before) for as_of so every year uses
    the same rule. Returns keys + poll_margin, poll_count_30d, days_since_poll,
    n_polls_used. Races with no polls are simply absent: after merging, fill
    poll_count_30d with 0 and leave the rest NaN (the doc's "Null").
    """
    p = polls.copy()
    p["end_date"] = pd.to_datetime(p["end_date"])
    as_of = pd.Timestamp(as_of)
    age = (as_of - p["end_date"]).dt.days
    p = p[(age >= 0) & (age <= max_age_days)].copy()
    if p.empty:
        return pd.DataFrame(columns=keys + POLL_COLUMNS)
    p["age"] = (as_of - p["end_date"]).dt.days
    rating = p["pollster_rating"].fillna(1.0) if "pollster_rating" in p else 1.0
    internal = (p["internal"].fillna(False).astype(bool)
                if "internal" in p else pd.Series(False, index=p.index))
    p["w"] = (0.5 ** (p["age"] / half_life_days) * np.sqrt(p["sample_size"].clip(lower=1))
              * rating * np.where(internal, internal_discount, 1.0))
    two_party = (p["dem_pct"] + p["rep_pct"]).where(lambda s: s > 0)
    p["wm"] = p["w"] * (p["dem_pct"] - p["rep_pct"]) / two_party
    p["recent"] = (p["age"] <= 30).astype(int)
    g = p.groupby(keys)
    return pd.DataFrame({
        "poll_margin": g["wm"].sum() / g["w"].sum(),
        "poll_count_30d": g["recent"].sum(),
        "days_since_poll": g["age"].min(),
        "n_polls_used": g.size(),
    }).reset_index()


def poll_cutoff(year: int, days_before_election: int = 5) -> pd.Timestamp:
    """The doc's cutoff: a fixed number of days before election day, same rule every year."""
    return election_day(year) - pd.Timedelta(days=days_before_election)


def poll_gap(poll_margin, baseline_margin):
    """Poll margin vs. baseline (D minus R): NaN when unpolled. Both in two-party units."""
    return (pd.to_numeric(pd.Series(poll_margin), errors="coerce").to_numpy()
            - pd.to_numeric(pd.Series(baseline_margin), errors="coerce").to_numpy())
