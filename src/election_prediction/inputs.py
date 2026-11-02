"""
Loaders for the hand-built baseline input files (the team's spreadsheets).

Every file has its own layout and its own quirks, so each gets its own loader that
returns a tidy table with clean integer years. Nothing here models anything; it
only reads, cleans and checks.

    file (configs/baseline.yaml -> data:)   loader                   returns (one row per ...)
    generic_ballot   Generic_Ballot_Data      load_generic_ballot      year
    midterm_flag     MidtermFlag              load_flag                year
    president_flag   President_Flag           load_flag                year
    approval         Presidental__Approval    load_approval            year x sheet x window
    incumbency       congressional_incumb.    load_incumbency          race (keys)

build_national() joins the year-level files into one table:

    year, generic_ballot, midterm_flag, pres_flag,
    approval_<sheet>_<window>       approve %   (sheet: reliable | broad, window: latest | 30d | 90d)
    disapproval_<sheet>_<window>    disapprove %

Scales (the model's target is the two-party margin, -1 = all R, +1 = all D):
    generic_ballot   (D - R) / (D + R) from the Dem / Rep poll numbers, so it is on
                     the same -1..1 scale as the target
    flags            midterm 1 / 0; president +1 D / -1 R
    approval         percent (0-100), kept raw; features.approval_x_pres applies (approve - 50)

Run `python -m election_prediction.inputs --config configs/baseline.yaml` to print
what was loaded and write the cleaned tables to data/processed/ for a look.
"""

from __future__ import annotations

import argparse
import re

import numpy as np
import pandas as pd

from .data import KEYS, standardize_keys

APPROVAL_SHEETS = {"broad": 0, "reliable": 1}       # sheet order in the workbook
APPROVAL_WINDOWS = ["latest", "30d", "90d"]
APPROVAL_SOURCES = [f"{s}_{w}" for s in APPROVAL_SHEETS for w in APPROVAL_WINDOWS]
# If the chosen window is blank for a year, try these next (same sheet). Every year
# is filled in the current files, so this only matters if a file changes.
APPROVAL_FALLBACK = {"latest": ["30d", "90d"], "30d": ["90d", "latest"], "90d": ["30d", "latest"]}


# ------------------------------------------------------------------ small parsers
def parse_year(v) -> float:
    """2026, 2026.0, '2026', ' 2026 ', '2026-11-03', Timestamp -> 2026. Anything else -> NaN."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    if isinstance(v, pd.Timestamp):
        return float(v.year)
    m = re.search(r"(19|20)\d{2}", str(v))
    return float(m.group(0)) if m else np.nan


def parse_margin(v) -> float:
    """'D+8.9' -> 8.9, 'R+1.7' -> -1.7, 'Tie' / 'EVEN' -> 0 (points, D positive)."""
    s = str(v).strip().upper().replace(" ", "")
    if s in ("TIE", "EVEN", "E", "0"):
        return 0.0
    m = re.fullmatch(r"([DR])\+([\d.]+)", s)
    if not m:
        return np.nan
    return (1 if m.group(1) == "D" else -1) * float(m.group(2))


def parse_pair(v) -> tuple[float, float]:
    """'66 / 30' -> (66.0, 30.0). Blank -> (NaN, NaN)."""
    if pd.isna(v):
        return np.nan, np.nan
    parts = re.findall(r"[\d.]+", str(v))
    if len(parts) != 2:
        return np.nan, np.nan
    return float(parts[0]), float(parts[1])


def as_percent(x) -> pd.Series:
    """Approval values sometimes arrive as fractions (0.29); put everything in percent."""
    x = pd.to_numeric(pd.Series(x), errors="coerce")
    return x.where(~(x <= 1.0), x * 100)


def _year_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        if str(c).strip().lower() == "year":
            return c
    raise KeyError(f"no 'Year' column in {list(df.columns)}")


def _check_years(df: pd.DataFrame, what: str) -> pd.DataFrame:
    bad = df["year"].isna()
    if bad.any():
        raise ValueError(f"{what}: {bad.sum()} rows with an unreadable year")
    df["year"] = df["year"].astype(int)
    dup = df["year"].duplicated()
    if dup.any():
        raise ValueError(f"{what}: duplicate years {sorted(df.loc[dup, 'year'])}")
    return df.sort_values("year").reset_index(drop=True)


# ------------------------------------------------------------------ year-level files
def load_generic_ballot(path) -> pd.DataFrame:
    """
    year, gb_dem, gb_rep, gb_margin_points, generic_ballot (two-party, -1..1), gb_imputed.

    Two-party margin = (Dem - Rep) / (Dem + Rep): drops undecided, like the race
    results drop third parties. A year with only a margin (2002: 'R+1.7') is divided
    by the average Dem + Rep share of the other years instead (flag gb_imputed).
    """
    d = pd.read_excel(path)
    yc = _year_col(d)
    cols = {str(c).strip().lower(): c for c in d.columns}
    out = pd.DataFrame({
        "year": d[yc].map(parse_year),
        "gb_dem": pd.to_numeric(d[cols["dem"]], errors="coerce"),
        "gb_rep": pd.to_numeric(d[cols["rep"]], errors="coerce"),
        "gb_margin_points": d[cols["margin"]].map(parse_margin),
    }).dropna(subset=["year"])
    out = _check_years(out, "generic ballot")
    have = out["gb_dem"].notna() & out["gb_rep"].notna()
    out.loc[have, "gb_margin_points"] = out["gb_dem"] - out["gb_rep"]
    major_share = (out.loc[have, "gb_dem"] + out.loc[have, "gb_rep"]).mean()
    out["generic_ballot"] = np.where(have, (out["gb_dem"] - out["gb_rep"]) / (out["gb_dem"] + out["gb_rep"]),
                                     out["gb_margin_points"] / major_share)
    out["gb_imputed"] = ~have
    if out["generic_ballot"].isna().any():
        missing = list(out.loc[out["generic_ballot"].isna(), "year"])
        raise ValueError(f"generic ballot: no usable value for {missing}")
    return out


def load_flag(path, name: str, allowed: set) -> pd.DataFrame:
    """Two-column sheet (Year, Number) -> year, <name>. Checks every value is allowed."""
    d = pd.read_excel(path)
    yc = _year_col(d)
    val = [c for c in d.columns if c != yc][0]
    out = pd.DataFrame({"year": d[yc].map(parse_year), name: pd.to_numeric(d[val], errors="coerce")})
    out = _check_years(out.dropna(subset=["year"]), name)
    bad = ~out[name].isin(allowed)
    if bad.any():
        raise ValueError(f"{name}: values {out.loc[bad, name].tolist()} not in {sorted(allowed)}")
    out[name] = out[name].astype(int)
    return out


def _approval_broad(d: pd.DataFrame) -> pd.DataFrame:
    """'Unreliable data but more' sheet: separate approve / disapprove columns."""
    def col(start, end):
        hits = [c for c in d.columns
                if str(c).lower().startswith(start) and str(c).lower().endswith(end)]
        if len(hits) != 1:
            raise KeyError(f"approval (broad sheet): expected one column {start}*{end}, got {hits}")
        return d[hits[0]]
    rows = []
    for window, (a, b, n) in {
        "latest": (col("last", "_approve"), col("last", "_disapprove"), None),
        "30d": (col("avg_all_polls_30d", "_approve"), col("avg_all_polls_30d", "_disapprove"),
                col("n_polls_30d", "30d")),
        "90d": (col("avg_all_polls_90d", "_approve"), col("avg_all_polls_90d", "_disapprove"),
                col("n_polls_90d", "90d")),
    }.items():
        rows.append(pd.DataFrame({"year": d[_year_col(d)].map(parse_year), "sheet": "broad",
                                  "window": window, "approve": as_percent(a),
                                  "disapprove": as_percent(b),
                                  "n_polls": pd.to_numeric(n, errors="coerce") if n is not None else np.nan}))
    return pd.concat(rows, ignore_index=True)


def _approval_reliable(d: pd.DataFrame) -> pd.DataFrame:
    """'Reliable' sheet: values like '66 / 30'; two columns named 'Records' (poll counts)."""
    cols = list(d.columns)
    yc = _year_col(d)

    def find(word):
        return next(c for c in cols if word in str(c).lower())
    latest, avg90, avg30 = find("latest"), find("90-day"), find("30-day")
    # the 'Records' column right after each average is its poll count
    n90, n30 = cols[cols.index(avg90) + 1], cols[cols.index(avg30) + 1]
    rows = []
    for window, src, n in (("latest", latest, None), ("30d", avg30, n30), ("90d", avg90, n90)):
        pair = d[src].map(parse_pair)
        rows.append(pd.DataFrame({"year": d[yc].map(parse_year), "sheet": "reliable", "window": window,
                                  "approve": as_percent(pair.str[0]), "disapprove": as_percent(pair.str[1]),
                                  "n_polls": (pd.to_numeric(d[n], errors="coerce")
                                              if n is not None else np.nan)}))
    return pd.concat(rows, ignore_index=True)


def load_approval(path) -> pd.DataFrame:
    """
    Long table: year, sheet (broad | reliable), window (latest | 30d | 90d),
    approve, disapprove (percent), n_polls. Blank spacer rows are dropped.
    Sheets are found by position: first = broad ('Unreliable data but more'),
    second = reliable.
    """
    x = pd.ExcelFile(path)
    if len(x.sheet_names) < 2:
        raise ValueError(f"approval workbook needs 2 sheets, found {x.sheet_names}")
    broad = _approval_broad(pd.read_excel(x, x.sheet_names[APPROVAL_SHEETS["broad"]]))
    rel = _approval_reliable(pd.read_excel(x, x.sheet_names[APPROVAL_SHEETS["reliable"]]))
    out = pd.concat([broad, rel], ignore_index=True).dropna(subset=["year"])
    out["year"] = out["year"].astype(int)
    if out.duplicated(["year", "sheet", "window"]).any():
        raise ValueError("approval: duplicate year/sheet/window rows")
    bad = out["approve"].notna() & ~out["approve"].between(5, 95)
    if bad.any():
        raise ValueError(f"approval: implausible values\n{out[bad]}")
    return out.sort_values(["sheet", "window", "year"]).reset_index(drop=True)


def approval_wide(approval: pd.DataFrame) -> pd.DataFrame:
    """year + approval_<sheet>_<window> / disapproval_<sheet>_<window>, with the window fallback."""
    a = approval.pivot_table(index="year", columns=["sheet", "window"], values="approve")
    b = approval.pivot_table(index="year", columns=["sheet", "window"], values="disapprove")
    out = pd.DataFrame(index=sorted(approval["year"].unique()))
    out.index.name = "year"
    for sheet in APPROVAL_SHEETS:
        for window in APPROVAL_WINDOWS:
            col = (sheet, window)
            ap = a[col].reindex(out.index) if col in a else pd.Series(np.nan, index=out.index)
            dis = b[col].reindex(out.index) if col in b else pd.Series(np.nan, index=out.index)
            for alt in APPROVAL_FALLBACK[window]:
                if (sheet, alt) in a:
                    fill = ap.isna()
                    ap = ap.where(~fill, a[(sheet, alt)].reindex(out.index))
                    dis = dis.where(~fill, b[(sheet, alt)].reindex(out.index))
            out[f"approval_{sheet}_{window}"] = ap
            out[f"disapproval_{sheet}_{window}"] = dis
    return out.reset_index()


def build_national(generic_ballot, midterm_flag, president_flag, approval) -> pd.DataFrame:
    """One row per year: generic ballot, both flags, all six approval versions."""
    gb = load_generic_ballot(generic_ballot)
    mt = load_flag(midterm_flag, "midterm_flag", {0, 1})
    pf = load_flag(president_flag, "pres_flag", {-1, 1})
    ap = approval_wide(load_approval(approval))
    out = (gb[["year", "generic_ballot", "gb_imputed"]]
           .merge(mt, on="year", how="outer").merge(pf, on="year", how="outer")
           .merge(ap, on="year", how="outer").sort_values("year").reset_index(drop=True))
    # sanity: the midterm flag file must agree with the calendar (midterms are year % 4 == 2)
    wrong = out.dropna(subset=["midterm_flag"])
    wrong = wrong[wrong["midterm_flag"].astype(int) != (wrong["year"] % 4 == 2).astype(int)]
    if len(wrong):
        raise ValueError(f"midterm flag disagrees with the calendar for {wrong['year'].tolist()}")
    return out


# ------------------------------------------------------------------ incumbency
def load_incumbency(path) -> pd.DataFrame:
    """
    House + Senate sheets -> keys + inc_D, inc_R, incumbency_signed (+1 D incumbent
    running, -1 R, 0 neither or both), election ('G' regular / 'S' special).

    * Header row is found by looking for 'Year' (the sheets start with title rows).
    * Blank D/R cell = no candidate in that slot -> counts as 0 (not an incumbent).
    * House at-large seats are numbered 1 in the file; standardize_keys makes them 0.
    * 'S' rows are special elections (special=True). Training drops special races
      (baseline.drop_specials), but someone who WON a special is already marked 1 in
      later regular ('G') races, so their incumbency still counts.
    * Odd-year rows exist (off-year specials); they never match a training race.
    """
    frames = []
    x = pd.ExcelFile(path)
    for chamber in ("House", "Senate"):
        raw = pd.read_excel(x, chamber, header=None)
        hdr = raw.index[raw.iloc[:, 0].astype(str).str.strip().str.lower() == "year"][0]
        d = pd.read_excel(x, chamber, header=hdr)
        d.columns = [str(c).strip() for c in d.columns]
        e = d["Election"].astype(str).str.strip().str.upper()
        if not e.isin(["G", "S"]).all():
            raise ValueError(f"incumbency {chamber}: unexpected Election codes {sorted(set(e) - {'G', 'S'})}")
        out = pd.DataFrame({
            "year": d["Year"].map(parse_year), "chamber": chamber, "state": d["State"],
            "district": d["District"] if chamber == "House" else 0,
            "special": e.eq("S"), "election": e,
            "inc_D": pd.to_numeric(d["D"], errors="coerce"),
            "inc_R": pd.to_numeric(d["R"], errors="coerce"),
            "dem_candidate": d["D candidate(s)"], "rep_candidate": d["R candidate(s)"],
        })
        if chamber == "Senate":
            out["seat_class"] = pd.to_numeric(d["Seat class"], errors="coerce")
        frames.append(out)
    inc = standardize_keys(pd.concat(frames, ignore_index=True))
    for c in ("inc_D", "inc_R"):
        bad = inc[c].notna() & ~inc[c].isin([0, 1])
        if bad.any():
            raise ValueError(f"incumbency: {c} has values other than 0/1/blank")
    inc["incumbency_signed"] = inc["inc_D"].fillna(0) - inc["inc_R"].fillna(0)
    inc["year"] = inc["year"].astype(int)
    dup = inc.duplicated(KEYS, keep=False)
    if dup.any():
        raise ValueError(f"incumbency: duplicate races\n{inc.loc[dup, KEYS + ['election']]}")
    return inc.sort_values(KEYS).reset_index(drop=True)


# ------------------------------------------------------------------ CLI
def main(argv=None):
    from .config import load_config, repo_path

    ap = argparse.ArgumentParser(description="load + check the baseline input spreadsheets")
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--write", action="store_true",
                    help="also write data/processed/{national,incumbency}.csv")
    a = ap.parse_args(argv)
    d = load_config(a.config)["data"]
    nat = build_national(repo_path(d["generic_ballot"]), repo_path(d["midterm_flag"]),
                         repo_path(d["president_flag"]), repo_path(d["approval"]))
    inc = load_incumbency(repo_path(d["incumbency"]))
    pd.set_option("display.width", 200)
    show = (["year", "generic_ballot", "midterm_flag", "pres_flag"]
            + [f"approval_{s}" for s in APPROVAL_SOURCES])
    print(nat[show].round(3).to_string(index=False))
    print(f"\nincumbency: {len(inc)} races "
          f"({inc['special'].sum()} specials); per chamber {inc['chamber'].value_counts().to_dict()}")
    if a.write:
        out = repo_path("data/processed")
        out.mkdir(parents=True, exist_ok=True)
        nat.to_csv(out / "national.csv", index=False)
        inc.to_csv(out / "incumbency.csv", index=False)
        print(f"wrote {out / 'national.csv'} and {out / 'incumbency.csv'}")


if __name__ == "__main__":
    main()
