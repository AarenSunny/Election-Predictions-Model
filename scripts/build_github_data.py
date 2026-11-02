"""
Build tidy tables (project data contract: see data/README.md) from public GitHub repos.

Sources (clone each with `git clone --depth 1 <url> <dir>`):
  fivethirtyeight/election-results   https://github.com/fivethirtyeight/election-results
  fivethirtyeight/data               https://github.com/fivethirtyeight/data
  unitedstates/congress-legislators  https://github.com/unitedstates/congress-legislators

Usage:
  python scripts/build_github_data.py --e538 <election-results dir> --d538 <data dir> \
      --legis <congress-legislators dir> --out data/raw

Writes:
  results_538_1998_2024.csv      one row per House/Senate general race: D/R votes, flags
  incumbency_1998_2026.csv       incumbent party, whether a sitting member ran, terms served
  polls_538_1998_2022.csv        House/Senate general polls (final ~3 weeks before each election)

1998 is the first election with a Cook PVI (the 1997 sheet) to use as lean.
  pollster_ratings_538_2023.csv  538 pollster grades + the 0-1 weight used in polls file
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

START = 1998
YEARS = range(START, 2027, 2)
KEYS = ["year", "chamber", "state", "district", "special"]
STATES_50 = {"AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN",
             "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV",
             "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN",
             "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"}   # no DC/territory delegates
D_LINES = {"DEM", "DFL", "D"}       # Minnesota DFL counts as Democratic
R_LINES = {"REP", "R"}

# 538 letter grade -> poll weight (pollster_rating). Documented choice; edit freely.
GRADE_WEIGHT = {"A+": 1.0, "A": 0.95, "A-": 0.9, "A/B": 0.85, "B+": 0.8, "B": 0.75,
                "B-": 0.7, "B/C": 0.65, "C+": 0.6, "C": 0.55, "C-": 0.5, "C/D": 0.45,
                "D+": 0.4, "D": 0.35, "D-": 0.3, "F": 0.1}
DEFAULT_WEIGHT = 0.65   # unrated pollster -> most common grade (B/C); flagged


# ------------------------------------------------------------------ helpers
def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z ]", " ", s.lower()).strip()


def seat_to_district(seat: str) -> int:
    m = re.search(r"District\s+(\d+)", str(seat))
    return int(m.group(1)) if m else 0   # "At-Large" and Senate classes -> 0


# ------------------------------------------------------------------ results
def november_race_ids(e538: Path) -> set:
    """
    Race ids that are the November contest: stage 'general', or a 'jungle primary'
    held in November (Louisiana House, Georgia 2020 Senate special). California's
    June top-two primaries are also called jungle primaries and are excluded.
    """
    r = pd.read_csv(e538 / "races.csv")
    month = pd.to_datetime(r["date"], format="%m/%d/%y", errors="coerce").dt.month
    keep = (r["stage"] == "general") | ((r["stage"] == "jungle primary") & (month == 11))
    return set(r.loc[keep, "id"])


def build_results(e538: Path) -> pd.DataFrame:
    nov = november_race_ids(e538)
    frames = []
    for chamber, fn in (("House", "election_results_house.csv"),
                        ("Senate", "election_results_senate.csv")):
        d = pd.read_csv(e538 / fn, low_memory=False)
        d = d[d["race_id"].isin(nov) & d["cycle"].isin(list(YEARS))].copy()
        d["chamber"] = chamber
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    # ranked-choice: keep first round only (consistent with plurality races; flagged)
    d["rcv"] = d["ranked_choice_round"].notna()
    d = d[d["ranked_choice_round"].isna() | (d["ranked_choice_round"] == 1)]
    d = d[d["state_abbrev"].isin(STATES_50)]
    d["year"] = d["cycle"].astype(int)
    d["state"] = d["state_abbrev"]
    d["district"] = np.where(d["chamber"] == "Senate", 0, d["office_seat_name"].map(seat_to_district))
    d["special"] = d["special"].astype(str).str.lower().eq("true")
    d["jungle"] = d["stage"] == "jungle primary"
    d["line"] = np.select([d["ballot_party"].isin(D_LINES), d["ballot_party"].isin(R_LINES)],
                          ["D", "R"], default="O")

    # Votes per candidate across ballot lines (fusion); a candidate is D/R if any line is.
    cand = (d.dropna(subset=["candidate_id"])
              .groupby(KEYS + ["race_id", "candidate_id", "candidate_name"], dropna=False)
              .agg(votes=("votes", "sum"), unopposed=("unopposed", "max"),
                   winner=("winner", "max"), rcv=("rcv", "max"), jungle=("jungle", "max"),
                   party=("line", lambda s: "D" if (s == "D").any() else ("R" if (s == "R").any() else "O")))
              .reset_index())
    rows = []
    for k, g in cand.groupby(KEYS + ["race_id"]):
        rec = dict(zip(KEYS + ["race_id"], k))  # noqa: B905 (strict= needs Python 3.10)
        dd = g[g["party"] == "D"].sort_values("votes", ascending=False)
        rr = g[g["party"] == "R"].sort_values("votes", ascending=False)
        rec.update(
            dem_votes=dd["votes"].iloc[0] if len(dd) else np.nan,
            rep_votes=rr["votes"].iloc[0] if len(rr) else np.nan,
            dem_name=dd["candidate_name"].iloc[0] if len(dd) else None,
            rep_name=rr["candidate_name"].iloc[0] if len(rr) else None,
            total_votes=g["votes"].sum(),
            n_candidates=len(g),
            winner_party=g.loc[g["winner"] == True, "party"].iloc[0] if (g["winner"] == True).any() else None,  # noqa: E712
            uncontested=bool((g["unopposed"] == True).any()) or len(g) == 1,  # noqa: E712
            rcv=bool(g["rcv"].any()),
            jungle_primary=bool(g["jungle"].any()),
            n_dem=len(dd), n_rep=len(rr),
        )
        rows.append(rec)
    out = pd.DataFrame(rows)
    out["same_party"] = (out["n_dem"] == 0) | (out["n_rep"] == 0)
    return out.sort_values(KEYS).reset_index(drop=True)


# ------------------------------------------------------------------ legislators
def load_terms(legis: Path) -> pd.DataFrame:
    rows = []
    for fn in ("legislators-historical.yaml", "legislators-current.yaml"):
        for p in yaml.safe_load(open(legis / fn)):
            nm = p.get("name", {})
            for t in p.get("terms", []):
                # keep every term, however old: prior-terms counts need the full career
                if t["type"] not in ("rep", "sen"):
                    continue
                rows.append({"bioguide": p["id"].get("bioguide"),
                             "first": nm.get("first", ""), "nick": nm.get("nickname", ""),
                             "last": nm.get("last", ""),
                             "chamber": "House" if t["type"] == "rep" else "Senate",
                             "state": t["state"], "district": max(int(t.get("district", 0) or 0), 0),
                             "party": {"Democrat": "D", "Republican": "R"}.get(t.get("party"), "O"),
                             "start": t["start"], "end": t["end"]})
    return pd.DataFrame(rows)


def terms_before(terms: pd.DataFrame, bioguide: str, chamber: str, date: str) -> int:
    t = terms[(terms["bioguide"] == bioguide) & (terms["chamber"] == chamber) & (terms["start"] < date)]
    return len(t)


def name_match(member: pd.Series, cand_name: str) -> bool:
    c = norm(cand_name).split()
    last = norm(member["last"]).split()
    if not last or not c:
        return False
    if not all(tok in c for tok in last):
        return False
    initials = {norm(member["first"])[:1], norm(member["nick"])[:1]} - {""}
    return any(tok[:1] in initials for tok in c[: max(1, len(c) - len(last))])


def build_incumbency(e538: Path, legis: Path, results: pd.DataFrame) -> pd.DataFrame:
    races = pd.read_csv(e538 / "races.csv")
    nov = november_race_ids(e538)
    races = races[races["type"].isin(["HouseRace", "SenateRace"]) & races["id"].isin(nov)
                  & races["cycle"].isin(list(YEARS)) & races["state_abbrev"].isin(STATES_50)].copy()
    races["year"] = races["cycle"].astype(int)
    races["chamber"] = np.where(races["type"] == "SenateRace", "Senate", "House")
    races["state"] = races["state_abbrev"]
    races["district"] = np.where(races["chamber"] == "Senate", 0,
                                 races["office_seat_name"].map(seat_to_district))
    races["special"] = races["special"].astype(str).str.lower().eq("true")
    ip = races["incumbent_party"].fillna("")
    races["incumbent_party"] = np.select([ip == "DEM", ip == "REP", ip.isin(["vacant", "new_seat", ""])],
                                         ["D", "R", ""], default="O")
    races["seat_status"] = np.where(ip.isin(["vacant", "new_seat"]), ip, "")
    races = races.drop_duplicates(KEYS)

    terms = load_terms(legis)
    out = []
    for _, r in races.iterrows():
        rec = {k: r[k] for k in KEYS}
        rec.update(incumbent_party=r["incumbent_party"], seat_status=r["seat_status"],
                   incumbent_running=None, incumbent_name=None, incumbent_prior_terms=np.nan)
        res = results[(results["year"] == r["year"]) & (results["chamber"] == r["chamber"])
                      & (results["state"] == r["state"]) & (results["district"] == r["district"])
                      & (results["special"] == r["special"])]
        if r["year"] <= 2024 and len(res):
            edate = f"{r['year']}-10-01"
            serving = terms[(terms["chamber"] == r["chamber"]) & (terms["state"] == r["state"])
                            & (terms["start"] <= edate) & (terms["end"] >= edate)]
            names = [n for n in (res["dem_name"].iloc[0], res["rep_name"].iloc[0]) if isinstance(n, str)]
            hit = None
            for _, m in serving.iterrows():
                if any(name_match(m, n) for n in names):
                    hit = m
                    break
            rec["incumbent_running"] = hit is not None
            if hit is not None:
                rec["incumbent_name"] = f"{hit['first']} {hit['last']}"
                rec["incumbent_running_party"] = hit["party"]
                rec["incumbent_prior_terms"] = terms_before(terms, hit["bioguide"], r["chamber"], edate) - 1
        out.append(rec)
    inc = pd.DataFrame(out)
    party = inc.get("incumbent_running_party", pd.Series(index=inc.index, dtype=object))
    inc["incumbency_signed"] = np.where(inc["incumbent_running"] == True,  # noqa: E712
                                        np.select([party == "D", party == "R"], [1, -1], 0),
                                        np.where(inc["incumbent_running"] == False, 0, np.nan))  # noqa: E712
    return inc.sort_values(KEYS).reset_index(drop=True)


# ------------------------------------------------------------------ polls
def build_pollster_ratings(d538: Path) -> pd.DataFrame:
    pr = pd.read_csv(d538 / "pollster-ratings/2023/pollster-ratings.csv")
    out = pd.DataFrame({
        "pollster": pr["Pollster"], "pollster_rating_id": pr["Pollster Rating ID"],
        "grade_538": pr["538 Grade"], "predictive_plus_minus": pr["Predictive Plus-Minus"],
        "banned_by_538": pr["Banned by 538"].astype(str).str.lower().eq("yes"),
    })
    out["pollster_rating"] = out["grade_538"].map(GRADE_WEIGHT)
    out.loc[out["banned_by_538"], "pollster_rating"] = 0.0
    return out


def build_polls(d538: Path, ratings: pd.DataFrame) -> pd.DataFrame:
    p = pd.read_csv(d538 / "pollster-ratings/2023/raw-polls.csv")
    p = p[p["type_simple"].isin(["House-G", "Sen-G"]) & (p["location"] != "US")
          & p["year"].isin(list(YEARS))].copy()
    p["chamber"] = np.where(p["type_simple"] == "Sen-G", "Senate", "House")
    loc = p["location"].str.extract(r"^([A-Z]{2})(?:-(\d+))?$")
    p["state"] = loc[0].where(loc[0].isin(STATES_50))
    p["district"] = pd.to_numeric(loc[1], errors="coerce").fillna(0).astype(int)
    p["special"] = p["type_detail"].astype(str).str.contains("Special|-S$|GS", regex=True)

    def pct(row, party):
        for i in (1, 2):
            if row[f"cand{i}_party"] == party:
                return row[f"cand{i}_pct"]
        return np.nan
    p["dem_pct"] = p.apply(pct, axis=1, party="DEM")
    p["rep_pct"] = p.apply(pct, axis=1, party="REP")
    p = p.dropna(subset=["dem_pct", "rep_pct", "state"])
    p = p.merge(ratings[["pollster_rating_id", "pollster_rating", "grade_538"]],
                on="pollster_rating_id", how="left")
    p["rating_imputed"] = p["pollster_rating"].isna()
    p["pollster_rating"] = p["pollster_rating"].fillna(DEFAULT_WEIGHT)
    p["internal"] = p["partisan"].notna()
    out = pd.DataFrame({
        "year": p["year"].astype(int), "chamber": p["chamber"], "state": p["state"],
        "district": p["district"], "special": p["special"],
        "end_date": pd.to_datetime(p["polldate"], format="%m/%d/%Y").dt.strftime("%Y-%m-%d"),
        "pollster": p["pollster"], "sample_size": p["samplesize"],
        "dem_pct": p["dem_pct"], "rep_pct": p["rep_pct"],
        "pollster_rating": p["pollster_rating"], "internal": p["internal"],
        "rating_imputed": p["rating_imputed"], "grade_538": p["grade_538"],
        "methodology": p["methodology"], "poll_id": p["poll_id"], "race_538": p["race"],
    })
    return out.sort_values(KEYS + ["end_date"]).reset_index(drop=True)


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--e538", required=True, type=Path)
    ap.add_argument("--d538", required=True, type=Path)
    ap.add_argument("--legis", required=True, type=Path)
    ap.add_argument("--out", default=Path("data/raw"), type=Path)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)

    res = build_results(a.e538)
    res.drop(columns=["race_id"]).to_csv(a.out / f"results_538_{START}_2024.csv", index=False)
    print(f"results: {len(res)} races")

    inc = build_incumbency(a.e538, a.legis, res)
    inc.to_csv(a.out / f"incumbency_{START}_2026.csv", index=False)
    print(f"incumbency: {len(inc)} races")

    rat = build_pollster_ratings(a.d538)
    rat.to_csv(a.out / "pollster_ratings_538_2023.csv", index=False)
    pol = build_polls(a.d538, rat)
    pol.to_csv(a.out / f"polls_538_{START}_2022.csv", index=False)
    print(f"polls: {len(pol)} polls, ratings: {len(rat)} pollsters")


if __name__ == "__main__":
    main()
