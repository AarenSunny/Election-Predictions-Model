"""
election_prediction.inputs on small workbooks shaped like the team's spreadsheets,
including their quirks (title rows, '66 / 30' text, fractions, blank spacer rows,
'R+1.7' margin-only years, at-large seats numbered 1, special-election rows).
"""

import numpy as np
import pandas as pd
import pytest

from election_prediction.features import (
    add_baseline_inputs,
    approval_x_pres,
    approval_x_pres_midterm,
    midterm_x_pres,
)
from election_prediction.inputs import (
    approval_wide,
    build_national,
    load_approval,
    load_flag,
    load_generic_ballot,
    load_incumbency,
    parse_margin,
    parse_pair,
    parse_year,
)


def test_small_parsers():
    assert parse_year(2026) == 2026 and parse_year("2026.0") == 2026 and parse_year(" 2026 ") == 2026
    assert parse_year(pd.Timestamp("2026-11-03")) == 2026 and np.isnan(parse_year("n/a"))
    assert parse_margin("D+8.9") == 8.9 and parse_margin("R+1.7") == -1.7 and parse_margin("Tie") == 0
    assert parse_pair("66 / 30") == (66.0, 30.0) and np.isnan(parse_pair(np.nan)[0])


def _write(path, sheets: dict):
    with pd.ExcelWriter(path) as w:
        for name, df in sheets.items():
            df.to_excel(w, sheet_name=name, index=False, header=name != "_noheader")


def _national_files(tmp_path):
    gb = pd.DataFrame({"Year": [2004, 2002, 2000], "Dem": [45.4, np.nan, 47], "Rep": [45.4, np.nan, 46],
                       "Margin": ["Tie", "R+1.7", "D+1"], "Source / status": ["RCP", "margin only", "x"]})
    mt = pd.DataFrame({"Year": [2004, 2002, 2000], "Number": [0, 1, 0]})
    pf = pd.DataFrame({"Year": ["2004", "2002", "2000"], "Number": [-1, -1, 1]})   # years as text
    broad = pd.DataFrame({
        "year": [2000, 2002, 2004], "president": ["Clinton", "Bush", "Bush"],
        "last_gallup_approve": [57, 0.63, 51], "last_gallup_disapprove": [38, 0.29, 46],  # fraction
        "last_gallup_end_date": ["2000-10-28"] * 3,
        "avg_all_polls_90d_approve": [59.3, 61.2, 49.8], "avg_all_polls_90d_disapprove": [36.5, 27.7, 46.2],
        "n_polls_90d": [43, 65, 87],
        "avg_all_polls_30d_approve": [59.1, np.nan, 49.5],                               # blank -> fallback
        "avg_all_polls_30d_disapprove": [36.8, np.nan, 46.8], "n_polls_30d": [17, 0, 40]})
    rel = pd.DataFrame({
        "Year": [2000, 2002, np.nan, 2004], "Latest poll": ["57 / 38", "63 / 29", np.nan, "48 / 47"],
        "Poll end date": ["x"] * 4, "90-day average": ["59.5 / 36.3", "66.2 / 28.3", np.nan, "50.3 / 46.7"],
        "Records": [6, 11, np.nan, 11],
        "30-day average": ["57.5 / 37.5", "64.8 / 29.0", np.nan, "49.2 / 47.7"],
        "Records.1": [2, 4, np.nan, 6]})
    paths = {k: tmp_path / f"{k}.xlsx" for k in ("gb", "mt", "pf", "ap")}
    _write(paths["gb"], {"Sheet1": gb})
    _write(paths["mt"], {"Sheet1": mt})
    _write(paths["pf"], {"Sheet1": pf})
    _write(paths["ap"], {"Unreliable data but more": broad, "Reliable": rel})
    return paths


def test_generic_ballot_two_party_and_margin_only_year(tmp_path):
    p = _national_files(tmp_path)
    g = load_generic_ballot(p["gb"]).set_index("year")
    assert np.isclose(g.loc[2000, "generic_ballot"], 1 / 93)        # (47 - 46) / (47 + 46)
    assert g.loc[2004, "generic_ballot"] == 0
    # 2002 has only 'R+1.7': divided by the average D+R share of the other years
    share = np.mean([45.4 + 45.4, 47 + 46])
    assert np.isclose(g.loc[2002, "generic_ballot"], -1.7 / share) and g.loc[2002, "gb_imputed"]


def test_flags_and_bad_values(tmp_path):
    p = _national_files(tmp_path)
    assert load_flag(p["pf"], "pres_flag", {-1, 1})["year"].tolist() == [2000, 2002, 2004]
    with pytest.raises(ValueError):
        load_flag(p["mt"], "midterm_flag", {5})


def test_approval_both_sheets(tmp_path):
    p = _national_files(tmp_path)
    a = load_approval(p["ap"])
    assert len(a) == 3 * 2 * 3                                       # spacer row dropped
    b = a.set_index(["sheet", "window", "year"])["approve"]
    assert b[("broad", "latest", 2002)] == 63.0                      # 0.63 -> 63
    assert b[("reliable", "90d", 2002)] == 66.2                      # '66.2 / 28.3'
    w = approval_wide(a).set_index("year")
    assert w.loc[2002, "approval_broad_30d"] == 61.2                 # blank 30d -> 90d fallback
    assert w.loc[2004, "disapproval_reliable_30d"] == 47.7


def test_build_national_checks_midterm_calendar(tmp_path):
    p = _national_files(tmp_path)
    n = build_national(p["gb"], p["mt"], p["pf"], p["ap"])
    assert list(n["year"]) == [2000, 2002, 2004] and n["pres_flag"].tolist() == [1, -1, -1]
    _write(p["mt"], {"Sheet1": pd.DataFrame({"Year": [2004, 2002, 2000], "Number": [1, 1, 0]})})
    with pytest.raises(ValueError, match="calendar"):
        build_national(p["gb"], p["mt"], p["pf"], p["ap"])


def _incumbency_file(tmp_path):
    title = [["House incumbency | 1998-2026"] + [None] * 11, ["1 = sitting member"] + [None] * 11,
             [None] * 12]
    cols = ["Year", "State", "District", "Election", "D", "R", "D candidate(s)", "R candidate(s)",
            "D actual party", "R actual party", "Source ID", "Notes"]
    house = [[2018, "NY", 19, "G", 0, 1, "Delgado", "Faso", "D", "R", "S1", None],
             [2018, "WY", 1, "G", 0, 1, "Hunter", "Cheney", "D", "R", "S1", None],    # at-large numbered 1
             [2018, "PA", 7, "S", 0, 0, "Wild", "Nothstein", "D", "R", "S1", None],  # special
             [2018, "AL", 1, "G", None, 1, None, "Byrne", None, "R", "S1", None]]   # no D candidate
    scols = cols.copy()
    scols[2] = "Seat class"
    senate = [[2018, "MN", 1, "G", 1, 0, "Klobuchar", "Newberger", "D", "R", "S2", None],
              [2018, "MN", 2, "S", 1, 0, "Smith", "Housley", "D", "R", "S2", None]]
    path = tmp_path / "inc.xlsx"
    with pd.ExcelWriter(path) as w:
        pd.DataFrame(title + [cols] + house).to_excel(w, sheet_name="House", index=False, header=False)
        pd.DataFrame(title + [scols] + senate).to_excel(w, sheet_name="Senate", index=False, header=False)
    return path


def test_load_incumbency(tmp_path):
    inc = load_incumbency(_incumbency_file(tmp_path)).set_index(
        ["year", "chamber", "state", "district", "special"])["incumbency_signed"]
    assert inc[(2018, "House", "NY", 19, False)] == -1
    assert inc[(2018, "House", "WY", 0, False)] == -1            # at-large 1 -> 0
    assert inc[(2018, "House", "AL", 1, False)] == -1            # blank D slot counts as 0
    assert inc[(2018, "House", "PA", 7, True)] == 0              # special kept, flagged special
    assert inc[(2018, "Senate", "MN", 0, False)] == 1
    assert inc[(2018, "Senate", "MN", 0, True)] == 1             # same state-year, special seat


def test_baseline_input_formulas():
    # Bush 2008 at 25%: (25 - 50) x -1 = +25 (helps Ds); Clinton 1998 at 66%: +16
    assert approval_x_pres([25, 66], [-1, 1]).tolist() == [25, 16]
    assert approval_x_pres_midterm([25, 66], [-1, 1], [0, 1]).tolist() == [0, 16]
    assert midterm_x_pres([1, 1, 0], [1, -1, 1]).tolist() == [1, -1, 0]


def test_add_baseline_inputs_joins_by_year_and_race(tmp_path):
    p = _national_files(tmp_path)
    nat = build_national(p["gb"], p["mt"], p["pf"], p["ap"])
    races = pd.DataFrame({"year": [2002, 2004], "chamber": ["House", "House"], "state": ["NY", "NY"],
                          "district": [19, 19], "special": [False, False], "lean": [0.1, 0.1]})
    inc = races[["year", "chamber", "state", "district", "special"]].assign(incumbency_signed=[1, 0])
    out = add_baseline_inputs(races, nat, inc, "reliable_90d")
    assert out["approval"].tolist() == [66.2, 50.3]
    assert out["approval_x_pres"].round(6).tolist() == [-16.2, -0.3]   # R president: popular = helps R
    assert out["midterm_x_pres"].tolist() == [-1, 0]
    assert out["incumbency_signed"].tolist() == [1, 0]
    with pytest.raises(KeyError):
        add_baseline_inputs(races, nat, inc, "reliable_7d")
