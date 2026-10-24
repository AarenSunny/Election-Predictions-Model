"""Required columns, unique ids, missing data, key cleanup."""

import numpy as np
import pandas as pd
import pytest

from election_prediction.data import (
    DataValidationError,
    load_mit_results,
    party_code,
    race_id,
    standardize_district,
    standardize_keys,
    standardize_state,
    validate,
)


def test_standardize_state_and_district():
    assert standardize_state(["ny", "New York", "TX", "Narnia"]).tolist()[:3] == ["NY", "NY", "TX"]
    assert pd.isna(standardize_state(["Narnia"])[0])
    assert standardize_district(["AL", "07", 3.0, "at-large"]).tolist() == [0, 7, 3, 0]


def test_at_large_and_senate_get_district_zero():
    df = pd.DataFrame({"year": [2022, 2022, 2022, 2018, 2022, 2022],
                       "chamber": ["House", "House", "House", "House", "House", "Senate"],
                       "state": ["WY", "NY", "NY", "MT", "MT", "NY"],
                       "district": [1, 1, 2, 1, 1, 5]})
    out = standardize_keys(df)
    # WY at-large; NY-1 stays 1 even alone; MT at-large only before 2022
    assert out["district"].tolist() == [0, 1, 2, 0, 1, 0]
    assert (~out["special"]).all()


def test_partial_table_keeps_district_numbers():
    out = standardize_keys(pd.DataFrame({"year": [2026], "chamber": ["House"],
                                         "state": ["NY"], "district": [17]}))
    assert out["district"].tolist() == [17]


def test_race_id_format():
    df = standardize_keys(pd.DataFrame({"year": [2026, 2026], "chamber": ["House", "Senate"],
                                        "state": ["NY", "OH"], "district": [17, 0],
                                        "special": [False, True]}))
    assert race_id(df).tolist() == ["2026-H-NY-17", "2026-S-OH-00-SP"]


def test_party_code():
    assert party_code(["DEMOCRAT", "DEMOCRATIC-FARMER-LABOR", "REPUBLICAN", "LIBERTARIAN", None]
                      ).tolist() == ["D", "D", "R", "O", "O"]


def _races():
    return pd.DataFrame({"year": [2018, 2018], "chamber": ["House", "House"],
                         "state": ["NY", "NY"], "district": [1, 2], "special": [False, False],
                         "dem_votes": [100, 200], "rep_votes": [150, 100]})


def test_validate_passes_clean_table():
    assert len(validate(_races(), "races")) == 2


def test_validate_reports_every_problem_at_once():
    bad = pd.concat([_races(), _races().iloc[[0]]], ignore_index=True)  # duplicate race
    bad.loc[1, "state"] = "XX"                   # bad state
    bad = bad.drop(columns="rep_votes")          # missing column
    with pytest.raises(DataValidationError) as e:
        validate(bad, "races")
    msg = str(e.value)
    assert "missing columns" in msg and "unknown state" in msg and "same" in msg


def test_validate_catches_percent_points_instead_of_fractions():
    nat = pd.DataFrame({"year": [2018], "generic_ballot": [8.6], "net_approval": [-0.1],
                        "pres_party": ["R"]})
    with pytest.raises(DataValidationError, match="units"):
        validate(nat, "national")


def test_load_mit_results_collapses_fusion_and_flags(tmp_path):
    rows = [
        # NY-1: D candidate on two lines (fusion), R candidate
        (2018, "NEW YORK", "NY", 1, "GEN", False, "ANN D", "DEMOCRAT", False, 100, 260),
        (2018, "NEW YORK", "NY", 1, "GEN", False, "ANN D", "WORKING FAMILIES", False, 20, 260),
        (2018, "NEW YORK", "NY", 1, "GEN", False, "BOB R", "REPUBLICAN", False, 140, 260),
        # CA-5: top-two with two Democrats
        (2018, "CALIFORNIA", "CA", 5, "GEN", False, "CAT D", "DEMOCRAT", False, 60, 100),
        (2018, "CALIFORNIA", "CA", 5, "GEN", False, "DAN D", "DEMOCRAT", False, 40, 100),
        # a primary row that must be ignored
        (2018, "NEW YORK", "NY", 1, "PRI", False, "EVE D", "DEMOCRAT", False, 999, 999),
    ]
    cols = ["year", "state", "state_po", "district", "stage", "special", "candidate", "party",
            "writein", "candidatevotes", "totalvotes"]
    p = tmp_path / "mit.csv"
    pd.DataFrame(rows, columns=cols).to_csv(p, index=False)
    out = load_mit_results(p, "House").set_index("state")
    assert out.loc["NY", "dem_votes"] == 120 and out.loc["NY", "rep_votes"] == 140
    assert not out.loc["NY", "same_party"]
    assert out.loc["CA", "same_party"] and out.loc["CA", "top_two_state"]
    assert np.isnan(out.loc["CA", "rep_votes"])
