import numpy as np
import pandas as pd
import pytest

from election_prediction import baseline, train
from election_prediction.features import (
    add_horizon_changes,
    add_missing_flags,
    approval_signed,
    candidates_to_wide,
    change_over,
    county_to_district,
    d_share,
    distance_advantage,
    election_day,
    incumbency_x_lean,
    midterm_flag,
    midterm_pres_penalty,
    moderation,
    money_features,
    poll_average,
    poll_cutoff,
    poll_gap,
    pres_party_D,
    signed_incumbency,
    two_party_margin,
)


def test_basic_transforms():
    assert np.allclose(two_party_margin([60], [40]), [0.2])
    assert np.isnan(d_share([0], [0])[0])
    assert signed_incumbency(["D", "R", "open"]).tolist() == [1, -1, 0]
    assert approval_signed([-0.1, -0.1], ["D", "R"]).tolist() == [-0.1, 0.1]
    assert midterm_flag([2018, 2020]).tolist() == [1, 0]


def test_moderation_sign_means_moderate_for_both_parties():
    # D at -0.5 vs party mean -1.0 is moderate; R at +0.5 vs mean +1.0 is moderate.
    m = moderation([-0.5, 0.5, -1.5, 1.5], ["D", "R", "D", "R"], [-1.0, 1.0, -1.0, 1.0])
    assert (m[:2] > 0).all() and (m[2:] < 0).all()


def test_distance_advantage_positive_when_D_closer():
    assert distance_advantage([-0.2], [1.0], [0.0])[0] > 0


def test_candidates_to_wide_and_money():
    c = pd.DataFrame({"year": [2018] * 3, "chamber": ["House"] * 3, "state": ["NY"] * 3,
                      "district": [1] * 3, "special": [False] * 3, "party": ["D", "R", "O"],
                      "receipts": [300.0, 100.0, 5.0], "self_funding": [0.0, 50.0, 0.0],
                      "cash_on_hand": [50.0, 50.0, 1.0], "pac_contrib": [30.0, 50.0, 0.0]})
    w = candidates_to_wide(c, ["receipts", "self_funding", "cash_on_hand", "pac_contrib"])
    assert len(w) == 1 and "O_receipts" not in w
    m = money_features(w)
    assert np.isclose(m.loc[0, "fundraising_share"], 300 / 350)   # R self-funding excluded
    assert np.isclose(m.loc[0, "cash_share"], 0.5)
    assert np.isclose(m.loc[0, "pac_tilt"], 0.1 - 0.5)


def test_change_over_matches_same_geography():
    df = pd.DataFrame({"state": ["NY", "NY", "TX"], "year": [2018, 2022, 2022],
                       "income": [50.0, 60.0, 70.0]})
    ch = change_over(df, "income", ["state"], 4)
    assert np.isnan(ch.iloc[0]) and ch.iloc[1] == 10 and np.isnan(ch.iloc[2])


def test_missing_flags_by_group():
    df = pd.DataFrame({"a": [1, np.nan], "b": [np.nan, np.nan]})
    out = add_missing_flags(df, {"polls_missing": ["a", "b"]})
    assert out["polls_missing"].tolist() == [0, 1]


def test_county_to_district_weighted():
    counties = pd.DataFrame({"county_fips": [1, 2], "unemp": [0.04, 0.08]})
    xw = pd.DataFrame({"state": ["NY", "NY"], "district": [1, 1],
                       "county_fips": [1, 2], "afact": [0.75, 0.25]})
    out = county_to_district(counties, xw, ["unemp"])
    assert np.isclose(out.loc[0, "unemp"], 0.05) and out.loc[0, "largest_county_share"] == 0.75


def test_poll_average_ignores_future_polls():
    keys = {"year": 2026, "chamber": "House", "state": "NY", "district": 17, "special": False}
    polls = pd.DataFrame([
        {**keys, "end_date": "2026-10-20", "pollster": "A", "sample_size": 400,
         "dem_pct": 48, "rep_pct": 44},
        {**keys, "end_date": "2026-11-02", "pollster": "B", "sample_size": 400,
         "dem_pct": 30, "rep_pct": 60},   # after as_of: must be ignored
    ])
    out = poll_average(polls, as_of="2026-11-01")
    # two-party margin (48 - 44) / (48 + 44); the Nov 2 poll is after the cutoff
    assert out.loc[0, "n_polls_used"] == 1
    assert np.isclose(out.loc[0, "poll_margin"], 4 / 92)
    assert out.loc[0, "days_since_poll"] == 12 and out.loc[0, "poll_count_30d"] == 1


def test_poll_weights_rating_internal_and_recency():
    keys = {"year": 2026, "chamber": "House", "state": "NY", "district": 17, "special": False}
    polls = pd.DataFrame([
        # same age and size; poll A rated 1.0, poll B internal (x0.5) -> A gets 2/3 of the weight
        {**keys, "end_date": "2026-10-20", "pollster": "A", "sample_size": 400,
         "dem_pct": 55, "rep_pct": 45, "pollster_rating": 1.0, "internal": False},
        {**keys, "end_date": "2026-10-20", "pollster": "B", "sample_size": 400,
         "dem_pct": 45, "rep_pct": 55, "pollster_rating": 1.0, "internal": True},
        # 70 days old: outside the 60-day window
        {**keys, "end_date": "2026-08-20", "pollster": "C", "sample_size": 400,
         "dem_pct": 90, "rep_pct": 10},
    ])
    out = poll_average(polls, as_of="2026-10-29")
    assert out.loc[0, "n_polls_used"] == 2
    assert np.isclose(out.loc[0, "poll_margin"], (2 / 3) * 0.10 + (1 / 3) * -0.10)


def test_election_day_and_poll_cutoff():
    assert election_day(2026) == pd.Timestamp("2026-11-03")
    assert election_day(2022) == pd.Timestamp("2022-11-08")
    assert election_day(2016) == pd.Timestamp("2016-11-08")  # Nov 1 was a Tuesday
    assert poll_cutoff(2026, 5) == pd.Timestamp("2026-10-29")


def test_poll_gap_is_null_when_unpolled():
    gap = poll_gap([0.05, np.nan], [0.02, 0.02])
    assert np.isclose(gap[0], 0.03) and np.isnan(gap[1])


def test_baseline_inputs_from_doc():
    # midterm under a D president hurts Ds; presidential years -> 0
    assert midterm_pres_penalty([2022, 2018, 2024], ["D", "R", "D"]).tolist() == [-1, 1, 0]
    assert pres_party_D(["D", "R"]).tolist() == [1, -1]
    assert np.allclose(incumbency_x_lean(["D", "R", "open"], [0.05, 0.05, 0.05]),
                       [0.05, -0.05, 0.0])


def test_add_horizon_changes_names_and_values():
    df = pd.DataFrame({"state": ["NY"] * 3, "year": [2016, 2021, 2026],
                       "unemployment": [0.05, 0.06, 0.04]})
    out = add_horizon_changes(df, "unemployment", ["state"], [5, 10])
    assert {"unemployment_chg_5y", "unemployment_chg_10y"} <= set(out.columns)
    assert np.isclose(out.loc[2, "unemployment_chg_5y"], -0.02)
    assert np.isclose(out.loc[2, "unemployment_chg_10y"], -0.01)
    assert np.isnan(out.loc[0, "unemployment_chg_5y"])  # no earlier year -> Null


def test_donor_mix_diff():
    w = pd.DataFrame({"D_receipts": [100.0], "R_receipts": [100.0],
                      "D_self_funding": [0.0], "R_self_funding": [0.0],
                      "D_cash_on_hand": [10.0], "R_cash_on_hand": [10.0],
                      "D_pac_contrib": [10.0], "R_pac_contrib": [40.0],
                      "D_small_donor_contrib": [50.0], "R_small_donor_contrib": [20.0]})
    m = money_features(w)
    # D: 0.5 - 0.1 = 0.4 ; R: 0.2 - 0.4 = -0.2 ; diff = 0.6
    assert np.isclose(m.loc[0, "donor_mix_diff"], 0.6)
    assert np.isnan(money_features(w.drop(columns=["D_small_donor_contrib"])).loc[0, "donor_mix_diff"])


def test_model_files_are_still_skeletons():
    with pytest.raises(NotImplementedError):
        baseline.fit_baseline(pd.DataFrame())
    with pytest.raises(NotImplementedError):
        train.fit_residual_model(pd.DataFrame(), [], {})
