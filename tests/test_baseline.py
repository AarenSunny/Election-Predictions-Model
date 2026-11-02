"""Stage-1 baseline: table building, fitting, and leak-free predictions (hand-made rows)."""

import numpy as np
import pandas as pd

from election_prediction.baseline import baseline_residuals, describe_baseline, fit_baseline

from election_prediction.data import build_baseline_table


def _results():
    rows = []
    for year in (2018, 2020, 2022):
        for d, lean in enumerate([-0.2, -0.1, 0.0, 0.1, 0.2], start=1):
            dem = 1000 * (1 + lean)          # margin = lean exactly
            rows.append({"year": year, "chamber": "House", "state": "NY", "district": d,
                         "special": False, "dem_votes": dem, "rep_votes": 2000 - dem,
                         "total_votes": 2000, "uncontested": False, "same_party": False})
    return pd.DataFrame(rows)


def _lean():
    r = _results()
    return r[["year", "chamber", "state", "district", "special"]].assign(
        lean=[-0.2, -0.1, 0.0, 0.1, 0.2] * 3)


def test_margin_equals_lean_gives_slope_one_and_zero_residuals():
    table = build_baseline_table(_results(), _lean(), verbose=False)
    coef = describe_baseline(table, ["lean"])
    assert np.isclose(coef["lean"], 1.0) and np.isclose(coef["intercept"], 0.0)
    out = baseline_residuals(table, ["lean"])
    assert np.allclose(out["residual"], 0.0, atol=1e-9)


def test_table_drops_bad_races_and_keeps_keys_unique():
    res = _results()
    res.loc[0, "uncontested"] = True
    res.loc[1, "same_party"] = True
    res.loc[2, ["dem_votes", "rep_votes", "total_votes"]] = [300, 400, 2000]   # third party won
    lean = _lean().drop(index=3)                                                # missing lean
    table = build_baseline_table(res, lean, verbose=False)
    assert len(table) == 15 - 4
    assert not table["race_id"].duplicated().any()
    assert np.isclose(table.loc[table["race_id"] == "2018-H-NY-05", "margin"].iloc[0], 0.2)


def test_predictions_are_out_of_fold():
    # 2022 races sit 0.1 above the line; a model trained WITH 2022 would shrink that gap.
    table = build_baseline_table(_results(), _lean(), verbose=False)
    table.loc[table["year"] == 2022, "margin"] += 0.1
    out = baseline_residuals(table, ["lean"])
    gap_2022 = out.loc[out["year"] == 2022, "residual"].mean()
    in_sample = table["margin"] - fit_baseline(table, ["lean"]).predict(table[["lean"]])
    assert gap_2022 > in_sample[table["year"] == 2022].mean()


def test_build_baseline_table_drops_specials_by_default():
    from election_prediction.data import build_baseline_table

    keys = {"chamber": "House", "state": "NY", "district": 19}
    res = pd.DataFrame([{**keys, "year": 2018, "special": s, "dem_votes": 60, "rep_votes": 40,
                         "total_votes": 100, "uncontested": False, "same_party": False}
                        for s in (False, True)])
    lean = res[["year", "chamber", "state", "district", "special"]].assign(lean=0.1)
    assert len(build_baseline_table(res, lean, verbose=False)) == 1
    assert len(build_baseline_table(res, lean, verbose=False, drop_specials=False)) == 2


def test_compare_score_prefers_the_input_that_matters():
    from election_prediction.compare import score

    rng = np.random.default_rng(0)
    rows = []
    for y in range(2000, 2016, 2):
        lean = rng.uniform(-0.4, 0.4, 60)
        inc = rng.choice([-1, 0, 1], 60)
        rows.append(pd.DataFrame({"year": y, "lean": lean, "incumbency_signed": inc,
                                  "margin": lean + 0.1 * inc + rng.normal(0, 0.03, 60)}))
    t = pd.concat(rows, ignore_index=True)
    with_inc, without = score(t, ["lean", "incumbency_signed"]), score(t, ["lean"])
    assert with_inc["rmse"] < without["rmse"] and not with_inc["wrong_sign"]
    assert abs(with_inc["coefs"]["incumbency_signed"] - 0.1) < 0.02
