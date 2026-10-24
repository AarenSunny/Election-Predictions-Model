"""
Tests for src/election_prediction/benchmarks.py
===============================================

WHAT benchmarks.py IS FOR
-------------------------
It answers one question: "Is our model actually earning points, or would a
simple strategy do just as well?" It compares four strategies:

    always_favorite   pick whoever the market says is more likely
    always_underdog   pick whoever the market says is LESS likely
    model_favorite    pick whoever OUR MODEL says is more likely
    model_ev          pick the side with the higher expected points
                      (P_model x -log2 P_market). This is our submission rule.

With this contest's scoring, always_underdog is the bar to beat: if the market
is accurate, picking underdogs has the higher expected score.

WHAT DATA THESE TESTS USE
-------------------------
No real data. Each test uses a tiny table typed out below, small enough that
every number can be checked by hand. The tests prove the CODE does the math
correctly. Whether the MODEL is good is a separate question that needs real
past races (run benchmarks.backtest on the real out-of-fold predictions).

Run just this file:   pytest tests/test_benchmarks.py -v
"""

import numpy as np
import pandas as pd

from election_prediction.benchmarks import (
    backtest,
    expected_totals,
    rating_to_prob_D,
    strategy_picks,
)

# ---------------------------------------------------------------------------
# 1. Converting expert ratings into a stand-in market
# ---------------------------------------------------------------------------
# Past Kalshi prices don't exist for most old races, so the backtest uses
# expert ratings (Cook / Sabato / Inside Elections) as a fake "market".
# rating_to_prob_D() turns a rating into P(Democrat wins):
#
#     Safe/Solid 97%   Likely 85%   Lean 70%   Tilt 60%   Toss-up 50%
#
# For an R rating the D probability is flipped: "Lean R" -> 1 - 0.70 = 0.30.
# Anything it can't read returns NaN (blank) so a typo can't silently become
# a real-looking probability.


def test_rating_to_prob():
    assert rating_to_prob_D("Safe D") == 0.97
    assert np.isclose(rating_to_prob_D("Lean Republican"), 0.30)  # full party names work
    assert rating_to_prob_D("Toss-up") == 0.5
    assert np.isnan(rating_to_prob_D("Very weird R"))  # unknown rating word
    assert np.isnan(rating_to_prob_D(None))  # blank cell


# ---------------------------------------------------------------------------
# 2. The example table used by the remaining tests
# ---------------------------------------------------------------------------
#  race  year  p_model_D  p_market_D  winner   notes
#  A     2018     0.80       0.70       D      model and market both favor D
#  B     2018     0.30       0.30       D      both favor R, but D wins (upset)
#  C     2022     0.60       0.50       R      market says coin flip
#  D     2022     0.50       (blank)    D      no market price -> must be skipped
#
# Points for a correct pick = -log2(market probability of the side picked):
#   D in race A: -log2(0.70) = 0.515     R in race A: -log2(0.30) = 1.737
#   D in race B: -log2(0.30) = 1.737     R in race B: -log2(0.70) = 0.515
#   either side in race C: -log2(0.50) = 1.000
# A wrong pick scores 0.


def _df():
    return pd.DataFrame({
        "year": [2018, 2018, 2022, 2022],
        "p_model_D": [0.8, 0.3, 0.6, 0.5],
        "p_market_D": [0.7, 0.3, 0.5, np.nan],
        "winner": ["D", "D", "R", "D"],
    })


# ---------------------------------------------------------------------------
# 3. Which side each strategy picks
# ---------------------------------------------------------------------------
# Race D is dropped first (no market price), leaving races A, B, C.
#
#  strategy          A   B   C    why
#  always_favorite   D   R   D    market favorite; a 50/50 tie counts as D
#  always_underdog   R   D   R    the opposite
#  model_favorite    D   R   D    model's more likely side (0.8, 0.3, 0.6)
#  model_ev          D   D   D    higher expected points, see below
#
# model_ev, expected points = P_model(side) x points if correct:
#   A: D = 0.8 x 0.515 = 0.412   vs  R = 0.2 x 1.737 = 0.347  -> D
#   B: D = 0.3 x 1.737 = 0.521   vs  R = 0.7 x 0.515 = 0.360  -> D
#      (the model agrees with the market here, and the underdog wins on EV)
#   C: D = 0.6 x 1.000 = 0.600   vs  R = 0.4 x 1.000 = 0.400  -> D


def test_strategy_picks_match_hand_calculation():
    picks = strategy_picks(_df().dropna(subset=["p_market_D"]))
    assert picks["always_favorite"].tolist() == ["D", "R", "D"]
    assert picks["always_underdog"].tolist() == ["R", "D", "R"]
    assert picks["model_favorite"].tolist() == ["D", "R", "D"]
    assert picks["model_ev"].tolist() == ["D", "D", "D"]


# ---------------------------------------------------------------------------
# 4. backtest(): points actually earned, given who really won
# ---------------------------------------------------------------------------
# Winners were A = D, B = D, C = R.
#
#  strategy          A       B       C      2018    2022   TOTAL
#  always_underdog   0     1.737   1.000    1.737   1.000  2.737
#  model_ev        0.515   1.737     0      2.252     0    2.252
#  always_favorite 0.515     0       0      0.515     0    0.515
#  model_favorite  0.515     0       0      0.515     0    0.515
#
# backtest() returns one row per strategy, one column per year plus TOTAL,
# sorted best first. Race D never appears because it has no market price.


def test_backtest_matches_hand_calculation():
    bt = backtest(_df())
    pts = lambda p: -np.log2(p)  # noqa: E731

    assert np.isclose(bt.loc["always_underdog", "TOTAL"], pts(0.3) + pts(0.5))
    assert np.isclose(bt.loc["model_ev", "TOTAL"], pts(0.7) + pts(0.3))
    assert np.isclose(bt.loc["always_favorite", "TOTAL"], pts(0.7))
    assert np.isclose(bt.loc["model_favorite", "TOTAL"], pts(0.7))
    assert bt.index[0] == "always_underdog"  # sorted best first


def test_backtest_year_columns_add_up_to_total():
    bt = backtest(_df())
    years = [c for c in bt.columns if c != "TOTAL"]
    assert years == ["2018", "2022"]
    assert np.allclose(bt[years].sum(axis=1), bt["TOTAL"])


def test_backtest_skips_unpriced_races():
    # Give race D a price: always_underdog would then pick R and lose (D won),
    # so its total is unchanged; always_favorite picks D (0.5 ties go to D) and
    # gains 1 point. That only works if race D was really excluded before.
    priced = _df().assign(p_market_D=[0.7, 0.3, 0.5, 0.5])
    before = backtest(_df())
    after = backtest(priced)
    assert np.isclose(after.loc["always_favorite", "TOTAL"],
                      before.loc["always_favorite", "TOTAL"] + 1.0)


# ---------------------------------------------------------------------------
# 5. expected_totals(): what each strategy should score ON AVERAGE
# ---------------------------------------------------------------------------
# Instead of using the real winners, this assumes the model's probabilities are
# the truth and averages over every possible outcome:
#
#  strategy          expected points
#  model_ev          0.412 + 0.521 + 0.600 = 1.533   <- highest, by design
#  always_favorite   0.412 + 0.360 + 0.600 = 1.372
#  model_favorite    same picks as always_favorite = 1.372
#  always_underdog   0.347 + 0.521 + 0.400 = 1.268
#
# KEY LESSON: in section 4, always_underdog SCORED more (2.737 vs 2.252), yet
# here model_ev is EXPECTED to score more. With 3 races, luck dominates. Judge
# strategies on hundreds of past races, never on a handful.


def test_expected_totals_match_hand_calculation():
    out = expected_totals(_df(), n_sims=200).set_index("strategy")
    assert out.index[0] == "model_ev"  # EV rule maximizes expected points by construction
    assert np.isclose(out.loc["model_ev", "expected"], 1.533, atol=1e-3)
    assert np.isclose(out.loc["always_favorite", "expected"], 1.372, atol=1e-3)
    assert np.isclose(out.loc["always_underdog", "expected"], 1.268, atol=1e-3)


def test_expected_totals_spread_is_ordered():
    # p10 / median / p90 come from simulating outcomes; they must be in order.
    out = expected_totals(_df(), n_sims=500)
    assert (out["p10"] <= out["median"]).all()
    assert (out["median"] <= out["p90"]).all()


# ---------------------------------------------------------------------------
# 6. Command-line runner: prints results and logs to experiments.csv
# ---------------------------------------------------------------------------
# run_backtest() is what `python -m election_prediction.benchmarks <file>` calls.
# It also accepts a 'rating' column instead of p_market_D.


def test_run_backtest_logs_to_experiments_csv(tmp_path):
    from election_prediction.benchmarks import run_backtest

    preds = tmp_path / "preds.csv"
    _df().drop(columns="p_market_D").assign(
        rating=["Lean D", "Lean R", "Toss-up", None]).to_csv(preds, index=False)
    log = tmp_path / "experiments.csv"

    bt = run_backtest(preds, name="test run", log_path=log, notes="hand example", show=False)

    row = pd.read_csv(log).iloc[0]
    assert row["name"] == "test run"
    assert np.isclose(row["backtest_points"], bt.loc["model_ev", "TOTAL"], atol=1e-4)
    assert np.isclose(row["always_underdog_points"], bt.loc["always_underdog", "TOTAL"], atol=1e-4)
    assert "n_races=3" in row["notes"]  # race D has no rating -> skipped
