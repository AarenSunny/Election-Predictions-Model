import numpy as np
import pandas as pd

from election_prediction.evaluate import (
    MAX_POINTS,
    brier,
    calibration_table,
    contest_score,
    log_experiment,
    log_loss,
    margin_errors,
    points_if_correct,
    score_picks,
    underdog_check,
)


def test_slide_examples():
    assert round(float(points_if_correct(0.90)), 2) == 0.15
    assert float(points_if_correct(0.50)) == 1.0
    assert float(points_if_correct(0.25)) == 2.0


def test_floor():
    assert round(MAX_POINTS, 2) == 6.64
    assert float(points_if_correct(0.0)) == MAX_POINTS


def test_wrong_pick_zero():
    pts = score_picks(["D", "R"], ["R", "R"], [0.3, 0.3])
    assert pts[0] == 0 and np.isclose(pts[1], -np.log2(0.7))


def test_contest_score_counts():
    df = pd.DataFrame({"pick": ["D", "R", "D"], "winner": ["D", "R", "R"],
                       "p_market_D": [0.2, 0.8, 0.5]})
    s = contest_score(df)
    assert s["n_correct"] == 2 and s["n_upsets_called"] == 2
    assert np.isclose(s["total_points"], 2 * -np.log2(0.2))


def test_margin_errors():
    e = margin_errors([0.1, -0.1], [0.05, 0.05])
    assert np.isclose(e["bias"], -0.05) and e["winner_accuracy"] == 0.5


def test_probability_metrics_and_tables():
    p = np.array([0.1, 0.9, 0.5, 0.02])
    y = np.array([0, 1, 1, 0])
    assert log_loss(p, y) < log_loss(1 - p, y)
    assert brier(p, y) < brier(1 - p, y)
    assert calibration_table(p, y)["n"].sum() == 4
    assert (underdog_check(p, y)["pred_upset_rate"] <= 0.5).all()


def test_log_experiment_replaces_placeholder_then_appends(tmp_path):
    f = tmp_path / "experiments.csv"
    f.write_text("settings, validation scores, and notes\n")
    log_experiment(f, "run1", rmse=0.081, cv="loyo", extra_thing=3)
    log_experiment(f, "run2", rmse=0.079)
    df = pd.read_csv(f)
    assert df["name"].tolist() == ["run1", "run2"]
    assert "extra_thing=3" in df.loc[0, "notes"]
