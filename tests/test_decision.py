import numpy as np
import pandas as pd

from election_prediction.decision import (
    add_picks,
    break_even_underdog_prob,
    choose_side,
    review_list,
    side_evs,
)


def test_agreeing_with_market_picks_underdog():
    for p in [0.6, 0.75, 0.9, 0.99]:
        assert choose_side(p, p)[()] == "R"
        assert choose_side(1 - p, 1 - p)[()] == "D"


def test_confident_model_picks_favorite():
    assert choose_side(0.99, 0.90)[()] == "D"


def test_break_even_is_indifference_point():
    q = float(break_even_underdog_prob(0.10))
    ev_D, ev_R = side_evs(1 - q, 0.90)
    assert np.isclose(ev_D, ev_R) and 0 < q < 0.10


def test_add_picks_and_review_list():
    df = pd.DataFrame({"p_model_D": [0.7, 0.2, 0.5], "p_market_D": [0.6, 0.3, 0.9]})
    out = add_picks(df)
    assert np.allclose(out["pick_ev"], np.maximum(out["ev_D"], out["ev_R"]))
    assert review_list(out, n=1)["p_market_D"].iloc[0] == 0.9   # biggest disagreement
