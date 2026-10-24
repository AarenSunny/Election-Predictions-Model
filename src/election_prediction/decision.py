"""
Pick rule: for each race choose the side with the higher expected points,

    EV(side) = P_model(side) * (-log2 P_market(side))

If the model exactly agrees with the market, the UNDERDOG has the higher EV
(market 90/10: favorite 0.14, underdog 0.33). The favorite is picked only when
the model is MORE confident in it than the market; break_even_underdog_prob()
says how much more.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluate import points_if_correct


def side_evs(p_model_D, p_market_D):
    pm = np.asarray(p_model_D, dtype=float)
    pk = np.asarray(p_market_D, dtype=float)
    return pm * points_if_correct(pk), (1.0 - pm) * points_if_correct(1.0 - pk)


def choose_side(p_model_D, p_market_D):
    ev_D, ev_R = side_evs(p_model_D, p_market_D)
    return np.where(ev_D >= ev_R, "D", "R")


def break_even_underdog_prob(p_market_underdog):
    """Model underdog probability above which the underdog is the better pick."""
    q = np.clip(np.asarray(p_market_underdog, dtype=float), 1e-9, 1 - 1e-9)
    a = points_if_correct(1.0 - q)
    b = points_if_correct(q)
    return a / (a + b)


def add_picks(df: pd.DataFrame, model_col="p_model_D", market_col="p_market_D") -> pd.DataFrame:
    """Adds pick, ev_D, ev_R, pick_ev, edge_D (model - market), pick_is_underdog."""
    out = df.copy()
    ev_D, ev_R = side_evs(out[model_col], out[market_col])
    out["ev_D"], out["ev_R"] = ev_D, ev_R
    out["pick"] = np.where(ev_D >= ev_R, "D", "R")
    out["pick_ev"] = np.maximum(ev_D, ev_R)
    out["edge_D"] = out[model_col] - out[market_col]
    fav_D = out[market_col] >= 0.5
    out["pick_is_underdog"] = np.where(fav_D, out["pick"] == "R", out["pick"] == "D")
    return out


def review_list(df: pd.DataFrame, n: int = 40) -> pd.DataFrame:
    """Races where model and market disagree most: hand-check these first."""
    d = df.copy()
    d["abs_edge"] = d["edge_D"].abs()
    return d.sort_values("abs_edge", ascending=False).head(n)
