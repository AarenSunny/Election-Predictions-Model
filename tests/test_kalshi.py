"""Parsing uses market records shaped like real Kalshi API responses (Oct 2026)."""

import numpy as np
import pandas as pd

from election_prediction.kalshi import (
    market_party,
    market_price,
    markets_to_table,
    parse_event_ticker,
    race_probabilities,
)


def _m(ticker, subtitle, bid, ask, last, volume="5000.00"):
    return {"ticker": ticker, "event_ticker": ticker.rsplit("-", 1)[0], "subtitle": subtitle,
            "yes_sub_title": ticker[-4:], "yes_bid_dollars": bid, "yes_ask_dollars": ask,
            "last_price_dollars": last, "volume_fp": volume, "status": "active"}


def test_parse_event_ticker():
    assert parse_event_ticker("KXHOUSERACE-NY17-26", "House") == {
        "year": 2026, "chamber": "House", "state": "NY", "district": 17}
    assert parse_event_ticker("KXHOUSERACE-WYAL-26", "House")["district"] == 0
    assert parse_event_ticker("garbage", "House") is None


def test_party_and_price():
    assert market_party(_m("KXHOUSERACE-WA05-26-R", "Republican party:: Republican party",
                           "0.85", "0.86", "0.85")) == "R"
    assert market_party(_m("KXHOUSERACE-MN05-26-DJAC", ":: Independent", "0", "0.01", "0.01")) == "O"
    assert np.isclose(market_price({"yes_bid_dollars": "0.85", "yes_ask_dollars": "0.86",
                                    "last_price_dollars": "0.80"}), 0.855)
    # wide spread -> last trade
    assert market_price({"yes_bid_dollars": "0.10", "yes_ask_dollars": "0.40",
                         "last_price_dollars": "0.30"}) == 0.30


def test_race_probabilities_two_party_and_flags():
    mk = [
        _m("KXHOUSERACE-WA05-26-R", "Republican party:: Republican party", "0.85", "0.86", "0.85"),
        _m("KXHOUSERACE-WA05-26-D", "Democratic party:: Democratic party", "0.13", "0.15", "0.14"),
        _m("KXHOUSERACE-WA05-26-AIND", ":: Independent", "0.00", "0.01", "0.01"),
        _m("KXHOUSERACE-WYAL-26-R", "Republican party:: Republican party", "0.955", "0.956", "0.955",
           volume="10.00"),
    ]
    races = race_probabilities(markets_to_table(mk, "House")).set_index("state")
    assert np.isclose(races.loc["WA", "p_market_D"], 0.14 / (0.14 + 0.855))
    assert races.loc["WA", "n_candidates"] == 3
    assert races.loc["WY", "p_market_D"] == 0.0 and races.loc["WY", "no_D_market"]
    assert races.loc["WY", "low_volume"]
    assert isinstance(races, pd.DataFrame)
