"""
Fetch Kalshi market prices and turn them into P(D wins) per race.

Public market data needs no login. Format checked Oct 2026:
  * House series:   KXHOUSERACE
  * event_ticker:   KXHOUSERACE-NY17-26     (state + district or AL, then 2-digit year)
  * market ticker:  KXHOUSERACE-NY17-26-D   one market PER CANDIDATE; major parties
                    end in -D / -R and have subtitles like "Democratic party:: ...".
                    Independents get initials (-DJAC) and ":: Independent".
  * prices:         yes_bid_dollars / yes_ask_dollars / last_price_dollars (strings, $0-1)
  * Senate series:  NOT "KXSENATERACE" (empty). Open any Senate race on kalshi.com,
                    read the series prefix from its URL/ticker, and set it in the config.

Run from the repo root (needs internet; run on your own computer):
    python -m election_prediction.kalshi --out data/raw/kalshi_2026-11-01.csv
    python -m election_prediction.kalshi --senate-series <SERIES> --out ...
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import time

import numpy as np
import pandas as pd

BASE_URL = "https://external-api.kalshi.com/trade-api/v2"
HOUSE_SERIES = "KXHOUSERACE"

_EVENT_RE = re.compile(r"^(?P<series>[A-Z0-9]+)-(?P<st>[A-Z]{2})(?P<dist>AL|\d{1,2})?-(?P<yy>\d{2})")


# ------------------------------------------------------------------ fetch
def fetch_markets(series_ticker: str, status: str | None = "open", limit: int = 1000,
                  session=None, pause: float = 0.2, max_pages: int = 100) -> list[dict]:
    """All markets in a series, following the pagination cursor."""
    import requests  # imported here so the rest of the module works offline

    s = session or requests.Session()
    params = {"series_ticker": series_ticker, "limit": limit}
    if status:
        params["status"] = status
    out, cursor = [], None
    for _ in range(max_pages):
        if cursor:
            params["cursor"] = cursor
        r = s.get(f"{BASE_URL}/markets", params=params, timeout=30)
        r.raise_for_status()
        js = r.json()
        out.extend(js.get("markets", []))
        cursor = js.get("cursor")
        if not cursor or not js.get("markets"):
            break
        time.sleep(pause)
    return out


# ------------------------------------------------------------------ parse
def parse_event_ticker(event_ticker: str, chamber: str) -> dict | None:
    """'KXHOUSERACE-NY17-26' -> {year: 2026, chamber, state: 'NY', district: 17}."""
    m = _EVENT_RE.match(str(event_ticker))
    if not m:
        return None
    dist = m.group("dist")
    district = 0 if (dist in (None, "AL") or chamber == "Senate") else int(dist)
    return {"year": 2000 + int(m.group("yy")), "chamber": chamber,
            "state": m.group("st"), "district": district}


def market_party(m: dict) -> str:
    """D / R from the ticker suffix or subtitle; otherwise O."""
    suffix = str(m.get("ticker", "")).rsplit("-", 1)[-1]
    if suffix in ("D", "R"):
        return suffix
    sub = str(m.get("subtitle", "")).lower()
    if sub.startswith("democratic"):
        return "D"
    if sub.startswith("republican"):
        return "R"
    return "O"


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def market_price(m: dict, max_spread: float = 0.10) -> float:
    """
    Probability from one market: bid/ask midpoint when both sides exist and the
    spread is reasonable, otherwise the last trade price. CONFIRM with the
    organizers which price they score with and change this to match.
    """
    bid, ask = _f(m.get("yes_bid_dollars")), _f(m.get("yes_ask_dollars"))
    last = _f(m.get("last_price_dollars"))
    if bid > 0 and ask > 0 and (ask - bid) <= max_spread:
        return (bid + ask) / 2
    return last if not np.isnan(last) else (ask if ask > 0 else np.nan)


def markets_to_table(markets: list[dict], chamber: str) -> pd.DataFrame:
    """One row per candidate market with race keys, party, prices, volume."""
    rows = []
    for m in markets:
        keys = parse_event_ticker(m.get("event_ticker", ""), chamber)
        if keys is None:
            continue
        rows.append({**keys, "special": False,
                     "event_ticker": m.get("event_ticker"), "ticker": m.get("ticker"),
                     "candidate": m.get("yes_sub_title"), "party": market_party(m),
                     "price": market_price(m),
                     "yes_bid": _f(m.get("yes_bid_dollars")), "yes_ask": _f(m.get("yes_ask_dollars")),
                     "last": _f(m.get("last_price_dollars")), "volume": _f(m.get("volume_fp")),
                     "status": m.get("status")})
    return pd.DataFrame(rows)


def race_probabilities(cands: pd.DataFrame, method: str = "two_party",
                       min_volume: float = 1000.0) -> pd.DataFrame:
    """
    Collapse candidate markets to P(D wins) per race.
      method="two_party":       p_D / (p_D + p_R)   (ignores independents)
      method="all_candidates":  p_D / sum of every candidate's price
    Flags: no_D_market / no_R_market (e.g. same-party top-two, uncontested),
    low_volume (thin market, less reliable).
    """
    keys = ["year", "chamber", "state", "district", "special"]
    out = []
    for k, g in cands.groupby(keys):
        pD = g.loc[g["party"] == "D", "price"].max()
        pR = g.loc[g["party"] == "R", "price"].max()
        total = g["price"].sum()
        if method == "two_party":
            denom = np.nansum([pD, pR])
        elif method == "all_candidates":
            denom = total
        else:
            raise ValueError("method must be 'two_party' or 'all_candidates'")
        p = pD / denom if (denom and not np.isnan(pD)) else (0.0 if not np.isnan(pR) else np.nan)
        rec = dict(zip(keys, k, strict=False))
        rec.update(p_market_D=p, price_D=pD, price_R=pR, price_sum=total,
                   n_candidates=len(g), no_D_market=bool(np.isnan(pD)),
                   no_R_market=bool(np.isnan(pR)),
                   low_volume=bool(g.loc[g["party"].isin(["D", "R"]), "volume"].sum() < min_volume))
        out.append(rec)
    return pd.DataFrame(out)


def snapshot(house_series: str = HOUSE_SERIES, senate_series: str | None = None,
             method: str = "two_party") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch now and return (candidate_table, race_table), both timestamped."""
    frames = [markets_to_table(fetch_markets(house_series), "House")]
    if senate_series:
        frames.append(markets_to_table(fetch_markets(senate_series), "Senate"))
    cands = pd.concat(frames, ignore_index=True)
    stamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    cands["pulled_at_utc"] = stamp
    races = race_probabilities(cands, method=method)
    races["pulled_at_utc"] = stamp
    return cands, races


def main(argv=None):
    ap = argparse.ArgumentParser(description="Pull a Kalshi snapshot.")
    ap.add_argument("--out", required=True, help="race-level CSV, e.g. data/raw/kalshi_2026-11-01.csv")
    ap.add_argument("--house-series", default=HOUSE_SERIES)
    ap.add_argument("--senate-series", default=None)
    ap.add_argument("--method", default="two_party", choices=["two_party", "all_candidates"])
    a = ap.parse_args(argv)
    cands, races = snapshot(a.house_series, a.senate_series, a.method)
    races.to_csv(a.out, index=False)
    cands.to_csv(a.out.replace(".csv", "_candidates.csv"), index=False)
    print(f"{len(races)} races, {len(cands)} candidate markets -> {a.out}")
    for flag in ("no_D_market", "no_R_market", "low_volume"):
        print(f"  {flag}: {int(races[flag].sum())}")


if __name__ == "__main__":
    main()
