"""
2026 forecast + submission: all 470 races (435 House + 33 regular Senate + OH/FL specials).

    PYTHONPATH=src python3 -m election_prediction.forecast \
        [--kalshi data/raw/kalshi_2026-11-02.csv]

Steps:
  1. race list = 2026 regular ('G') rows of the incumbency file + the 2 Senate specials
  2. inputs: lean (Cook 2026 post-redistricting lines; Cook Senate PVI), generic ballot,
     incumbency, midterm x president (2026 row of the national files)
  3. baseline (config features) fit on ALL training years -> predicted margin
  4. sigma from leave-one-year-out errors of the training years (per chamber if
     probability.sigma_by_chamber), Student-t -> P(D wins)
  5. races with no R candidate -> D (P = 0.999); no D candidate -> R (P = 0.001)
  6. with --kalshi: blend model and market in log-odds,
        p_blend = logistic(w * logit(p_model) + (1 - w) * logit(p_market)),
     w = probability.model_weight_vs_market (0.25: the market has polls and candidate
     information the baseline lacks), then the expected-points pick rule on p_blend;
     unpriced races: pick the model favourite (flagged no_market)
Writes outputs/predictions_2026.csv (everything) and outputs/submission.csv (validated).
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .data import KEYS, race_id


def race_list_2026(incumbency: pd.DataFrame) -> pd.DataFrame:
    i = incumbency[incumbency["year"] == 2026]
    races = i[(i["chamber"] == "Senate") | ~i["special"]].copy()   # House specials aren't contest races
    races["no_D"] = races["dem_candidate"].isna()
    races["no_R"] = races["rep_candidate"].isna()
    return races.reset_index(drop=True)


def forecast(cfg: dict, kalshi_path=None, verbose: bool = True) -> pd.DataFrame:
    from .baseline import baseline_residuals, describe_baseline, fit_baseline, predict_baseline
    from .config import repo_path
    from .features import add_baseline_inputs
    from .predict import build_submission, validate_submission, write_submission
    from .probability import prob_from_margin, sigma_components
    from .train import build_training_table, load_baseline_inputs
    from .twostage import chamber_sigma

    feats = list(cfg["baseline"]["features"])
    prob = cfg.get("probability", {})
    national, incumbency = load_baseline_inputs(cfg)

    # 3. fit on all training years
    train = build_training_table(cfg, verbose=False, inputs=(national, incumbency))
    train = train.dropna(subset=feats).reset_index(drop=True)
    model = fit_baseline(train, feats)

    # 4. sigma from held-out errors (each training year predicted without itself)
    oof = baseline_residuals(train, feats).rename(columns={"residual": "error"})
    sig = sigma_components(oof)
    by_chamber = chamber_sigma(oof) if prob.get("sigma_by_chamber", False) else {}

    # 1-2. 2026 races and inputs
    races = race_list_2026(incumbency)
    lean = pd.read_csv(repo_path(cfg["data"]["lean"]))
    from .data import standardize_keys
    lean = standardize_keys(lean)
    races = races.merge(lean[KEYS + ["lean"]], on=KEYS, how="left", validate="one_to_one")
    races = add_baseline_inputs(races.drop(columns=["incumbency_signed"]), national, incumbency,
                                cfg["baseline"].get("approval_source", "reliable_30d"))
    missing = races[feats].isna().any(axis=1)
    if missing.any():
        raise ValueError(f"2026 races with a blank input: {race_id(races[missing]).tolist()[:10]}")

    races["pred_margin"] = predict_baseline(model, races, feats)
    races["sigma"] = races["chamber"].map(by_chamber).fillna(sig["sigma_total"])
    races["p_model_D"] = prob_from_margin(races["pred_margin"], races["sigma"],
                                          dist=prob.get("dist", "t"), t_df=prob.get("t_df", 5))
    races.loc[races["no_R"] & ~races["no_D"], "p_model_D"] = 0.999
    races.loc[races["no_D"] & ~races["no_R"], "p_model_D"] = 0.001
    races["only_party"] = np.select([races["no_R"] & ~races["no_D"], races["no_D"] & ~races["no_R"]],
                                    ["D", "R"], default="")
    races["race_id"] = race_id(races)

    # 6. market + picks
    if kalshi_path:
        from .data import standardize_keys as sk
        k = sk(pd.read_csv(kalshi_path))
        k = k[k["year"] == 2026] if "year" in k else k.assign(year=2026)
        k = k.drop_duplicates(KEYS)
    else:
        k = races[KEYS].assign(p_market_D=np.nan)
    w = float(prob.get("model_weight_vs_market", 1.0))
    races = races.merge(k[KEYS + ["p_market_D"]], on=KEYS, how="left", validate="one_to_one")
    races["p_model_raw"] = races["p_model_D"]
    eps = 1e-4
    pm = races["p_model_raw"].clip(eps, 1 - eps)
    pk = races["p_market_D"].clip(eps, 1 - eps)
    blend = 1 / (1 + np.exp(-(w * np.log(pm / (1 - pm)) + (1 - w) * np.log(pk / (1 - pk)))))
    has = races["p_market_D"].notna() & (races["only_party"] == "")
    races.loc[has, "p_model_D"] = blend[has]
    races = races.drop(columns="p_market_D")
    sub = build_submission(races, k, uncontested=races[races["only_party"] != ""][KEYS + ["only_party"]])
    sub = sub.merge(races[KEYS + ["pred_margin", "sigma"]], on=KEYS, how="left")
    validate_submission(sub, expected_races=races, expected_count=cfg["submission"]["expected_count"])

    out = races.merge(sub[KEYS + ["pick", "p_market_D", "no_market"]], on=KEYS, how="left")
    keep = ["race_id", "chamber", "state", "district", "special", "dem_candidate", "rep_candidate",
            "incumbency_signed", "lean", "generic_ballot", "midterm_x_pres", "pred_margin", "sigma",
            "p_model_raw", "p_model_D", "p_market_D", "pick", "no_market", "only_party"]
    out = out[keep].sort_values(["chamber", "state", "district", "special"]).reset_index(drop=True)
    pred_path = repo_path("outputs/predictions_2026.csv")
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(pred_path, index=False)
    sub_path = write_submission(sub, repo_path(cfg["submission"]["path"]))

    if verbose:
        coefs = {k2: round(v, 4) for k2, v in describe_baseline(train, feats).items()}
        print(f"baseline fit on {len(train)} races ({sorted(train['year'].unique())}): {coefs}")
        per = ", ".join(f"{c} {v:.4f}" for c, v in by_chamber.items())
        print(f"sigma: overall {sig['sigma_total']:.4f}" + (f", by chamber {per}" if by_chamber else ""))
        print(f"\n{len(out)} races: {out['chamber'].value_counts().to_dict()}; "
              f"uncontested/same-party {int((out['only_party'] != '').sum())}; "
              f"priced on Kalshi {int((~out['no_market']).sum())}")
        for ch in ("House", "Senate"):
            o = out[out["chamber"] == ch]
            print(f"{ch}: model favours D in {int((o['p_model_D'] > 0.5).sum())}, R in "
                  f"{int((o['p_model_D'] < 0.5).sum())}; expected D seats {o['p_model_D'].sum():.1f}; "
                  f"picks D {int((o['pick'] == 'D').sum())} / R {int((o['pick'] == 'R').sum())}")
        close = out[(out["p_model_raw"] > 0.2) & (out["p_model_raw"] < 0.8)]
        print(f"\n{len(close)} races between 20% and 80%:")
        print(f"(p_model_D = blend: model weight {w} vs market {1 - w}; p_model_raw = model alone)")
        cols = ["race_id", "dem_candidate", "rep_candidate", "incumbency_signed",
                "pred_margin", "p_model_raw", "p_model_D", "p_market_D", "pick"]
        print(close[cols].round(3).to_string(index=False))
        print(f"\nsaved {pred_path} and {sub_path}")
    return out


def main(argv=None):
    from .config import load_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--kalshi", default=None, help="race-level Kalshi CSV from election_prediction.kalshi")
    a = ap.parse_args(argv)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        forecast(load_config(a.config), a.kalshi)


if __name__ == "__main__":
    main()
