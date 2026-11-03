"""
Fill the organizers' templates from outputs/predictions_2026.csv.

    python3 scripts/write_competition_files.py \
        --full-template <full_congress_track_template.xlsx> \
        --senate-template <senate_track_template.xlsx>

Writes outputs/full_congress_track.csv and outputs/senate_track.csv (+ .xlsx copies) with the
template's columns: race_id (e.g. NY-17, WY-AL, OH-SEN), chamber, predicted_winner_party,
candidate_name, win_probability.
  * predicted_winner_party: our pick (D / R; I where the 'D' slot is an independent such as
    a Democratic-backed independent)
  * candidate_name: that party's nominee (if a party has several general-election candidates,
    e.g. California's same-party runoffs, the incumbent if one is running, else the first listed)
  * win_probability: our probability that the predicted winner wins (the blended model/market
    probability the pick was made with). Under the contest's reward rule an underdog pick can
    be the best expected-points choice, so this can be below 0.5.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def template_id(r) -> str:
    if r["chamber"] == "Senate":
        return f"{r['state']}-SEN"
    d = int(r["district"])
    return f"{r['state']}-" + ("AL" if d == 0 else f"{d:02d}")


# names truncated in the team's incumbency file (first letter lost)
NAME_FIXES = {"assamin Ansari": "Yassamin Ansari", "olanda Prince": "Yolanda Prince"}


def clean(name) -> str:
    if not isinstance(name, str):
        return ""
    n = re.sub(r"\s+", " ", re.sub(r"[^\w .,'\-]", "", name)).strip()
    return NAME_FIXES.get(n, n)


def incumbents_2026(incumbency_xlsx: Path) -> dict:
    """{(chamber, state, district): {party: incumbent name}} from the 'Candidate detail' sheet."""
    d = pd.read_excel(incumbency_xlsx, "Candidate detail")
    d = d[(d["Year"] == 2026) & (d["Incumbent"] == 1) & (d["Election"] == "G")]
    out = {}
    for _, r in d.iterrows():
        ch = "Senate" if str(r["Chamber"]).upper().startswith("S") else "House"
        dist = 0 if ch == "Senate" else int(r["Seat"])
        out.setdefault((ch, r["State"], dist), {})[str(r["Slot"])] = clean(r["Candidate"])
    return out


def build(pred: pd.DataFrame, template: pd.DataFrame, inc: dict, actual_party: dict) -> pd.DataFrame:
    p = pred.copy()
    p["race_id_t"] = p.apply(template_id, axis=1)
    rows = {}
    for _, r in p.iterrows():
        party = r["pick"]
        names = [clean(n) for n in str(r["dem_candidate" if party == "D" else "rep_candidate"]).split(";")
                 if clean(n) and clean(n).lower() != "nan"]
        incumbent = inc.get((r["chamber"], r["state"], 0 if r["chamber"] == "Senate" else int(r["district"])),
                            {}).get(party)
        name = next((n for n in names if incumbent and n.split()[-1].upper() in incumbent.upper()), None)
        name = name or (names[0] if names else "")
        prob = r["p_model_D"] if party == "D" else 1 - r["p_model_D"]
        label = actual_party.get((r["chamber"], r["state"], int(r["district"]), party), party)
        rows[r["race_id_t"]] = (label, name, round(float(prob), 4))
    out = template.copy()
    missing = [x for x in out["race_id"] if x not in rows]
    if missing:
        raise ValueError(f"template races with no prediction: {missing[:10]}")
    out["predicted_winner_party"] = [rows[x][0] for x in out["race_id"]]
    out["candidate_name"] = [rows[x][1] for x in out["race_id"]]
    out["win_probability"] = [rows[x][2] for x in out["race_id"]]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-template", required=True, type=Path)
    ap.add_argument("--senate-template", required=True, type=Path)
    ap.add_argument("--predictions", default=ROOT / "outputs" / "predictions_2026.csv", type=Path)
    ap.add_argument("--incumbency", type=Path,
                    default=ROOT / "data" / "raw" / "incumbency" / "congressional_incumbency_1998_2026.xlsx")
    a = ap.parse_args(argv)
    pred = pd.read_csv(a.predictions)
    inc = incumbents_2026(a.incumbency)
    # independents sitting in the 'D' slot of the incumbency file (e.g. Democratic-backed independents)
    actual = {}
    for sheet, ch in (("House", "House"), ("Senate", "Senate")):
        raw = pd.read_excel(a.incumbency, sheet, header=None)
        hdr = raw.index[raw.iloc[:, 0].astype(str).str.strip().str.lower() == "year"][0]
        d = pd.read_excel(a.incumbency, sheet, header=hdr)
        d = d[(d["Year"] == 2026) & (d["Election"] == "G")]
        for _, r in d.iterrows():
            dist = 0 if ch == "Senate" else (0 if str(r["State"]) in {"AK", "DE", "ND", "SD", "VT", "WY"}
                                             else int(r["District"]))
            if str(r.get("D actual party", "D")).strip().upper() in ("I", "IND", "INDEPENDENT"):
                actual[(ch, r["State"], dist, "D")] = "I"
    out_dir = ROOT / "outputs"
    for tpl, name in ((a.full_template, "full_congress_track"), (a.senate_template, "senate_track")):
        t = pd.read_excel(tpl)
        out = build(pred, t, inc, actual)
        out.to_csv(out_dir / f"{name}.csv", index=False)
        out.to_excel(out_dir / f"{name}.xlsx", index=False)
        print(f"{name}: {len(out)} races, picks {out['predicted_winner_party'].value_counts().to_dict()}, "
              f"blank names {int((out['candidate_name'] == '').sum())} -> {out_dir / (name + '.csv')}")


if __name__ == "__main__":
    main()
