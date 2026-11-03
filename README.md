# Election-Predictions-Model

Model Citizens (Stanford Democracy Day) 2026, **Full Congress track** (and Senate track):
predict all 470 races on November 3, 2026 (435 House, 33 regular Senate, the Ohio and
Florida Senate specials). A correct pick scores −log₂(p), where p is Kalshi's probability
for the picked side (floored at 1%); a wrong pick scores 0.

**Submission files:** `outputs/full_congress_track.csv` (470 races) and
`outputs/senate_track.csv` (35 races), in the organizers' template format.

---

## Method in one page

1. **Partisan lean.** Cook PVI for every House district (on each election's own district
   lines, including the 2026 redrawn maps) and every state (Senate), converted to the margin
   scale: lean = PVI × 2 / 100, Democrats positive.
2. **Baseline model:** linear regression of the two-party margin (D − R)/(D + R) on four
   inputs, trained on every contested regular House and Senate race 2010–2024 (3,235 races):

   | Input | Definition | Coefficient |
   | --- | --- | --- |
   | `lean` | Cook PVI × 2 / 100 | 0.878 |
   | `generic_ballot` | (D − R)/(D + R) of the final generic-ballot polling | 0.221 |
   | `incumbency_signed` | +1 Democratic incumbent running, −1 Republican, 0 neither | 0.079 |
   | `midterm_x_pres` | midterm flag × president's party (+1 D, −1 R) | −0.041 |

   Inputs were chosen with `election_prediction.compare`, which scores every combination
   leave-one-year-out (each election year predicted by a model that never saw it). Approval
   rating and an approval × midterm term were tested and dropped (no gain; the latter flipped
   signs). Held-out error: **RMSE 0.089, 95.0% of winners correct** (2018–2024: RMSE 0.075).
3. **Win probabilities.** Margin → P(D wins) with a Student-t distribution (5 df). Its spread
   (sigma) comes from held-out errors, with a shared national part and separate House/Senate
   race parts (House 0.089, Senate 0.097). Calibration was checked on held-out years
   (log loss 0.130; normal, t and logistic calibration were compared, t kept).
4. **Market blend.** Our probability is combined with Kalshi's in log-odds:
   p = logistic(0.25 · logit(model) + 0.75 · logit(Kalshi)). The market carries polls and
   candidate information the baseline lacks; 0.25 was the most robust weight whether the
   market or the model turns out to be right.
5. **Picks.** For each race, pick the side with the higher expected points,
   p(side) × −log₂(Kalshi(side)). Because upsets pay far more, this often picks the
   underdog; it maximizes expected points, not the number of correct picks. One-party races
   (e.g. California D-vs-D runoffs) take that party. Races without a Kalshi market (10 of
   470) take the model's favourite.
6. **`win_probability`** in the submission is the blended probability that the *picked*
   candidate wins.

**Residual model (built, tested, not used).** A second stage predicting the baseline's
misses from ACS demographics, FEC fundraising and each incumbent's past overperformance was
implemented (`train.fit_residual_model`) and evaluated with fully nested leave-one-year-out
validation (`twostage.py`, residuals rebuilt inside each test year). It improved older years
but not 2018–2024, and historical FEC totals include post-election money, so it was left out
(`residual_model.use: false`). Details: `docs/CHANGES_2026-11-02.md`.

---

## Data

| Data | Source | File(s) |
| --- | --- | --- |
| Election results 1998–2024 | FiveThirtyEight `election-results` (GitHub) | `data/raw/results_538_1998_2024.csv` (built by `scripts/build_github_data.py`) |
| Partisan lean | Cook Political Report PVI (House sheets 1997–2026, state PVI) | `data/raw/lean.csv` (built by `scripts/convert_cook_pvi.py`; Cook workbooks not redistributed) |
| Generic ballot, midterm flag, president flag, presidential approval | team-compiled (RealClearPolitics, Pew, Gallup and others) | `data/raw/national/*.xlsx` |
| Incumbency 1998–2026 (who is running, specials) | team-compiled (CAGE, FEC, House/Senate directories) | `data/raw/incumbency/congressional_incumbency_1998_2026.xlsx` |
| Market prices | Kalshi public API, pulled 2026-11-02 | `data/raw/kalshi_2026-11-02.csv` |
| ACS demographics (residual tests only) | Census ACS 5-year API | `scripts/build_acs_features.py` |
| Campaign finance (residual tests only) | FEC bulk "All candidates" files | `scripts/fetch_fec_weball.py`, `scripts/build_candidate_features.py` |

Rules followed throughout: each election uses only information available before it (lean
from earlier presidential results, the latest ACS release, final pre-election polling),
district lines match the race year (state-years whose lines changed mid-decade without a
matching PVI are excluded), special elections are excluded from training, races where a
third candidate took over 20% are excluded. More: `data/README.md`, `data/raw/SOURCES.md`,
`docs/BASELINE_CHANGES.md`.

---

## How to run

Python 3.9+ with the packages below.

```bash
pip install numpy pandas scipy scikit-learn pyyaml openpyxl requests pytest
cd Election-Predictions-Model

python3 -m pytest -q                                              # tests

# 1. (optional) rebuild lean from the Cook workbooks
python3 scripts/convert_cook_pvi.py --xlsx data/raw/cook/Cook_PVI_1997-2026.xlsx \
    --senate-xlsx data/raw/cook/COOK_SENATE_PVI.xlsx --out data/raw/lean.csv

# 2. evaluate the baseline and the input choices (leave-one-year-out)
PYTHONPATH=src python3 -m election_prediction.compare
PYTHONPATH=src python3 -m election_prediction.twostage            # nested test + calibration

# 3. pull Kalshi prices (needs internet), forecast 2026, make the picks
PYTHONPATH=src python3 -m election_prediction.kalshi --out data/raw/kalshi_2026-11-02.csv \
    --races outputs/predictions_2026.csv
PYTHONPATH=src python3 -m election_prediction.forecast --kalshi data/raw/kalshi_2026-11-02.csv

# 4. write the competition files from the templates
python3 scripts/write_competition_files.py \
    --full-template data/raw/templates/full_congress_track_template.xlsx \
    --senate-template data/raw/templates/senate_track_template.xlsx
```

Step 3's forecast writes `outputs/predictions_2026.csv` (every race: candidates, inputs,
predicted margin, sigma, model-only and blended probability, Kalshi price, pick) and
`outputs/submission.csv`. To reproduce the submitted picks exactly, skip the Kalshi pull and
use the committed `data/raw/kalshi_2026-11-02.csv`.

All settings (training years, baseline inputs, probability distribution, House/Senate sigma,
market weight) live in `configs/baseline.yaml`.

---

## Layout

| Path | What it does |
| --- | --- |
| `configs/baseline.yaml` | data paths, years, features, model and probability settings |
| `src/election_prediction/inputs.py` | loads and checks the team spreadsheets |
| `src/election_prediction/data.py` | race keys, validation, results + lean table |
| `src/election_prediction/features.py` | baseline inputs and other feature helpers |
| `src/election_prediction/baseline.py` | stage-1 linear regression |
| `src/election_prediction/train.py` | training table, baseline run, residual model |
| `src/election_prediction/compare.py` | scores input combinations and approval series |
| `src/election_prediction/twostage.py` | leak-free two-stage evaluation, sigma, calibration |
| `src/election_prediction/calibration.py` | calibration comparison and scoring audit |
| `src/election_prediction/forecast.py` | 2026 forecast, market blend, picks, submission |
| `src/election_prediction/kalshi.py` | Kalshi prices (series + per-race markets) |
| `src/election_prediction/probability.py`, `decision.py`, `evaluate.py`, `predict.py` | probabilities, pick rule, scoring, submission checks |
| `scripts/` | data builders and the competition-file writer |
| `docs/` | change log and detailed notes |
| `tests/` | unit tests, including leakage checks |

## Limitations

- The baseline has no polls or candidate-quality inputs; the market blend supplies that
  information. The 0.25 market weight is a judgment call (no historical race-level market
  prices to tune it on).
- Prices are from November 2; the contest scores with organizers' November 1 odds.
- Most picks are underdogs by design, so the realized score has high variance.
