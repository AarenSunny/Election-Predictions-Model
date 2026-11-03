# Election-Predictions-Model

Our entry for Model Citizens 2026 at Stanford Democracy Day, by Aaren Sun and Amun Reddy.

The contest asks for a pick in every House and Senate race on November 3, 2026. That's 470
races in the Full Congress track (435 House seats, 33 regular Senate seats, and the Ohio and
Florida Senate specials) and 35 in the Senate track. A correct pick scores −log₂(p), where p
is Kalshi's price for the side you picked, with a 1% floor. A wrong pick scores 0.

Our submitted picks are in `outputs/full_congress_track.csv` and `outputs/senate_track.csv`.
The slides are in `docs/Model_Citizens_2026_Sun_Reddy.pptx`.

## How it works

We built the forecast in five steps.

1. Lean. Every House district and every state gets a partisan lean from the Cook PVI, using
   the district lines in place for that election (including the redrawn 2026 maps). We put it
   on the same scale as the vote margin: lean = PVI × 2 / 100, with Democrats positive.

2. Predicted margin. A linear regression predicts the two-party margin, (D − R) / (D + R),
   from four inputs. We trained it on 3,235 contested regular House and Senate races from
   2010 to 2024 (exclusions are listed under Data).

   | Input | What it is | Coefficient |
   | --- | --- | --- |
   | `lean` | Cook PVI × 2 / 100 | 0.878 |
   | `generic_ballot` | final generic-ballot polling margin, (D − R) / (D + R) | 0.221 |
   | `incumbency_signed` | +1 if a Democratic incumbent is running, −1 if a Republican is, 0 otherwise | 0.079 |
   | `midterm_x_pres` | midterm flag × president's party (+1 D, −1 R) | −0.041 |

   In plain terms, a sitting member is worth about 8 points of margin, and in a midterm the
   president's party loses about 4.

   We picked these inputs with `election_prediction.compare`, which tries every combination
   and tests each one year at a time: train on every other year, then predict the held-out
   year blind. Presidential approval, and approval times the midterm flag, didn't help, so
   they're out. On held-out years the typical miss on the margin is 8.9 points
   (RMSE 0.089), or 7.5 points since 2018, and it calls 95% of winners.

3. Win probability. We turn each margin into a chance of winning with a Student-t curve
   (5 degrees of freedom). The width comes from the model's own held-out errors, split into
   a national part shared by every race and a race-level part that's a bit different for the
   House (0.089) and the Senate (0.097). We compared normal, t and logistic curves on the
   held-out years and kept t.

4. Blend with the market. Our model knows nothing about individual polls or candidates, and
   Kalshi traders do. So we average the two in log-odds, weighting our model 25% and the
   market 75%:

   `p = logistic(0.25 · logit(model) + 0.75 · logit(Kalshi))`

   We chose 0.25 because it held up best whether the market or our model turns out to be
   the better one.

5. Pick. For each race we take the side with more expected points, p(side) × −log₂(price of
   that side). Upsets pay a lot (a 10% underdog is worth 3.3 points, a 90% favorite 0.15), so
   this usually means picking the underdog. That's deliberate. We're maximizing expected
   points, not the number of races we get right. One-party races, like California's
   same-party runoffs, go to that party, and the 10 races without a Kalshi market go to our
   model's favorite.

The `win_probability` column in the submission is our blended chance that the candidate we
picked wins.

## What we tried and didn't use

We also built a second-stage model to predict where the baseline misses, using Census ACS
demographics, FEC fundraising and each incumbent's past overperformance. The code is
`train.fit_residual_model`, and `twostage.py` tests it with fully nested leave-one-year-out
validation (residuals are rebuilt inside each test year so nothing leaks). On 2018 to 2024
it barely moved the results, and even that small gain probably came from a leak:
historical FEC totals include money raised after Election Day. So it's switched off
(`residual_model.use: false`). There's more in `docs/CHANGES_2026-11-02.md`.

## Data

| Data | Where it came from | File |
| --- | --- | --- |
| Election results, 1998 to 2024 | FiveThirtyEight `election-results` on GitHub | `data/raw/results_538_1998_2024.csv`, built by `scripts/build_github_data.py` |
| Partisan lean | Cook Political Report PVI (House 1997 to 2026, plus state PVI) | `data/raw/lean.csv`, built by `scripts/convert_cook_pvi.py` (we don't redistribute Cook's workbooks) |
| Generic ballot, midterm flag, president's party, approval | compiled by us from RealClearPolitics, Pew, Gallup and others | `data/raw/national/*.xlsx` |
| Incumbency and candidates, 1998 to 2026 | compiled by us from CAGE, the FEC and the House and Senate directories | `data/raw/incumbency/congressional_incumbency_1998_2026.xlsx` |
| Market prices | Kalshi public API, pulled November 2, 2026 | `data/raw/kalshi_2026-11-02.csv` |
| ACS demographics (residual tests only) | Census ACS 5-year API | `scripts/build_acs_features.py` |
| Campaign finance (residual tests only) | FEC bulk "all candidates" files | `scripts/fetch_fec_weball.py`, `scripts/build_candidate_features.py` |

A few rules we stuck to:
- Each election only sees information from before it (lean from earlier presidential
  results, the latest ACS release, final pre-election polls).
- District lines match the race year. State-years whose lines changed mid-decade without a
  matching PVI are left out.
- Special elections aren't used for training, and neither are races where a third candidate
  got more than 20%.

`data/README.md`, `data/raw/SOURCES.md` and `docs/BASELINE_CHANGES.md` have the details.

## Running it

You need Python 3.9 or newer.

```bash
pip install numpy pandas scipy scikit-learn pyyaml openpyxl requests pytest
cd Election-Predictions-Model

python3 -m pytest -q        # run the tests

# optional: rebuild lean from the Cook workbooks
python3 scripts/convert_cook_pvi.py --xlsx data/raw/cook/Cook_PVI_1997-2026.xlsx \
    --senate-xlsx data/raw/cook/COOK_SENATE_PVI.xlsx --out data/raw/lean.csv

# check the baseline and the input choices (leave-one-year-out)
PYTHONPATH=src python3 -m election_prediction.compare
PYTHONPATH=src python3 -m election_prediction.twostage    # nested test and calibration

# pull Kalshi prices (needs internet), forecast 2026 and make the picks
PYTHONPATH=src python3 -m election_prediction.kalshi --out data/raw/kalshi_2026-11-02.csv \
    --races outputs/predictions_2026.csv
PYTHONPATH=src python3 -m election_prediction.forecast --kalshi data/raw/kalshi_2026-11-02.csv

# fill in the contest templates
python3 scripts/write_competition_files.py \
    --full-template data/raw/templates/full_congress_track_template.xlsx \
    --senate-template data/raw/templates/senate_track_template.xlsx
```

The forecast step writes `outputs/predictions_2026.csv`, which has every race with its
candidates, inputs, predicted margin, sigma, model-only and blended probability, Kalshi price
and pick. It also writes `outputs/submission.csv`. To get exactly our submitted picks, skip
the Kalshi pull and use the `data/raw/kalshi_2026-11-02.csv` that's already in the repo.

Every setting (training years, inputs, probability curve, House and Senate sigma, market
weight) is in `configs/baseline.yaml`.

## Where things are

| Path | What's in it |
| --- | --- |
| `configs/baseline.yaml` | data paths, years, inputs, model and probability settings |
| `src/election_prediction/inputs.py` | loads and checks our spreadsheets |
| `src/election_prediction/data.py` | race keys, validation, the results and lean table |
| `src/election_prediction/features.py` | baseline inputs and other feature helpers |
| `src/election_prediction/baseline.py` | the linear regression |
| `src/election_prediction/train.py` | training table, baseline run, residual model |
| `src/election_prediction/compare.py` | scores input combinations and approval series |
| `src/election_prediction/twostage.py` | leak-free two-stage test, sigma, calibration |
| `src/election_prediction/calibration.py` | calibration comparison and scoring audit |
| `src/election_prediction/forecast.py` | 2026 forecast, market blend, picks |
| `src/election_prediction/kalshi.py` | Kalshi prices (series and per-race markets) |
| `src/election_prediction/probability.py`, `decision.py`, `evaluate.py`, `predict.py` | probabilities, pick rule, scoring, submission checks |
| `scripts/` | data builders and the template writer |
| `docs/` | change logs, notes and our slides |
| `tests/` | unit tests, including checks for leakage |

## Limitations

- The baseline has no polls and no sense of candidate quality. The market blend is how that
  information gets in, and the 25% weight is our judgment call, since there's no history of
  race-level Kalshi prices to tune it on.
- Our prices are from November 2. The contest scores with the organizers' November 1 odds.
- Races aren't independent. One national surprise moves all of them at once.
- Mid-decade redistricting in several states makes 2026 lean less tested than usual.
- We picked a lot of underdogs on purpose, so our score will swing a lot. A simulation of
  election night puts our expected Full Congress score around 80 to 90 points, with only
  about 60 picks correct, against about 41 for picking every favorite.

## Thanks

Special thanks to the Cook Political Report for giving us free access to their Partisan
Voting Index data. The district and state lean at the heart of this model comes from them.
