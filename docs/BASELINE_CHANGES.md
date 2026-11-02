# Baseline model v1: changes and how to use them

Date: 2026-11-02. This covers everything changed to plug the team's final baseline data
(Cook House + Senate PVI, generic ballot, midterm flag, president flag, presidential
approval, incumbency) into the stage-1 baseline.

**Short version.** All six inputs are loaded, cleaned and built. The config still trains
on `lean` only, as asked, and keeps your 2010–2024 training years. A new command scores every combination of inputs and every
approval series on held-out years, so you can decide what to add:

```bash
PYTHONPATH=src python3 -m election_prediction.compare --config configs/baseline.yaml   # compare inputs
PYTHONPATH=src python3 -m election_prediction.train   --config configs/baseline.yaml   # run the baseline
PYTHONPATH=src python3 -m election_prediction.inputs  --config configs/baseline.yaml   # check the input files
```

To use more inputs, edit `baseline.features` in `configs/baseline.yaml` (e.g. add
`generic_ballot`), then rerun `train`.

---

## 1. The scale everything is on

Target (what the model predicts): **two-party margin = (D − R) / (D + R)**.
−1 = 100% R, +1 = 100% D, 0 = 50/50, 0.5 = 75% D / 25% R. Each 0.01 = half a point of vote share.

| Input column | Formula | Range | Example |
| --- | --- | --- | --- |
| `lean` | Cook **raw** PVI × 2 / 100 (House sheets; Senate state PVI) | about −0.9 … +0.9 | Cook D+5.00 → 0.10 |
| `generic_ballot` | (Dem − Rep) / (Dem + Rep) from the generic-ballot poll | −1 … 1 | 2026: (52 − 43.1) / 95.1 = 0.094 |
| `incumbency_signed` | D flag − R flag from the incumbency file | −1, 0, +1 | D incumbent running → +1 |
| `approval_x_pres` | (approval − 50) × president flag | −50 … 50 (percent points) | Bush 2008, 26.5%: (26.5 − 50) × −1 = +23.5 |
| `midterm_x_pres` | midterm flag × president flag | −1, 0, +1 | 2018 (midterm, R president): −1 |
| `approval_x_pres_midterm` | `approval_x_pres` × midterm flag | −50 … 50 | 0 in presidential years |

Why the products are built by hand: linear regression only **adds** columns, it never
multiplies them. Approval only means something once you know which party the president
belongs to, and the same goes for the midterm penalty. Signs don't matter: the regression
learns each coefficient's sign itself (a minus in front only flips the learned sign).

Why lean is × 2: Cook PVI is in vote-share points; the target is a margin (twice as
big). With × 2 the lean coefficient reads naturally (~1 = "a seat 10 points bluer than
the country votes ~10 points bluer"). Leaving it out would give identical predictions
with a coefficient ~2× larger, as long as House and Senate use the same scale (they do:
both Cook now).

Approval uses (approval − 50) **without** dividing by 50, as specified. Dividing would
only rescale the coefficient; predictions are identical.

---

## 2. Files

### New
| File | What it does |
| --- | --- |
| `src/election_prediction/inputs.py` | Loaders for the team spreadsheets: `load_generic_ballot`, `load_flag`, `load_approval` (both sheets), `approval_wide`, `build_national`, `load_incumbency`. CLI prints/writes the cleaned tables. |
| `src/election_prediction/compare.py` | Scores feature sets and approval series with leave-one-year-out on identical races. |
| `tests/test_inputs.py` | 8 tests on toy workbooks that copy every quirk of the real files. |
| `docs/BASELINE_CHANGES.md` | This file. |
| `data/raw/national/` | `Generic_Ballot_Data.xlsx`, `MidtermFlag.xlsx`, `President_Flag.xlsx`, `Presidential_Approval_Ratings.xlsx` (your files, unchanged; approval file renamed from `Presidental__…` to fix the spelling) |
| `data/raw/incumbency/congressional_incumbency_1998_2026.xlsx` | your file, unchanged |
| `data/raw/cook/COOK_SENATE_PVI.xlsx` | your file, unchanged (gitignored with the rest of `data/raw/cook/`, since it's Cook's data) |

### Changed
| File | Change |
| --- | --- |
| `src/election_prediction/features.py` | + `approval_x_pres`, `approval_x_pres_midterm`, `midterm_x_pres`, `add_baseline_inputs`, `BASELINE_INPUTS`. Older helpers (`approval_signed`, `midterm_pres_penalty`) left in place but unused by the baseline. |
| `src/election_prediction/data.py` | `build_baseline_table(..., drop_specials=True)`: special elections are dropped from training (new `special` column in the drop report). |
| `src/election_prediction/train.py` | `load_baseline_inputs(cfg)`; `build_training_table` now adds every baseline input; `run` drops rows with a blank input (none today), prints approval series + incumbency match rate, logs the approval series. |
| `src/election_prediction/baseline.py` | Docstring lists the new inputs; default `BASELINE_FEATURES = ["lean"]`. |
| `scripts/convert_cook_pvi.py` | + `--senate-xlsx` / `build_senate_lean`: Senate lean from Cook's state PVI. |
| `configs/baseline.yaml` | Your edits kept (2010–2024 training years, run name). New data paths; `baseline.approval_source: reliable_30d`; `baseline.drop_specials: true`; candidate features listed as comments; `national.csv` placeholder removed. |
| `data/raw/lean.csv` | Senate rows rebuilt from Cook Senate PVI (House rows unchanged). |
| `tests/test_convert_cook_pvi.py`, `tests/test_baseline.py`, `tests/test_config.py` | + Senate converter test, special-drop test, comparison-scoring test, config keys. |
| `data/README.md`, `data/raw/SOURCES.md` | Document the new files. |

All 80 tests pass; `ruff` is clean.

---

## 3. How each file is read (format quirks handled)

Years: every loader runs years through `parse_year`, which accepts `2026`, `2026.0`,
`"2026"`, `" 2026 "`, dates and Timestamps. Each loader errors if a year is unreadable or
duplicated, so a bad file fails loudly instead of joining wrong.

**Generic ballot.** Two-party margin from the Dem and Rep columns (undecided dropped, the
same way race results drop third parties). Margins like `D+8.9`, `R+1.7`, `Tie` are parsed.
2002 has only `R+1.7`: it's divided by the average Dem + Rep share of the other years
(≈ 92%), giving −0.019, and flagged `gb_imputed`. Known caveats kept as given: 2000 is an
early-September poll; 2016's window runs to 11/7.

**Midterm flag / President flag.** Read as (Year, Number). Values must be 0/1 and −1/+1. The
midterm flag is also checked against the calendar (midterm = year % 4 == 2); both files pass.

**Approval** (`Presidential_Approval_Ratings.xlsx`). Sheets found by position: first =
"broad" ("Unreliable data but more"), second = "reliable".
- Broad: columns `last_gallup_*`, `avg_all_polls_30d_*`, `avg_all_polls_90d_*`. The 2026
  latest-poll value is stored as a fraction (0.29 / 0.67); anything ≤ 1 is × 100.
- Reliable: text like `66 / 30` is split into approve / disapprove; the blank spacer row
  before 2026 is dropped; each `Records` column is read as the poll count of the column to its left.
- Result: 6 series, `approval_{broad|reliable}_{latest|30d|90d}`. If a chosen window is
  blank for a year, it falls back within the same sheet (30d → 90d → latest; 90d → 30d → latest;
  latest → 30d → 90d). No year is blank today, so no fallback is used.
- Values outside 5–95% raise an error.

**Incumbency.** The header row is found by searching for `Year` (the sheets start with title
rows). `D`/`R` = 1 if that party's candidate is a sitting member of that chamber.
`incumbency_signed = D − R`, with a blank slot (no candidate) counted as 0. House at-large
seats (numbered 1 in the file) become district 0, like everywhere else. Senate rows key on
state + G/S (seat class is kept as a column).

**Special elections (your rule).** `S` rows are special elections. Training drops special
**races** (`drop_specials: true`; 1–10 per year). People who **won** a special are already
marked 1 in their later regular (`G`) races in your file, so their incumbency still counts.
The 2026 OH and FL Senate specials stay available for prediction (both have an appointed R
incumbent, −1).

**Cook Senate PVI.** One block of 50 states + D.C. per year (1997, 2001, …, 2025). A race in
year Y uses the latest block published before Y: 1997 → 1998/2000, 2001 → 2002/2004, …,
2025 → 2026. D.C. is dropped and `Nebraska †` cleaned. Raw (unrounded) column, × 2 / 100.
Compared with the presidential-results version it replaces: average difference under 0.002,
largest 0.068 (Hawaii 2010–12).

**Match rates.** Every year has generic ballot, both flags and all six approval series.
Incumbency matched **every** training race: 5,563 / 5,563 for 1998–2024, 3,235 / 3,235 for 2010–2024.

---

## 4. Results (leave-one-year-out)

Each year is predicted by a model fit on the other years. Same races in every row.
`RMSE 2018+` = error on 2018–2024 only (the years most like 2026).

Your config trains on **2010–2024** (your edit, kept). Results for that setup come first;
1998–2024 is below for reference.

### 4a. Training years 2010–2024 (current config), 3,235 races

| Inputs | RMSE | RMSE 2018+ | Winner acc. | Coefficients | Signs |
| --- | --- | --- | --- | --- | --- |
| lean | 0.1082 | 0.0843 | 92.1% | lean 1.05 | ok |
| lean + generic_ballot | 0.1030 | 0.0771 | 93.0% | gb 0.58 | ok |
| lean + generic + incumbency | 0.0909 | 0.0777 | 94.7% | inc 0.078 | ok |
| **lean + generic + incumbency + midterm_x_pres** | **0.0891** | **0.0753** | **95.0%** | lean 0.878, gb 0.221, inc 0.079, midterm −0.041 | **all ok** |
| lean + generic + incumbency + approval_x_pres | 0.0917 | 0.0780 | 94.4% | approval 0.0019 | ok |
| all six inputs | 0.0898 | 0.0755 | 95.2% | gb −0.34, approval×midterm −0.017 | **2 wrong** |

Per-year RMSE for the best row: 2010 0.107, 2012 0.102, 2014 0.095, 2016 0.099,
2018 0.077, 2020 0.077, 2022 0.076, 2024 0.072.

Approval series (lean + generic + incumbency + approval): all six within 0.002 RMSE.
`broad_latest` (0.0909) and `reliable_latest` / `reliable_30d` (0.0917) lead, `reliable_90d`
last (0.0929). With lean + approval only, `reliable_latest` / `reliable_90d` lead. No series
wins consistently, so the differences are noise.

### 4b. Training years 1998–2024 (for reference), 5,563 races

| Inputs | RMSE | RMSE 2018+ | Winner acc. |
| --- | --- | --- | --- |
| lean | 0.1616 | 0.0841 | 87.9% |
| + generic_ballot | 0.1560 | 0.0785 | 88.7% |
| + incumbency_signed | 0.1205 | 0.1008 | 93.9% |
| + approval_x_pres | 0.1203 | 0.1000 | 93.7% |
| + midterm_x_pres | 0.1208 | 0.1009 | 93.7% |
| + approval_x_pres_midterm | 0.1211 | 0.1013 | 93.9% |

---

## 5. What the numbers say (read before adding inputs)

1. **Recommended baseline (2010–2024): lean + generic_ballot + incumbency_signed +
   midterm_x_pres.** Lowest error overall and on 2018+, and every coefficient has the
   expected sign. To use it:
   ```yaml
   baseline:
     features: [lean, generic_ballot, incumbency_signed, midterm_x_pres]
   ```
   and remove `incumbency_signed` from `residual_model.features` (validate_config stops you
   otherwise).

2. **Approval doesn't earn a place.** With generic ballot and incumbency in, it makes things
   slightly worse (0.0909 → 0.0917). Generic ballot already carries the president's
   popularity. Its sign is right when used alone, so the data isn't broken; it's just redundant.

3. **All six inputs together overfit.** The error looks fine, but generic ballot flips to a
   negative coefficient. With only 8 training years, the four national inputs (generic ballot,
   approval, midterm, approval×midterm) are each learned from 8 numbers and start cancelling
   each other. Don't use it.

4. **Your 2010+ choice is the right one for incumbency.** An incumbent was worth about 0.23
   margin (≈ 11 vote-share points) in 1998–2004, about 0.11 in 2010–2016 and about 0.045
   (≈ 2 points) since 2018. Trained on 1998+, incumbency's coefficient (0.146) is ~3× too big
   for recent years and makes 2018+ worse (0.0785 → 0.1008). Trained on 2010+, it's 0.078
   and 2018+ is unchanged. It's still a bit larger than the recent effect; the residual
   model or an "incumbency × year" column could pick up the rest.

5. **midterm × president needs the 2010+ window.** On 1998+ it gets the wrong sign and
   doesn't help; on 2010+ it helps with the right sign (president's party loses about 2
   points in midterms beyond what the generic ballot shows). It rests on only 4 midterms,
   so treat it as useful but fragile.

6. **Approval series:** keep `reliable_30d`. Nothing beats it consistently.

## 6. Checks done

- Every input recomputed by hand for 8 known races (e.g. 2008 MS-04 Gene Taylor: lean −0.346,
  incumbency +1, approval_x_pres +23.5; 2014 ME Collins: −1; 2002 MN Mondale/Coleman: 0,
  since Wellstone died before the election). All match.
- Held-out-year RMSE recomputed independently with plain numpy least squares (1998–2024,
  lean + generic + incumbency): 0.1205 / 93.89%, identical to the pipeline.
- Margins recomputed from raw vote counts: exact.
- No special race in the training table; only the configured years; no race dropped for a blank input.
- `lean.csv`: House rows unchanged; Senate rows now Cook; 14,200 rows, validates.

## 7. Not done (on purpose)

- `baseline.features` is still `[lean]`. Add inputs yourself after reading section 5.
- The residual model is still a skeleton.
- Nothing builds 2026 prediction rows yet (the national row and incumbency for 2026 are
  loaded and ready; OH/FL specials included).
- The old `incumbency_1998_2026.csv` (my build from 538 + congress-legislators) is kept only
  as a cross-check.
