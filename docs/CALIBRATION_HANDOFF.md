# Calibration and scoring handoff — November 2, 2026

Codex owns only `src/election_prediction/calibration.py`, `tests/test_calibration.py`,
this document, and `outputs/calibration/`. It did not change Claude's train.py,
twostage.py, forecast.py, configuration, or final submission.

## Result and recommendation

Four small calibration methods were tested on 3,235 regular races, 2010–2024.
Each outer test year is excluded from both fitting and ALL the predictions used to
fit its calibrator. Additional forward validation trains only through 2022 and
predicts 2024. This is baseline-only evidence, not validation of the new residual
model. The raw race inputs were read from baseline_residuals.csv, but every baseline
prediction was rebuilt; its saved baseline/residual columns were NOT reused.

| Method | Nested all-years log loss | Brier | Forward-2024 log loss | Brier |
|---|---:|---:|---:|---:|
| Normal, shrunk chamber uncertainty | .13024 | .03794 | .06817 | .01881 |
| Student-t, df=5, shrunk chamber uncertainty | .13025 | .03801 | .06765 | .01847 |
| Regularized sigmoid | .13214 | .03871 | .06823 | .01867 |
| Sigmoid with shrunk chamber offset | .13221 | .03872 | .06824 | .01866 |

Keep Student-t as the conservative default. Normal is essentially tied overall;
the tiny forward advantage is NOT statistically established superiority. Flexible
sigmoid calibration did not improve this baseline. No isotonic search was added:
there are few genuinely competitive races and only eight national environments.
Do not promise improved contest points based on these small differences. 2024 has
already been examined, so it is a historical check, not a pristine final lockbox.

## CRITICAL: fixed benchmark versus forecasting evidence

The user confirmed the official rules directly:

- Reward for a correct pick is -log2(p), where p is the organizers' NOVEMBER 1 odds.
- Wrong pick gets zero; p is floored at 1%, max reward 6.643856.
- Highest total score wins.

Only November 2 snapshots were found in the repository. Do NOT silently substitute
them for the frozen reward odds. Obtain the official Nov 1 race-level probabilities.
Newer market odds can inform forecasts if allowed by the submission cutoff, but they
must be a separate series from the prices determining contest rewards.

The CLI preserves incoming p_market_D as p_evidence_D and optionally joins
--benchmark (official CSV with race keys + p_market_D) as the scoring probabilities.
Without --benchmark it labels the entire candidate provisional. No final submission
was overwritten. Giving a path asserts that YOU verified it is the official snapshot;
the program cannot certify provenance just from a filename.

## Exact scoring rule

Let a = -log2(max(p_benchmark_D, .01)) and
b = -log2(max(1-p_benchmark_D, .01)). Pick D precisely when
p_forecast_D >= b / (a + b).

That is the exact expected-total-points optimum conditional on the probabilities.
The original decision.py implements this correctly. Correlation does not change
this optimum for expected SUM. It does affect the probability of winning a
leaderboard and score uncertainty; no independent-race simulation is claimed to
estimate those here. Market agreement can rationally favor picking underdogs under
this contest's payoff rule. This is not evidence the forecast beat the market.

The audit includes expected points for each side, the break-even probability,
probability distance to flipping the pick, and sensitivity across calibrators.
Method min/max is a sensitivity range, NOT a confidence interval. Missing markets
are flagged, have no invented reward, and fall back to the model-favored winner.
Known single-party overrides are preserved.

A .25 model / .75 market blend remains a user-configured assumption. The sensitivity
CSV explores 0/.25/.5/.75/1; its expected points use each scenario's own probabilities,
so DO NOT select a blend by whichever row has highest self-reported expected score.
No historical aligned benchmark series was available to establish an optimal blend.

## Commands (run from the repository with its normal Python environment)

Baseline evaluation, optional provisional candidate, no changes to final submission:

```bash
python -m election_prediction.calibration \
  --training-table data/processed/baseline_residuals.csv \
  --forecast outputs/predictions_2026_baseline_backup.csv \
  --method t5 --model-weight 0.25 \
  --out-dir outputs/calibration
```

Omit --training-table to rebuild raw inputs from the configured spreadsheets.
The --forecast flag accepts a BASELINE forecast from the exact same features/data;
it checks its margins against a fresh fit and rejects a mismatched residual model.

Once the official odds are available, add:

```bash
--benchmark path/to/official_november_1.csv
```

Outputs: nested heldout probabilities, metrics overall/recent/competitive and by
chamber/year, forward-2024 scores, reliability buckets, a JSON calibrator, decision
audit, market-weight sensitivity, metadata, and a submission candidate in the
calibration output folder. Main outputs/submission.csv is not touched.

## Integrating Claude's residual model

Use the API, not the baseline CLI, for combined margins:

```python
from election_prediction.calibration import (
    MarginCalibrator, baseline_fold_predictor, nested_calibration,
    calibration_metrics, decision_audit, blend_market,
)

# residual_fp(train, test) fits from train['residual'] and returns corrections.
# prepare_fold(train, test) rebuilds any history-derived features using ONLY the
# permitted training history, returning the two frames. Both are caller-supplied.
predict_fold = baseline_fold_predictor(
    baseline_features, residual_fp=residual_fp, prepare_fold=prepare_fold,
)
validation = nested_calibration(raw_race_table, predict_fold)
print(calibration_metrics(validation))
forward_check = nested_calibration(
    raw_race_table, predict_fold, forward=True, test_years=[2024],
)

# For production, obtain fully cross-fitted COMBINED margins on all allowed
# historical training years using the same complete predict_fold pipeline.
# Then fit and apply the calibrator to the matching combined 2026 margins:
calibrator = MarginCalibrator('t5').fit(combined_oof_training_predictions)
races_2026['p_calibrated_raw'] = calibrator.predict(races_2026)
# Optional blend uses forecasting evidence; decision_audit uses frozen benchmark.
races_2026['p_model_D'] = blend_market(
    races_2026.p_calibrated_raw, races_2026.p_evidence_D, model_weight=0.25,
)
audit = decision_audit(races_2026)  # p_market_D MUST be official frozen reward odds
```

**Leakage hazards in current parallel work:**

1. build_candidate_features.py builds prev_overperf from the globally saved
baseline residuals. A prior race's saved residual can depend on the current outer
test year via its baseline fit. Rebuild it inside each split or omit it. The adapter
rejects a nonblank prev_overperf without a prepare_fold callback. A callback must
actually rebuild the column, not just return it unchanged. Do not pull future test
results into histories for later training races in leave-one-year-out evaluation.
2. twostage.py's current step 5 calls out_of_fold on residual targets/features
computed once for all outer training years. Those targets can already depend on
the INNER calibration-test year. Use nested_calibration's full-chain refitting for
calibration comparisons, rather than assuming the saved p_* columns are fully nested.
3. Early stopping should not use the outer test year. For a strict inner validation
of tuning, regenerate baseline-derived targets within the tuning split too.
4. FEC end-of-cycle receipts may postdate the forecast cutoff; audit timing before
claiming prospective performance. Neither a nested split nor calibration fixes a
historical feature that contains hindsight.
5. Refit uncertainty/calibration for the combined model; do not reuse baseline sigma
when enabling residual corrections. Preserve no-candidate overrides and validate
all 470 race identifiers against the actual competition list.

## Validation

Focused tests cover monotonicity, finite tail probabilities, JSON roundtrips,
unknown-chamber fallback, outer-label isolation, forward-year boundaries, residual
adapter leakage safeguards, exact scoring thresholds including the 1% floor,
invalid inputs, missing markets, party overrides, and fixed benchmark/evidence
separation. Test fixtures are synthetic only inside tests; production evaluation
used the real historical tables. See test execution reported in the chat.

Sources informing the conservative calibration design:
https://scikit-learn.org/stable/modules/calibration.html
https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html
