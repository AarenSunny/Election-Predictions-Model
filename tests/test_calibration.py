import numpy as np
import pandas as pd
import pytest

from election_prediction.calibration import (
    METHODS, MarginCalibrator, blend_market, decision_audit, nested_calibration,
)


def sample():
    rng = np.random.default_rng(481)
    n = 320
    pred = rng.normal(0, 0.18, n)
    return pd.DataFrame(dict(year=np.repeat([2018, 2020, 2022, 2024], n // 4),
                             chamber=np.where(np.arange(n) % 6 == 0, 'Senate', 'House'),
                             pred_margin=pred, margin=pred + rng.normal(0, 0.07, n)))


@pytest.mark.parametrize('method', METHODS)
def test_monotone_finite_and_serializable(method, tmp_path):
    fit = MarginCalibrator(method).fit(sample())
    test = pd.DataFrame({'pred_margin': [-5, -0.1, 0, 0.1, 5], 'chamber': ['House'] * 5})
    p = fit.predict(test)
    assert np.isfinite(p).all() and ((p > 0) & (p < 1)).all()
    assert (np.diff(p) >= 0).all()
    path = tmp_path / 'fit.json'
    fit.save(path)
    assert np.allclose(MarginCalibrator.load(path).predict(test), p)


def test_unknown_chamber_uses_pooled_fallback():
    for method in METHODS:
        fit = MarginCalibrator(method).fit(sample())
        assert np.isfinite(fit.predict(pd.DataFrame({'pred_margin': [0], 'chamber': ['Other']}))).all()


def test_entire_test_year_labels_cannot_change_its_probabilities():
    table = sample()
    def predict(train, test):
        return test.pred_margin.to_numpy() + train.margin.mean() * 0.1
    original = nested_calibration(table, predict, test_years=[2024])
    changed = table.copy()
    changed.loc[changed.year == 2024, 'margin'] *= -100
    altered = nested_calibration(changed, predict, test_years=[2024])
    for method in METHODS:
        assert np.allclose(original[f'p_{method}'], altered[f'p_{method}'])


def test_forward_never_uses_future_years():
    def predict(train, test):
        assert train.year.max() < 2024
        return test.pred_margin.to_numpy()
    out = nested_calibration(sample(), predict, forward=True, test_years=[2024])
    assert set(out.year) == {2024}


def test_reject_bad_inputs():
    bad = sample()
    bad.loc[0, 'pred_margin'] = np.nan
    with pytest.raises(ValueError, match='finite'):
        MarginCalibrator().fit(bad)
    with pytest.raises(ValueError):
        blend_market([0.5], [1.1], 0.25)
    with pytest.raises(ValueError):
        blend_market([0.5], [0.5], -1)
    with pytest.raises(ValueError):
        decision_audit(pd.DataFrame({'p_model_D': [np.inf], 'p_market_D': [0.5]}))


def test_blending_endpoints_and_missing_market():
    assert np.allclose(blend_market([0.3, 0.6], [0.8, np.nan], 1), [0.3, 0.6])
    assert np.allclose(blend_market([0.3, 0.6], [0.8, np.nan], 0), [0.8, 0.6])


def test_score_threshold_exhaustive_grid_and_floor():
    from election_prediction.evaluate import points_if_correct
    for market in [0, 0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 1]:
        p = np.linspace(0, 1, 101)
        d = decision_audit(pd.DataFrame({'p_model_D': p, 'p_market_D': market}))
        ed = p * points_if_correct(market)
        er = (1-p) * points_if_correct(1-market)
        chosen = np.where(d.pick == 'D', ed, er)
        assert np.allclose(chosen, np.maximum(ed, er))
        assert np.allclose(d.pick_ev, chosen)
        assert np.isfinite(d.p_D_break_even).all()


def test_missing_market_not_given_fake_score_and_party_override():
    d = decision_audit(pd.DataFrame({'p_model_D': [0.8, 0.7], 'p_market_D': [np.nan, 0.5],
                                     'only_party': ['', 'R']}))
    assert d.pick.tolist() == ['D', 'R']
    assert np.isnan(d.pick_ev.iloc[0]) and d.no_market.iloc[0]
    assert d.pick_ev.iloc[1] == pytest.approx(0.3)


def test_residual_adapter_requires_rebuilding_generated_history():
    from election_prediction.calibration import baseline_fold_predictor
    d = sample().assign(lean=lambda x: x.pred_margin, prev_overperf=0.1)
    fp = baseline_fold_predictor(['lean'], residual_fp=lambda train, test: np.zeros(len(test)))
    with pytest.raises(ValueError, match='prepare_fold'):
        fp(d[d.year < 2024], d[d.year == 2024])


def test_residual_training_targets_exclude_outer_test_labels():
    from election_prediction.calibration import baseline_fold_predictor
    d = sample().assign(lean=lambda x: x.pred_margin)
    def residual(train, test):
        assert 2024 not in set(train.year)
        return np.repeat(train.residual.mean(), len(test))
    fp = baseline_fold_predictor(['lean'], residual_fp=residual)
    first = nested_calibration(d, fp, methods=['t5'], test_years=[2024])
    altered = d.copy()
    altered.loc[altered.year == 2024, 'margin'] *= -10
    second = nested_calibration(altered, fp, methods=['t5'], test_years=[2024])
    assert np.allclose(first.p_t5, second.p_t5)


def test_new_market_evidence_does_not_replace_benchmark_rewards():
    model = blend_market([0.4], [0.7], 0.25)
    frozen = decision_audit(pd.DataFrame({'p_model_D': model, 'p_market_D': [0.1]}))
    newer = decision_audit(pd.DataFrame({'p_model_D': model, 'p_market_D': [0.7]}))
    assert frozen.pick.iloc[0] == 'D'
    assert newer.pick.iloc[0] == 'R'
    assert frozen.p_D_break_even.iloc[0] != newer.p_D_break_even.iloc[0]
