import numpy as np
import pandas as pd

from election_prediction.probability import fit_isotonic, prob_from_margin, sigma_components


def test_prob_from_margin_symmetry():
    assert np.isclose(prob_from_margin(0.0, 0.05), 0.5)
    assert np.isclose(prob_from_margin(0.03, 0.05) + prob_from_margin(-0.03, 0.05), 1.0)


def test_t_has_fatter_tails():
    assert prob_from_margin(-0.2, 0.05, dist="t") > prob_from_margin(-0.2, 0.05)


def test_sigma_components_simple_case():
    # Year A misses by +0.02 everywhere, year B by -0.02: purely national error.
    df = pd.DataFrame({"year": [1, 1, 2, 2], "error": [0.02, 0.02, -0.02, -0.02]})
    s = sigma_components(df)
    assert s["sigma_race"] == 0 and s["sigma_national"] > 0
    assert np.isclose(s["sigma_total"], s["sigma_national"])


def test_isotonic_stays_inside_0_1():
    iso = fit_isotonic([0.1, 0.4, 0.6, 0.9], [0, 0, 1, 1])
    out = iso.predict([0.0, 0.5, 1.0])
    assert out.min() > 0 and out.max() < 1
