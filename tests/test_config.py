"""The real settings file loads and passes its own consistency checks."""

import copy

import pytest

from election_prediction.config import ConfigError, load_config, validate_config


def test_baseline_yaml_loads_and_is_valid():
    cfg = validate_config(load_config("configs/baseline.yaml"))
    for section in ("years", "data", "baseline", "residual_model", "feature_horizons",
                    "probability", "polls", "kalshi", "submission"):
        assert section in cfg
    assert cfg["baseline"]["features"][:4] == ["lean", "generic_ballot",
                                               "approval_signed", "midterm_flag"]
    assert "incumbency_signed" in cfg["residual_model"]["features"]


def test_validate_config_catches_mistakes():
    cfg = load_config("configs/baseline.yaml")
    bad = copy.deepcopy(cfg)
    bad["years"]["train"].append(bad["years"]["predict"])          # leak
    bad["residual_model"]["features"].append("lean")               # in both stages
    bad["residual_model"]["features"].append("ba_plus_chg_3y")     # horizon not listed
    with pytest.raises(ConfigError) as e:
        validate_config(bad)
    msg = str(e.value)
    assert "also in train" in msg and "both baseline" in msg and "ba_plus_chg_3y" in msg
