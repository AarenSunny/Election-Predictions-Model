"""The real settings file loads and passes its own consistency checks."""

import copy

import pytest

from election_prediction.config import ConfigError, load_config, validate_config


def test_baseline_yaml_loads_and_is_valid():
    cfg = validate_config(load_config("configs/baseline.yaml"))
    for section in ("years", "data", "baseline", "residual_model", "feature_horizons",
                    "probability", "polls", "kalshi", "submission"):
        assert section in cfg
    assert cfg["baseline"]["features"][0] == "lean"
    for key in ("results", "lean", "baseline_residuals"):     # paths train.run() reads
        assert key in cfg["data"]
    # incumbency_signed lives in exactly one stage (baseline or residual model)
    stages = [f for f in ("baseline", "residual_model") if "incumbency_signed" in cfg[f]["features"]]
    assert len(stages) == 1
    for key in ("generic_ballot", "midterm_flag", "president_flag", "approval", "incumbency"):
        assert key in cfg["data"], key                                 # baseline input files
    from election_prediction.inputs import APPROVAL_SOURCES
    assert cfg["baseline"]["approval_source"] in APPROVAL_SOURCES
    assert cfg["baseline"]["drop_specials"] is True


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
