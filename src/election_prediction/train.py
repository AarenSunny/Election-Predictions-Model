"""
Stage 2: residual model + full pipeline.  *** SKELETON: model parts not implemented. ***

Pipeline (run with `python -m election_prediction.train --config configs/baseline.yaml`):
    1. load + validate data                  data.load_table / validate
    2. build features                        features.*
    3. stage-1 baseline, out-of-fold         baseline.baseline_residuals       [TODO model]
    4. stage-2 LightGBM on residuals         fit_residual_model below          [TODO model]
    5. margins -> probabilities              probability.sigma_components / prob_from_margin
    6. score + benchmark                     evaluate.*, benchmarks.backtest
    7. log the run                           evaluate.log_experiment -> reports/experiments.csv
"""

from __future__ import annotations

import argparse

import pandas as pd

from .config import load_config


def fit_residual_model(train: pd.DataFrame, features: list[str], params: dict):
    """LightGBM regressor on the 'residual' column. Return a fitted model."""
    raise NotImplementedError("residual model not built yet")


def predict_residual(model, df: pd.DataFrame, features: list[str]):
    raise NotImplementedError("residual model not built yet")


def build_training_table(cfg: dict) -> pd.DataFrame:
    """Load raw files listed in cfg['data'], merge on race keys, add features."""
    raise NotImplementedError("wire up once the raw files exist")


def run(cfg: dict) -> None:
    raise NotImplementedError("pipeline steps 1-7 above")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    a = ap.parse_args(argv)
    run(load_config(a.config))


if __name__ == "__main__":
    main()
