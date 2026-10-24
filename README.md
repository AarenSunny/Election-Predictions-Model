# Election-Predictions-Model

Model Citizens (Stanford Democracy Day) 2026, Full Congress track: predict the
winner of all 470 House and Senate races, scored against Kalshi odds.

## Approach

1. **Baseline**: linear regression on partisan lean + national environment.
2. **Residual model**: LightGBM predicts how far each race beats or misses the baseline
   (incumbency, money, candidate quality, polls, local conditions).
3. **Probabilities**: predicted margin / uncertainty → P(D wins).
4. **Picks**: for each race, the side with the highest expected points,
   P_model × (−log₂ P_Kalshi).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e . pytest ruff
pytest
```

## Layout

| Path | What it does |
| --- | --- |
| `configs/baseline.yaml` | data paths, years, features, model settings, seed |
| `data/README.md` | column contract for every raw file |
| `src/election_prediction/data.py` | loaders, key cleanup, validation |
| `src/election_prediction/features.py` | feature helpers |
| `src/election_prediction/splits.py` | leave-one-year-out splits |
| `src/election_prediction/evaluate.py` | contest score, error metrics, calibration, experiment log |
| `src/election_prediction/probability.py` | margin → probability, sigma |
| `src/election_prediction/decision.py` | expected-value pick rule |
| `src/election_prediction/benchmarks.py` | strategies to beat, backtests |
| `src/election_prediction/kalshi.py` | Kalshi snapshot |
| `src/election_prediction/predict.py` | build + validate submission |
| `src/election_prediction/baseline.py`, `train.py` | models (skeletons) |
| `reports/experiments.csv` | one row per experiment |

## Commands

```bash
python -m election_prediction.kalshi --out data/raw/kalshi_2026-11-01.csv   # needs internet
python -m election_prediction.train --config configs/baseline.yaml          # once models exist
```

## Data sources

See `data/README.md`.
