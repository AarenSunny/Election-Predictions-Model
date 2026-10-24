"""Model Citizens 2026 election-prediction package.

Module map
----------
config       load configs/*.yaml
data         raw-file loaders, key standardization, schema validation
features     feature-engineering helpers (D-minus-R, shares, changes, crosswalks, polls)
splits       leave-one-year-out and other leak-free splits
evaluate     contest scoring rule, error metrics, calibration tables, experiment log
probability  predicted margin -> win probability, sigma, recalibration
decision     expected-value pick rule
benchmarks   simple strategies to beat + backtests
kalshi       fetch and parse Kalshi market prices
predict      build and validate the submission CSV
baseline     stage-1 baseline model (SKELETON)
train        stage-2 residual model + CV loop (SKELETON)

Conventions: probabilities and margins are for the DEMOCRAT; margins are
fractions (D+8 = 0.08); sides are "D"/"R".
"""

__version__ = "0.1.0"
