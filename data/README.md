# Data

Raw downloads go in `data/raw/`, cleaned/merged tables in `data/processed/`.
File paths are set in `configs/baseline.yaml`.

## Conventions (every file)

| Rule | Example |
| --- | --- |
| Race keys: `year, chamber, state, district, special` | `2018, House, NY, 17, False` |
| `chamber` is `House` or `Senate` | |
| `state` is the two-letter code (full names are converted automatically) | `NY` |
| `district` is an integer; **0 for at-large seats and every Senate race** | `AL` → 0 |
| `special` is True only for special elections | 2026 OH/FL Senate |
| Margins, leans, approval, generic ballot are **fractions in D terms** | D+8 → `0.08`, R+5 → `-0.05` |
| Party columns use `D`, `R`, `O` | |
| District lines: each year uses **that year's** lines | 2026 rows on 2026 maps |

`data.validate()` checks all of this and lists every problem at once.

## Files and columns

| File | Source | Required columns |
| --- | --- | --- |
| `mit_house_1976_2024.csv`, `mit_senate_1976_2024.csv` | [MIT Election Data + Science Lab](https://electionlab.mit.edu/data) | as downloaded (loaded by `data.load_mit_results`) |
| `national.csv` | 538 archive, FiftyPlusOne | `year, generic_ballot, net_approval, pres_party` (one row per year, values as of the poll cutoff) |
| `lean.csv` | Cook PVI, Dave's Redistricting App, The Downballot | keys + `lean` |
| `candidates.csv` | Ballotpedia, DIME (CFscores), hand-coded | keys + `party, name, incumbent, scandal, cfscore, fec_candidate_id` |
| `fec/` | [FEC bulk data](https://www.fec.gov/data/browse-data/?tab=bulk-data) | "All candidates" summary (`weball{yy}.txt`, loaded by `data.load_fec_weball`) **plus unitemized individual receipts (< $200) per candidate** for the small-donor share; the summary file doesn't separate small donors |
| `polls.csv` | FiftyPlusOne (2026), 538 archive (past) | keys + `end_date, pollster, sample_size, dem_pct, rep_pct` (percent, 0–100); optional `pollster_rating` (weight, e.g. 0–1) and `internal` (True for internal / sponsor-released) |
| `ratings.csv` | Cook / Sabato / Inside Elections | keys + `rating` (e.g. `Lean R`, `Toss-up`) |
| `race_list_2026.csv` | Ballotpedia | keys + `incumbent_party, open_seat, uncontested, only_party, top_two, rcv, redistricted` |
| `kalshi_2026-11-01.csv` | `python -m election_prediction.kalshi` | produced by the script |

District/county measures (economy, industry, density, education, age,
religiosity, ethnicity) can arrive as one file per source; what matters is the
**processed table**: one row per race per year, with the exact column names
listed under `residual_model.features` in `configs/baseline.yaml`.

| Measure | Source | Level |
| --- | --- | --- |
| Unemployment | BLS LAUS | county → district |
| Home values | FHFA county HPI (Zillow ZHVI as a check) | county → district |
| Rent | ACS median gross rent + 2000 census (Zillow ZORI optional, from ~2016) | district |
| Median income (real $), poverty, education, median age, ethnicity, industry (C24050) | ACS (1990/2000 census for early years) | district |
| Population density | ACS + Census land area | district |
| Religiosity | U.S. Religion Census (county % religious) × Gallup/PRRI state attendance | county → district |

## Change columns

`<measure>_chg_<N>y` = current value minus the value N years earlier, same
district (`features.add_horizon_changes`). Which N each measure gets is set in
`feature_horizons` in the config. Rules from the plan:

- ACS items use only 5- and 10-year changes (1- and 2-year ACS changes overlap and are noise).
- 20-year changes only for unemployment, home value and religiosity.
- Anchor every ACS value to the **latest vintage released before that election**.
- Keep each series to one source (no splicing Zillow onto ACS).
- Candidate, money and issue features have no change columns.
- A change that can't be computed is left blank (Null).

## Timing rule

Every value must be what was known **at the poll cutoff that year** (5 days before
election day): polls, generic ballot, approval, money. End-of-cycle numbers leak
hindsight into training.
