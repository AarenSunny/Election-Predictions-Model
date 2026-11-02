# Data sources and coverage

Last updated 2026-10-30. Two ways data arrives:

1. **Built from public GitHub repos** with `scripts/build_github_data.py` (already in this folder).
2. **Downloaded from official sites** with `scripts/fetch_public_data.py`, run on your own computer.

## 1. Built from GitHub (in this folder now)

| File | Rows | Covers | Source |
| --- | --- | --- | --- |
| `results_538_1998_2024.csv` | 6,643 races | House + Senate general elections 1998–2024, 50 states | [fivethirtyeight/election-results](https://github.com/fivethirtyeight/election-results) |
| `incumbency_1998_2026.csv` | 7,112 races | incumbent party 1998–2026; whether a sitting member ran + prior terms 1998–2024 | election-results `races.csv` + [unitedstates/congress-legislators](https://github.com/unitedstates/congress-legislators) |
| `polls_538_1998_2022.csv` | 4,105 polls | House + Senate general polls, 1998–2022 (thin before 2006) | [fivethirtyeight/data](https://github.com/fivethirtyeight/data) `pollster-ratings/2023/raw-polls.csv` |
| `pollster_ratings_538_2023.csv` | 517 pollsters | 538 grades + the 0–1 weight used as `pollster_rating` | fivethirtyeight/data `pollster-ratings/2023` |

Feeds these residual features: `incumbency_signed`, `chamber_senate`, `poll_gap`, `has_polls`,
`poll_count_30d`, `days_since_poll` (and the results are the target for 1998–2024).

### Caveats

- **Results:** first round only for ranked-choice races (ME, AK). Louisiana House races and the 2020
  Georgia Senate special are the November all-party ("jungle") contest, flagged `jungle_primary`;
  only the top D and top R are counted. California and Washington's June top-two primaries are excluded.
  `same_party` = no D or no R in the general; `uncontested` = one candidate or marked unopposed.
- **Incumbency:** `incumbent_running` matches general-election candidates to members serving on
  Oct 1 of that year by last name + first initial (same state and chamber, any district, so members
  moved by redistricting count). Blank for 2026: candidate lists aren't in the source; fill from
  `race_list_2026.csv`. 2026 has 33 regular + 1 special Senate race; the second 2026 special is missing
  from the source.
- **Polls:** 538's archive only includes polls from roughly the **final 3 weeks** of each race, and
  the 5-day cutoff removes some of those. So historical `poll_count_30d` runs lower than 2026 will.
  `end_date` is 538's poll date. `internal` = sponsored by a party/campaign. `pollster_rating` =
  538 grade mapped to 0–1 (A+ 1.0 … F 0.1, banned 0; mapping in `build_github_data.py`).
- **2024 and 2026 polls are not here.** 538's polling database stopped updating in 2025.

## Lean (`lean.csv`, baseline input)

| Rows | How it's built |
| --- | --- |
| Senate, 1998–2026 | `scripts/convert_cook_pvi.py --senate-xlsx data/raw/cook/COOK_SENATE_PVI.xlsx`: Cook state PVI (1997, 2001, …, 2025 blocks; each race uses the latest block published before it). Backup: `scripts/build_lean.py` computes the same thing from presidential results (538 from 2000; 1992/1996 from MIT `data/raw/mit/`); it agrees with Cook within 0.01 on average |
| House, 1998–2026 | `scripts/convert_cook_pvi.py` from `data/raw/cook/Cook_PVI_1997-2026.xlsx` (Cook's data; not committed) |

Rebuild: `python scripts/convert_cook_pvi.py --xlsx data/raw/cook/Cook_PVI_1997-2026.xlsx --senate-xlsx data/raw/cook/COOK_SENATE_PVI.xlsx --out data/raw/lean.csv`

The workbook's sheet for each year is listed at the top of `convert_cook_pvi.py` (right lines,
presidential inputs all before that election). House rows missing on purpose, because those states
redrew mid-decade and no sheet has both the right lines and only earlier presidential results:
2000 NC; 2004 TX; 2016 FL, NC, VA; 2020 NC; 2024 AL, GA, LA, NC, NY. Specials in redistricting
years (2002, 2012, 2022) use the previous Congress's sheet, since they ran on the old lines. PA 2018 regular races use the new court map;
its 2018 specials use the old map. Cook weights the most recent presidential race 75/25 from 2023 on
(50/50 before; the 2003 sheet appears to use 2000 alone, since 1996 wasn't re-tabulated on the new
lines); Cook's Senate PVI follows the same formula changes.

## Baseline inputs (team spreadsheets, loaded by `src/election_prediction/inputs.py`)

| File | What it holds | Notes |
| --- | --- | --- |
| `national/Generic_Ballot_Data.xlsx` | generic-ballot D and R %, 1998–2026 | 2002 is margin-only (R+1.7); 2000 is an early-September poll |
| `national/MidtermFlag.xlsx` | 1 = midterm year | checked against the calendar on load |
| `national/President_Flag.xlsx` | +1 D president, −1 R | |
| `national/Presidential_Approval_Ratings.xlsx` | sheet 1 "Unreliable data but more" (broad), sheet 2 "Reliable"; latest poll, 30-day, 90-day | the comparison tool scores all 6 versions |
| `incumbency/congressional_incumbency_1998_2026.xlsx` | D / R incumbent-running flags for every House and Senate race, G (regular) and S (special) | replaces `incumbency_1998_2026.csv` as the source of `incumbency_signed`; agrees with it on 98% of races, and is right where they differ |

Details and the format quirks handled: `docs/BASELINE_CHANGES.md`.

## 2. Official sites: run `scripts/fetch_public_data.py`

These sites can't be reached from Claude's workspace, so they aren't downloaded yet. On your computer:

```bash
python scripts/fetch_public_data.py --email you@stanford.edu
```

| Script output | Feeds | Source |
| --- | --- | --- |
| `fec_money_2010_2026.csv`, `fec_nominees_2010_2024.csv` | `cash_share`, `donor_mix_diff` | [FEC candidate summary](https://www.fec.gov/data/browse-data/?tab=bulk-data) |
| `acs_district_2010_2024.csv` | income, poverty, education, age, ethnicity, industry, rent level | Census ACS 5-year API |
| `laus_county_annual.csv` | `unemployment` + changes | BLS LAUS |
| `cpi_u_annual.csv` | real-dollar income | BLS CPI-U |
| `zillow_county_annual.csv` | home value / rent changes | Zillow ZHVI, ZORI |
| `fhfa_county_hpi.csv` | `home_value_chg_*` | FHFA county HPI |
| `county_cd_relationship_118.csv` | county → district (area-weighted fallback) | Census relationship file |
| `cd_land_area.csv` | `log_density` | Census Gazetteer |

**Manual downloads** (instructions printed by the script): DIME CFscores (`cfscore_diff`,
`moderation_diff`), U.S. Religion Census + state attendance (`religiosity`), Geocorr population-
weighted crosswalk.

## Still not covered by either route

- `scandal_diff`: hand-coded by the team.
- `issue_gap_diff`: needs candidate issue positions + district voter positions/importance.
- 2024 and 2026 race polls: FiftyPlusOne, RealClearPolling, Wikipedia race pages.
- 2026 candidate list: Ballotpedia (`race_list_2026.csv`).
