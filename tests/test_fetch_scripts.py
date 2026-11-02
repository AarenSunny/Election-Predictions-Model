"""
Parsers in scripts/fetch_public_data.py, checked on hand-typed rows shaped like the
real downloads (FEC candidate summary, Census API, BLS flat file). No network.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("fetch", ROOT / "scripts" / "fetch_public_data.py")
fetch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch)


def test_cd_codes():
    assert fetch.cd_to_int("00") == 0 and fetch.cd_to_int("17") == 17
    assert np.isnan(fetch.cd_to_int("ZZ")) and np.isnan(fetch.cd_to_int("98"))


def _fec_rows():
    cols = ["Cand_Id", "Cand_Name", "Cand_Office", "Cand_Office_St", "Cand_Office_Dist",
            "Cand_Party_Affiliation", "Cand_Incumbent_Challenger_Open_Seat", "Total_Receipt",
            "Cand_Contribution", "Cand_Loan", "Individual_Unitemized_Contribution",
            "Individual_Itemized_Contribution", "Other_Committee_Contribution",
            "Party_Committee_Contribution", "Cash_On_Hand_COP", "Coverage_End_Date", "year"]
    rows = [
        ["H2NY17001", "MALONEY, SEAN PATRICK", "H", "NY", "17", "DEM", "I", "5000000", "0", "0",
         "800000", "3000000", "1000000", "0", "200000", "12/31/2022", 2022],
        ["H2NY17002", "LAWLER, MICHAEL", "H", "NY", "17", "REP", "C", "3000000", "100000", "50000",
         "300000", "2000000", "400000", "0", "150000", "12/31/2022", 2022],
        ["H2NY17003", "PRIMARYLOSER, ALEX", "H", "NY", "17", "DEM", "C", "9000000", "0", "0",
         "100", "100", "0", "0", "0", "12/31/2022", 2022],      # lost the primary, raised more
        ["S2PA00001", "FETTERMAN, JOHN", "S", "PA", "00", "DEM", "O", "70000000", "0", "0",
         "30000000", "30000000", "1000000", "0", "1000000", "12/31/2022", 2022],
    ]
    return pd.DataFrame([[str(v) for v in r] for r in rows], columns=cols).assign(year=2022)


def test_parse_fec_and_match_nominees():
    money = fetch.parse_fec(_fec_rows())
    lawler = money[money["name"].str.startswith("LAWLER")].iloc[0]
    assert lawler["party"] == "R" and lawler["self_funding"] == 150000
    assert lawler["small_donor_contrib"] == 300000 and lawler["pac_contrib"] == 400000
    assert money.loc[money["chamber"] == "Senate", "district"].iloc[0] == 0

    results = pd.DataFrame({"year": [2022, 2022], "chamber": ["House", "Senate"],
                            "state": ["NY", "PA"], "district": [17, 0],
                            "dem_name": ["Sean Patrick Maloney", "John Fetterman"],
                            "rep_name": ["Mike Lawler", "Mehmet Oz"]})
    noms = fetch.match_nominees(money, results)
    ny = noms[(noms["state"] == "NY")].set_index("party")
    assert ny.loc["D", "name"].startswith("MALONEY")   # nominee, not the bigger-spending loser
    assert ny.loc["R", "name"].startswith("LAWLER")
    assert len(noms[noms["state"] == "PA"]) == 1        # Oz not in this FEC sample -> no row


def test_tidy_acs_shares():
    raw = {"NAME": "Congressional District 17 (118th Congress), New York", "state": "36",
           "congressional district": "17", "vintage": 2022,
           "pop": 760000, "median_age": 41.0, "median_income_nominal": 118000,
           "pov_universe": 1000, "pov_below": 70,
           "race_total": 1000, "nh_white": 600, "nh_black": 90, "nh_aian": 5, "nh_asian": 60,
           "nh_nhpi": 1, "nh_other": 4, "nh_two": 20, "hispanic": 220,
           "median_gross_rent_nominal": 1700, "median_home_value_nominal": 550000,
           "emp_total": 1000, "emp_ag_mining": 5, "emp_construction": 60, "emp_manufacturing": 50,
           "emp_wholesale": 20, "emp_retail": 100, "emp_transport_util": 50, "emp_information": 30,
           "emp_finance_realestate": 90, "emp_professional": 140, "emp_edu_health": 280,
           "emp_arts_hospitality": 70, "emp_other_services": 40, "emp_public_admin": 65,
           **{f"edu_{i:03d}": 0 for i in range(1, 26)}}
    raw.update(edu_001=500, edu_010=40, edu_016=10, edu_022=120, edu_023=80, edu_025=30)
    t = fetch.tidy_acs(pd.DataFrame([raw])).iloc[0]
    assert t["state"] == "NY" and t["district"] == 17 and t["congress"] == 118
    assert np.isclose(t["poverty_rate"], 0.07) and np.isclose(t["pct_hispanic"], 0.22)
    assert np.isclose(t["pct_other"], 0.030)
    assert np.isclose(t["ind_prof_finance"], 0.26) and np.isclose(t["ind_edu_health"], 0.28)
    assert np.isclose(t["less_than_hs"], 50 / 500) and np.isclose(t["ba_plus"], 230 / 500)
    shares = t[["pct_white", "pct_black", "pct_hispanic", "pct_asian", "pct_other"]].sum()
    assert np.isclose(shares, 1.0)


def test_tidy_acs_census_missing_codes_become_blank():
    raw = {c: 100 for c in list(fetch.ACS_VARS.values()) + list(fetch.EDU_VARS.values())}
    raw.update({"NAME": "Congressional District 1 (113th Congress), Alaska", "state": "02",
                "congressional district": "00", "vintage": 2014,
                "median_income_nominal": -666666666})
    t = fetch.tidy_acs(pd.DataFrame([raw])).iloc[0]
    assert t["district"] == 0 and np.isnan(t["median_income_nominal"])


def test_parse_laus_keeps_annual_unemployment_rate():
    text = ("series_id                     \tyear\tperiod\tvalue\tfootnote_codes\n"
            "LAUCN060370000000003          \t2022\tM13\t   5.0\t\n"     # LA County rate, annual
            "LAUCN060370000000003          \t2022\tM01\t   5.9\t\n"     # monthly: dropped
            "LAUCN060370000000004          \t2022\tM13\t250000\t\n")    # unemployed count: dropped
    d = fetch.parse_laus(text.encode())
    assert len(d) == 1 and d.iloc[0]["county_fips"] == "06037"
    assert np.isclose(d.iloc[0]["unemployment"], 0.05)
