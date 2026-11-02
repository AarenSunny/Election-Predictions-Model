"""
Download the residual-model data that lives on official sites, and tidy it.

RUN THIS ON YOUR OWN COMPUTER (normal internet). From the repo root:

    python scripts/fetch_public_data.py --email you@stanford.edu
    python scripts/fetch_public_data.py --email you@stanford.edu --only fec acs

--email is sent in the User-Agent header: BLS (and politely, the others) require a
contact address from automated downloads. Add --census-key KEY if you have a free
Census API key (https://api.census.gov/data/key_signup.html); without one the
Census API still works for this many calls.

Each source runs on its own: if one fails, the rest still run, and a summary at the
end says what worked. Raw downloads are saved untouched under data/raw/<source>/ and
skipped on re-runs (use --refresh to re-download). Tidy tables go in data/raw/.

Sources and outputs
-------------------
fec      FEC candidate summary 2010-2026            -> fec_money_2010_2026.csv
acs      Census ACS 5-year, by congressional district -> acs_district_<first>_<last>.csv
laus     BLS county unemployment (annual average)     -> laus_county_annual.csv
cpi      BLS CPI-U annual average (for real dollars)  -> cpi_u_annual.csv
zillow   Zillow county home values + rents (annual)   -> zillow_county_annual.csv
fhfa     FHFA county house price index (annual)       -> fhfa_county_hpi.csv
crosswalk Census county <-> congressional district    -> county_cd_relationship_118.csv
gazetteer Census district land area (for density)     -> cd_land_area.csv

Manual (no stable download link; instructions printed at the end):
DIME CFscores, U.S. Religion Census, Geocorr population-weighted crosswalk.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CYCLES = list(range(2010, 2027, 2))
ACS_VINTAGES = list(range(2010, 2025))       # 5-year ACS; release year = vintage + 1 (December)
USER_AGENT = "election-prediction-model/0.1 (research; contact: {email})"


# ------------------------------------------------------------------ download helper
class Fetcher:
    def __init__(self, email: str, refresh: bool = False, pause: float = 0.5):
        self.ua = USER_AGENT.format(email=email)
        self.refresh = refresh
        self.pause = pause

    def get(self, url: str, dest: Path | None = None, tries: int = 3) -> bytes:
        if dest is not None and dest.exists() and not self.refresh:
            return dest.read_bytes()
        last = None
        for i in range(tries):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": self.ua})
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                if dest is not None:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(data)
                time.sleep(self.pause)
                return data
            except urllib.error.HTTPError as e:
                last = e
                if e.code in (400, 403, 404):
                    break                      # won't fix itself
            except Exception as e:             # noqa: BLE001  (network errors: retry)
                last = e
            time.sleep(2 * (i + 1))
        raise RuntimeError(f"{url}: {last}")


def cd_to_int(s) -> int | float:
    """Census congressional-district code -> int (00 at-large -> 0; ZZ/98 -> NaN)."""
    s = str(s).strip()
    if s in ("ZZ", "98", ""):
        return np.nan
    return int(s)


STATE_FIPS = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO", "09": "CT", "10": "DE",
    "11": "DC", "12": "FL", "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN", "19": "IA",
    "20": "KS", "21": "KY", "22": "LA", "23": "ME", "24": "MD", "25": "MA", "26": "MI", "27": "MN",
    "28": "MS", "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH", "34": "NJ", "35": "NM",
    "36": "NY", "37": "NC", "38": "ND", "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI",
    "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT", "50": "VT", "51": "VA", "53": "WA",
    "54": "WV", "55": "WI", "56": "WY", "72": "PR",
}


# ------------------------------------------------------------------ FEC
def fetch_fec(f: Fetcher) -> pd.DataFrame:
    """
    FEC candidate summary (one row per candidate per cycle). Has the small-donor
    column the doc needs (Individual_Unitemized_Contribution = gifts from donors
    giving <= $200). NOTE timing: these are cycle totals, which for past cycles
    include money raised after the election; 2026 is as of download date.
    """
    frames = []
    for y in CYCLES:
        url = f"https://www.fec.gov/files/bulk-downloads/{y}/candidate_summary_{y}.csv"
        try:
            raw = f.get(url, RAW / "fec" / f"candidate_summary_{y}.csv")
        except RuntimeError as e:
            print(f"   fec {y}: {e}")
            continue
        d = pd.read_csv(io.BytesIO(raw), dtype=str, low_memory=False)
        d["year"] = y
        frames.append(d)
    if not frames:
        raise RuntimeError("no FEC files downloaded")
    return parse_fec(pd.concat(frames, ignore_index=True))


def fec_party(p) -> str:
    return "D" if p in ("DEM", "DFL") else ("R" if p == "REP" else "O")


def parse_fec(d: pd.DataFrame) -> pd.DataFrame:
    """Candidate-summary columns (strings) + 'year' -> tidy money table."""
    def num(c):
        return pd.to_numeric(d.get(c), errors="coerce")
    office = d["Cand_Office"].str.upper()
    out = pd.DataFrame({
        "year": d["year"],
        "chamber": office.map({"H": "House", "S": "Senate"}),
        "state": d["Cand_Office_St"],
        "district": np.where(office == "S", 0,
                             pd.to_numeric(d["Cand_Office_Dist"], errors="coerce")),
        "special": False,
        "fec_candidate_id": d["Cand_Id"],
        "name": d["Cand_Name"],
        "party": d["Cand_Party_Affiliation"].map(fec_party),
        "fec_status": d["Cand_Incumbent_Challenger_Open_Seat"].map(
            {"I": "incumbent", "C": "challenger", "O": "open"}),
        "receipts": num("Total_Receipt"),
        "self_funding": num("Cand_Contribution").fillna(0) + num("Cand_Loan").fillna(0),
        "small_donor_contrib": num("Individual_Unitemized_Contribution"),
        "itemized_indiv_contrib": num("Individual_Itemized_Contribution"),
        "pac_contrib": num("Other_Committee_Contribution"),
        "party_contrib": num("Party_Committee_Contribution"),
        "cash_on_hand": num("Cash_On_Hand_COP"),
        "coverage_end": d.get("Coverage_End_Date"),
    })
    out = out[out["chamber"].notna()]
    # FEC files at-large seats and Senate races as district "00" -> 0, matching the results file
    return out


def match_nominees(money: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """
    FEC lists every candidate who filed, primary losers included. Keep the general-
    election nominee per party by matching last names to results (dem_name/rep_name).
    Returns one row per race-party with an added 'matched' flag.
    """
    def last(n):
        n = str(n).upper().replace(",", " ").split()
        return n[0] if n else ""          # FEC names are "LAST, FIRST"
    def last_from_full(n):
        toks = [t for t in str(n).upper().replace(".", "").split()
                if t not in ("JR", "SR", "II", "III", "IV")]
        return toks[-1] if toks else ""
    keys = ["year", "chamber", "state", "district"]
    rows = []
    for party, col in (("D", "dem_name"), ("R", "rep_name")):
        r = results[keys + [col]].dropna(subset=[col]).copy()
        r["nom_last"] = r[col].map(last_from_full)
        m = money[money["party"] == party].copy()
        m["fec_last"] = m["name"].map(last)
        j = r.merge(m, on=keys, how="left")
        j = j[j["fec_last"] == j["nom_last"]]
        j = j.sort_values("receipts", ascending=False).drop_duplicates(keys)
        rows.append(j.drop(columns=[col, "nom_last", "fec_last"]))
    return pd.concat(rows, ignore_index=True)


# ------------------------------------------------------------------ ACS
ACS_VARS = {
    "B01003_001E": "pop",
    "B01002_001E": "median_age",
    "B19013_001E": "median_income_nominal",
    "B17001_001E": "pov_universe", "B17001_002E": "pov_below",
    "B03002_001E": "race_total", "B03002_003E": "nh_white", "B03002_004E": "nh_black",
    "B03002_005E": "nh_aian", "B03002_006E": "nh_asian", "B03002_007E": "nh_nhpi",
    "B03002_008E": "nh_other", "B03002_009E": "nh_two", "B03002_012E": "hispanic",
    "B25064_001E": "median_gross_rent_nominal", "B25077_001E": "median_home_value_nominal",
    # industry, civilian employed 16+ (C24050 lines 001-014)
    "C24050_001E": "emp_total", "C24050_002E": "emp_ag_mining", "C24050_003E": "emp_construction",
    "C24050_004E": "emp_manufacturing", "C24050_005E": "emp_wholesale", "C24050_006E": "emp_retail",
    "C24050_007E": "emp_transport_util", "C24050_008E": "emp_information",
    "C24050_009E": "emp_finance_realestate", "C24050_010E": "emp_professional",
    "C24050_011E": "emp_edu_health", "C24050_012E": "emp_arts_hospitality",
    "C24050_013E": "emp_other_services", "C24050_014E": "emp_public_admin",
}
# education 25+ (B15003 exists from the 2012 5-year release)
EDU_VARS = {f"B15003_{i:03d}E": f"edu_{i:03d}" for i in range(1, 26)}


def _census(f: Fetcher, year: int, variables: list[str], key: str | None, tag: str) -> pd.DataFrame:
    params = {"get": ",".join(["NAME"] + variables),
              "for": "congressional district:*", "in": "state:*"}
    if key:
        params["key"] = key
    url = f"https://api.census.gov/data/{year}/acs/acs5?" + urllib.parse.urlencode(params)
    raw = f.get(url, RAW / "acs" / f"acs5_{year}_cd_{tag}.json")
    rows = json.loads(raw)
    return pd.DataFrame(rows[1:], columns=rows[0])


def fetch_acs(f: Fetcher, key: str | None) -> pd.DataFrame:
    """
    One row per district per ACS 5-year vintage. 'congress' tells you which district
    lines the vintage uses: it is the Congress in effect when the data were released,
    so the 2012 and 2022 elections (new maps) need care. Anchor rule from the plan:
    for an election in year Y use vintage Y-2 (vintage Y-1 is released in December,
    after the election).
    """
    out = []
    for y in ACS_VINTAGES:
        try:
            a = _census(f, y, list(ACS_VARS), key, "main")
        except RuntimeError as e:
            print(f"   acs {y}: {e}")
            continue
        try:
            e = _census(f, y, list(EDU_VARS), key, "edu")
            a = a.merge(e.drop(columns=["NAME"]), on=["state", "congressional district"], how="left")
        except RuntimeError:
            pass                              # pre-2012: no B15003 -> education stays blank
        a = a.rename(columns={**ACS_VARS, **EDU_VARS})
        a["vintage"] = y
        out.append(a)
    if not out:
        raise RuntimeError("no ACS vintages downloaded")
    return tidy_acs(pd.concat(out, ignore_index=True))


def tidy_acs(a: pd.DataFrame) -> pd.DataFrame:
    """Raw ACS columns (renamed via ACS_VARS/EDU_VARS) + 'vintage' -> shares per district."""
    a = a.copy()
    for c in a.columns:
        if c not in ("NAME", "state", "congressional district"):
            a[c] = pd.to_numeric(a[c], errors="coerce")
            a.loc[a[c] < 0, c] = np.nan       # Census uses negative codes for "not available"
    a["state_po"] = a["state"].map(STATE_FIPS)
    a["district"] = a["congressional district"].map(cd_to_int)
    a["congress"] = a["NAME"].str.extract(r"\((\d+)\w\w Congress\)")[0].astype(float)
    t = pd.DataFrame({
        "vintage": a["vintage"], "congress": a["congress"], "state": a["state_po"],
        "district": a["district"], "pop": a["pop"], "median_age": a["median_age"],
        "median_income_nominal": a["median_income_nominal"],
        "poverty_rate": a["pov_below"] / a["pov_universe"],
        "pct_black": a["nh_black"] / a["race_total"],
        "pct_hispanic": a["hispanic"] / a["race_total"],
        "pct_asian": a["nh_asian"] / a["race_total"],
        "pct_other": (a["nh_aian"] + a["nh_nhpi"] + a["nh_other"] + a["nh_two"]) / a["race_total"],
        "pct_white": a["nh_white"] / a["race_total"],        # reference group (dropped in model)
        "median_gross_rent_nominal": a["median_gross_rent_nominal"],
        "median_home_value_nominal": a["median_home_value_nominal"],
        "ind_manufacturing": a["emp_manufacturing"] / a["emp_total"],
        "ind_ag_mining": a["emp_ag_mining"] / a["emp_total"],
        "ind_construction": a["emp_construction"] / a["emp_total"],
        "ind_edu_health": a["emp_edu_health"] / a["emp_total"],
        # doc: "professional and finance" -> finance/real estate + professional services
        # (+ information, which the doc's reference list doesn't mention)
        "ind_prof_finance": (a["emp_finance_realestate"] + a["emp_professional"]
                             + a["emp_information"]) / a["emp_total"],
        "ind_public_admin": a["emp_public_admin"] / a["emp_total"],
    })
    if "edu_001" in a:
        tot = a["edu_001"]
        t["less_than_hs"] = a[[f"edu_{i:03d}" for i in range(2, 17)]].sum(axis=1, min_count=1) / tot
        t["ba_plus"] = a[[f"edu_{i:03d}" for i in range(22, 26)]].sum(axis=1, min_count=1) / tot
    return t[t["state"].notna() & t["district"].notna() & (t["state"] != "PR")].reset_index(drop=True)


# ------------------------------------------------------------------ BLS
def fetch_laus(f: Fetcher) -> pd.DataFrame:
    """County unemployment rate, annual average (period M13), 1990-present, as a fraction."""
    raw = f.get("https://download.bls.gov/pub/time.series/la/la.data.64.County",
                RAW / "bls" / "la.data.64.County")
    return parse_laus(raw)


def parse_laus(raw: bytes) -> pd.DataFrame:
    d = pd.read_csv(io.BytesIO(raw), sep="\t", dtype=str)
    d.columns = [c.strip() for c in d.columns]
    d["series_id"] = d["series_id"].str.strip()
    d = d[(d["period"].str.strip() == "M13") & d["series_id"].str.endswith("03")]  # 03 = unemployment rate
    return pd.DataFrame({
        "year": d["year"].astype(int),
        "county_fips": d["series_id"].str[5:10],
        "unemployment": pd.to_numeric(d["value"].str.strip(), errors="coerce") / 100.0,
    }).dropna().reset_index(drop=True)


def fetch_cpi(f: Fetcher) -> pd.DataFrame:
    """CPI-U all items, U.S. city average, annual average (to turn ACS dollars into real dollars)."""
    raw = f.get("https://download.bls.gov/pub/time.series/cu/cu.data.1.AllItems",
                RAW / "bls" / "cu.data.1.AllItems")
    d = pd.read_csv(io.BytesIO(raw), sep="\t", dtype=str)
    d.columns = [c.strip() for c in d.columns]
    d = d[(d["series_id"].str.strip() == "CUUR0000SA0") & (d["period"].str.strip() == "M13")]
    return pd.DataFrame({"year": d["year"].astype(int),
                         "cpi_u": pd.to_numeric(d["value"].str.strip())}).reset_index(drop=True)


# ------------------------------------------------------------------ housing
# County files follow Zillow's published naming pattern (Metro_* versions are linked on
# zillow.com/research/data); if a County_* name changes, download it from that page.
ZILLOW = {
    "zhvi": "https://files.zillowstatic.com/research/public_csvs/zhvi/"
            "County_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv",
    "zori": "https://files.zillowstatic.com/research/public_csvs/zori/"
            "County_zori_uc_sfrcondomfr_sm_month.csv",
}


def fetch_zillow(f: Fetcher) -> pd.DataFrame:
    """Zillow county home value (ZHVI, from 2000) and rent (ZORI, from ~2015), annual averages."""
    frames = []
    for name, url in ZILLOW.items():
        try:
            raw = f.get(url, RAW / "zillow" / Path(url).name)
        except RuntimeError as e:
            print(f"   zillow {name}: {e}")
            continue
        d = pd.read_csv(io.BytesIO(raw), dtype={"StateCodeFIPS": str, "MunicipalCodeFIPS": str})
        d["county_fips"] = d["StateCodeFIPS"].str.zfill(2) + d["MunicipalCodeFIPS"].str.zfill(3)
        months = [c for c in d.columns if c[:2] in ("19", "20") and len(c) == 10]
        long = d.melt(id_vars=["county_fips"], value_vars=months, var_name="month", value_name=name)
        long["year"] = long["month"].str[:4].astype(int)
        frames.append(long.groupby(["county_fips", "year"], as_index=False)[name].mean())
    if not frames:
        raise RuntimeError("no Zillow files downloaded")
    out = frames[0]
    for fr in frames[1:]:
        out = out.merge(fr, on=["county_fips", "year"], how="outer")
    return out


FHFA_URLS = [
    "https://www.fhfa.gov/hpi/download/annual/hpi_at_bdl_county.xlsx",
    "https://www.fhfa.gov/hpi/download/annual/hpi_at_bdl_county.csv",
]


def fetch_fhfa(f: Fetcher) -> pd.DataFrame:
    """FHFA county annual HPI (developmental index). Needs openpyxl for the .xlsx version."""
    last = None
    for url in FHFA_URLS:
        try:
            raw = f.get(url, RAW / "fhfa" / Path(url).name)
        except RuntimeError as e:
            last = e
            continue
        if url.endswith(".xlsx"):
            d = pd.read_excel(io.BytesIO(raw), skiprows=6, dtype={"FIPS code": str})
        else:
            d = pd.read_csv(io.BytesIO(raw), dtype=str)
        d.columns = [str(c).strip() for c in d.columns]
        fips = next(c for c in d.columns if "FIPS" in c.upper())
        hpi = next(c for c in d.columns
                   if c.upper().startswith("HPI") and "1990" not in c and "2000" not in c)
        return pd.DataFrame({"county_fips": d[fips].astype(str).str.zfill(5),
                             "year": pd.to_numeric(d["Year"], errors="coerce"),
                             "home_value_index": pd.to_numeric(d[hpi], errors="coerce")}).dropna()
    raise RuntimeError(f"FHFA download failed ({last}). Download 'Counties (Developmental Index)' "
                       "from https://www.fhfa.gov/data/hpi/datasets?tab=additional-data into "
                       "data/raw/fhfa/ and re-run with --only fhfa")


# ------------------------------------------------------------------ geography
def fetch_crosswalk(f: Fetcher) -> pd.DataFrame:
    """
    Census relationship file: 2020 counties x 118th-Congress districts with land-area
    overlap. Fallback only: weights are AREA shares, not population shares. Use the
    Geocorr population-weighted file when you have it (see manual steps).
    """
    url = "https://www2.census.gov/geo/docs/maps-data/data/rel2020/cd-sld/tab20_cd11820_county20_natl.txt"
    raw = f.get(url, RAW / "census" / "tab20_cd11820_county20_natl.txt")
    d = pd.read_csv(io.BytesIO(raw), sep="|", dtype=str)
    cd = d["GEOID_CD118_20"]
    area = pd.to_numeric(d["AREALAND_PART"], errors="coerce")
    out = pd.DataFrame({"state": cd.str[:2].map(STATE_FIPS), "district": cd.str[2:].map(cd_to_int),
                        "county_fips": d["GEOID_COUNTY_20"], "area_part": area})
    out = out.dropna(subset=["state", "district", "county_fips"])
    out["afact_area"] = out["area_part"] / out.groupby(["state", "district"])["area_part"].transform("sum")
    return out


GAZETTEER = {   # gazetteer year -> file stem; one file per set of district lines
    # file names checked against www2.census.gov/geo/docs/maps-data/data/gazetteer/ (Oct 2026)
    2012: "2012_Gaz_112CDs_national",   # lines used through the 2010 election
    2014: "2014_Gaz_114CDs_national",   # 2012-2016 lines
    2016: "2016_Gaz_115CDs_national",   # after 2016 FL/VA/NC changes
    2018: "2018_Gaz_116CDs_national",   # after 2018 PA change; used through 2020
    2023: "2023_Gaz_118CDs_national",   # 2022 maps
    2024: "2024_Gaz_119CDs_national",   # 2024 maps
    2026: "2026_Gaz_120CDs_national",   # 2026 maps (mid-decade redistricting)
}


def fetch_gazetteer(f: Fetcher) -> pd.DataFrame:
    """District land area (sq. miles) for population density, per set of district lines."""
    frames = []
    for y, stem in GAZETTEER.items():
        url = f"https://www2.census.gov/geo/docs/maps-data/data/gazetteer/{y}_Gazetteer/{stem}.zip"
        try:
            raw = f.get(url, RAW / "census" / f"{stem}.zip")
            z = zipfile.ZipFile(io.BytesIO(raw))
            txt = z.read(z.namelist()[0]).decode("latin-1")
        except Exception as e:  # noqa: BLE001
            print(f"   gazetteer {y}: {e}")
            continue
        d = pd.read_csv(io.StringIO(txt), sep="\t", dtype=str)
        d.columns = [c.strip() for c in d.columns]
        geoid = d["GEOID"].str.strip()
        frames.append(pd.DataFrame({
            "gazetteer_year": y, "congress": int(stem.split("_")[2][:3]),
            "state": geoid.str[:2].map(STATE_FIPS), "district": geoid.str[2:].map(cd_to_int),
            "land_sqmi": pd.to_numeric(d["ALAND_SQMI"], errors="coerce")}))
    if not frames:
        raise RuntimeError("no gazetteer files downloaded")
    return pd.concat(frames, ignore_index=True).dropna(subset=["state", "district"])


MANUAL = """
Manual downloads (no stable direct link). Save into data/raw/<folder>/ :

1. DIME CFscores (Bonica)  -> data/raw/dime/
   https://data.stanford.edu/dime  ->  "Recipients" file (dime_recipients_1979_2024 or newest).
   Columns needed: cycle, seat, state, district, party, name, recipient.cfscore, ICPSR/FEC ids.

2. U.S. Religion Census, county files 2010 and 2020  -> data/raw/religion/
   https://www.usreligioncensus.org (Data) or https://www.thearda.com (Data Archive,
   "U.S. Religion Census ... County File"). Need: county FIPS, total adherents, population.

3. Geocorr county -> congressional district, POPULATION weights  -> data/raw/geocorr/
   https://mcdc.missouri.edu/applications/geocorr2022.html
   Source = County, Target = Congressional district (118th and 119th), weight = population.
   One run per set of lines you need. (The script's Census relationship file is an
   area-weighted fallback.)

4. State weekly religious attendance (for religiosity = county % religious x state rate)
   Gallup / Pew Religious Landscape Study state tables -> data/raw/religion/state_attendance.csv
"""


SOURCES = ["fec", "acs", "laus", "cpi", "zillow", "fhfa", "crosswalk", "gazetteer"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--email", required=True, help="contact email for the User-Agent header")
    ap.add_argument("--census-key", default=None)
    ap.add_argument("--only", nargs="*", choices=SOURCES, default=SOURCES)
    ap.add_argument("--refresh", action="store_true", help="re-download files already saved")
    a = ap.parse_args(argv)
    f = Fetcher(a.email, a.refresh)
    RAW.mkdir(parents=True, exist_ok=True)
    report = {}

    def run(name, fn, out_name):
        if name not in a.only:
            return
        print(f"-> {name}")
        try:
            df = fn()
            df.to_csv(RAW / out_name, index=False)
            report[name] = f"OK  {len(df):>8,} rows -> data/raw/{out_name}"
        except Exception as e:  # noqa: BLE001
            report[name] = f"FAILED  {e}"
            traceback.print_exc(limit=1, file=sys.stdout)

    run("fec", lambda: fetch_fec(f), "fec_money_2010_2026.csv")
    if "fec" in a.only and (RAW / "fec_money_2010_2026.csv").exists() \
            and (RAW / "results_538_1998_2024.csv").exists():
        money = pd.read_csv(RAW / "fec_money_2010_2026.csv")
        res = pd.read_csv(RAW / "results_538_1998_2024.csv")
        noms = match_nominees(money, res)
        noms.to_csv(RAW / "fec_nominees_2010_2024.csv", index=False)
        report["fec nominees"] = f"OK  {len(noms):>8,} race-party rows -> data/raw/fec_nominees_2010_2024.csv"
    run("acs", lambda: fetch_acs(f, a.census_key), f"acs_district_{ACS_VINTAGES[0]}_{ACS_VINTAGES[-1]}.csv")
    run("laus", lambda: fetch_laus(f), "laus_county_annual.csv")
    run("cpi", lambda: fetch_cpi(f), "cpi_u_annual.csv")
    run("zillow", lambda: fetch_zillow(f), "zillow_county_annual.csv")
    run("fhfa", lambda: fetch_fhfa(f), "fhfa_county_hpi.csv")
    run("crosswalk", lambda: fetch_crosswalk(f), "county_cd_relationship_118.csv")
    run("gazetteer", lambda: fetch_gazetteer(f), "cd_land_area.csv")

    print("\n================ summary ================")
    for k, v in report.items():
        print(f"{k:14s} {v}")
    print(MANUAL)
    with open(RAW / "fetch_log.csv", "a", newline="") as fh:
        w = csv.writer(fh)
        for k, v in report.items():
            w.writerow([time.strftime("%Y-%m-%d %H:%M"), k, v])


if __name__ == "__main__":
    main()
