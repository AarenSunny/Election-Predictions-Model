"""
scripts/convert_cook_pvi.py on a tiny hand-made workbook shaped like Cook's
(old layout: State / District / PVI / Raw PVI; 2026 layout: Dist "AL-01").
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("cook", ROOT / "scripts" / "convert_cook_pvi.py")
cook = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cook)


def test_parse_pvi_and_district():
    assert cook.parse_pvi("R+12.07") == -12.07
    assert cook.parse_pvi("D+5") == 5.0
    assert cook.parse_pvi("EVEN") == 0.0
    with pytest.raises(ValueError):
        cook.parse_pvi("12")
    assert cook.parse_district("AL") == 0 and cook.parse_district("7") == 7


def _old_sheet(rows):
    return pd.DataFrame(rows, columns=["State", "District", "PVI", "Raw PVI"])


def test_build_house_lean(tmp_path):
    old = _old_sheet([["Pennsylvania", "7", "R+1", "R+0.6"], ["Pennsylvania", "15", "R+5", "R+4.6"],
                      ["Wyoming", "AL", "R+25", "R+25.0"], ["North Carolina", "1", "D+9", "D+9.0"]])
    new_pa = _old_sheet([["Pennsylvania", "7", "D+1", "D+1.0"], ["Pennsylvania", "15", "R+21", "R+21.6"],
                         ["Wyoming", "AL", "R+25", "R+25.0"], ["North Carolina", "1", "D+9", "D+9.0"]])
    y2026 = pd.DataFrame({"Dist": ["PA-07", "PA-08", "WY-AL", "NC-01"],
                          "PVI": ["EVEN", "R+8", "R+23", "D+3"]})
    path = tmp_path / "cook.xlsx"
    with pd.ExcelWriter(path) as w:
        sheets = (set(cook.SHEET_FOR_YEAR.values()) | set(cook.OVERRIDES.values())
                  | set(cook.SPECIAL_SHEET.values()))
        for name in sheets:
            (y2026 if name.startswith("2026") else new_pa if name in ("116th (19-20)", "108th (03-04)")
             else old) \
                .to_excel(w, sheet_name=name, index=False)
    h = cook.build_house_lean(path).set_index(["year", "state", "district", "special"])["lean"]

    assert np.isclose(h[(2018, "PA", 15, False)], -0.432)   # new PA map for regular races
    assert np.isclose(h[(2018, "PA", 15, True)], -0.092)    # old map for the 2018 special
    assert np.isclose(h[(2014, "WY", 0, False)], -0.50)     # Raw column, x2/100, at-large -> 0
    assert np.isclose(h[(2026, "PA", 7, False)], 0.0)       # 2026 layout
    # (NC has one district in this toy sheet, so it's district 0)
    assert (2020, "NC", 0, False) not in h.index            # redistricted state-year skipped
    assert (2018, "NC", 0, False) in h.index
    # 2002: regular races on the new map (108th), specials on the old one (107th)
    assert np.isclose(h[(2002, "PA", 15, False)], -0.432)
    assert np.isclose(h[(2002, "PA", 15, True)], -0.092)
    assert (1998, "WY", 0, False) in h.index                 # 1998-2008 sheets included


def test_build_senate_lean(tmp_path):
    rows = []
    for yr, pa in ((1997, "R+0.5"), (2001, "D+2.0"), (2025, "R+1.0")):
        for name, code in cook.STATE_CODES.items():
            label = "Nebraska †" if name == "Nebraska" else name     # dagger as in the file
            rows.append([yr, label, "x", pa if code == "PA" else "EVEN", 1])
        rows.append([yr, "D.C.", "D+40", "D+40.0", 1])                  # dropped
    path = tmp_path / "senate.xlsx"
    pd.DataFrame(rows).to_excel(path, index=False, header=False)
    s = cook.build_senate_lean(path).set_index(["year", "state", "special"])["lean"]
    assert np.isclose(s[(1998, "PA", False)], -0.01) and np.isclose(s[(2000, "PA", True)], -0.01)
    assert np.isclose(s[(2002, "PA", False)], 0.04)        # 2001 block -> 2002 ... 2024
    assert np.isclose(s[(2024, "PA", False)], 0.04)
    assert np.isclose(s[(2026, "PA", False)], -0.02)       # 2025 block -> 2026
    assert (2026, "NE", False) in s.index and "DC" not in s.index.get_level_values("state")
