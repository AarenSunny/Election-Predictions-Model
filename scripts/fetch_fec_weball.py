"""
Download the FEC "All candidates" summary files (weball{yy}.zip) for 2010-2026 into
data/raw/fec/weball{yy}.txt. Run on your own computer (needs internet + requests):

    python3 scripts/fetch_fec_weball.py

Source: https://www.fec.gov/data/browse-data/?tab=bulk-data  ("All candidates").
Timing caveat: past cycles are full two-year totals (through Dec 31 of the election
year), so they include some post-election money; 2026 is as of the latest reports.
"""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import requests

OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "fec"
YEARS = range(2010, 2027, 2)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ok = 0
    for y in YEARS:
        yy = f"{y % 100:02d}"
        url = f"https://www.fec.gov/files/bulk-downloads/{y}/weball{yy}.zip"
        try:
            r = requests.get(url, timeout=120, headers={"User-Agent": "election-model-research"})
            r.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                name = [n for n in z.namelist() if n.lower().endswith(".txt")][0]
                (OUT / f"weball{yy}.txt").write_bytes(z.read(name))
            print(f"{y}: ok ({len(r.content) // 1024} KB)")
            ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"{y}: FAILED {e}")
    print(f"{ok}/{len(YEARS)} cycles saved to {OUT}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
