"""
download_data.py
----------------
Run this file ONCE to download all raw data files.
Place this file in your project root (same level as src/).

Run with:  python download_data.py
"""

import urllib.request
from pathlib import Path

# Create the folder if it does not exist yet
Path("data/raw").mkdir(parents=True, exist_ok=True)

# Files to download — {filename we want : url to fetch from}
FILES = {
    "results.csv":     "https://raw.githubusercontent.com/jfjelstul/worldcup/master/data-csv/matches.csv",
    "goals.csv":       "https://raw.githubusercontent.com/jfjelstul/worldcup/master/data-csv/goals.csv",
    "teams.csv":       "https://raw.githubusercontent.com/jfjelstul/worldcup/master/data-csv/teams.csv",
    "tournaments.csv": "https://raw.githubusercontent.com/jfjelstul/worldcup/master/data-csv/tournaments.csv",
}

print("\n FIFA WC Optimizer — Downloading raw data")
print("=" * 45)

for filename, url in FILES.items():
    dest = Path("data/raw") / filename
    print(f"  Downloading {filename} ...", end=" ")
    try:
        urllib.request.urlretrieve(url, dest)
        size_kb = dest.stat().st_size // 1024
        print(f"done  ({size_kb} KB)")
    except Exception as e:
        print(f"FAILED\n  Error: {e}")

print("=" * 45)
print("  All files saved to data/raw/")
print("  Next step: run pipeline.py\n")