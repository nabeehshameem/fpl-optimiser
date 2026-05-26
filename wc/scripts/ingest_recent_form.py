"""
ingest_recent_form.py
Download recent international results and store in `recent_results`.

Source: martj42/international_results (GitHub, free, updated daily)
  https://raw.githubusercontent.com/martj42/international_results/master/results.csv

Columns: date, home_team, away_team, home_score, away_score, tournament, city, country, neutral

The Dixon-Coles model blends this data with WC2018/2022 StatsBomb data so that
predictions reflect current team form, not just 4-8 year old tournament performance.

Usage:
  python wc/scripts/ingest_recent_form.py               # last 24 months (all matches incl. friendlies)
  python wc/scripts/ingest_recent_form.py --months 36   # last 3 years
  python wc/scripts/ingest_recent_form.py --min-importance B  # exclude friendlies
"""

import argparse
import io
import sqlite3
import sys
from datetime import date, timedelta
from pathlib import Path

import requests
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from wc.scripts.init_db import DB_PATH, init_db

RESULTS_CSV_URL = (
    "https://raw.githubusercontent.com/martj42/international_results"
    "/master/results.csv"
)

# Tournament importance tiers — used for weight calculation in the DC model.
# "FIFA World Cup" is excluded here (already in StatsBomb matches table).
TOURNAMENT_TIERS = {
    # Tier A — competitive, high quality
    "FIFA World Cup qualification (CONMEBOL)":    "A",
    "FIFA World Cup qualification (UEFA)":         "A",
    "FIFA World Cup qualification (AFC)":          "A",
    "FIFA World Cup qualification (CAF)":          "A",
    "FIFA World Cup qualification (CONCACAF)":     "A",
    "FIFA World Cup qualification (OFC)":          "A",
    "FIFA World Cup qualification":                "A",
    "UEFA European Championship":                  "A",
    "UEFA European Championship qualification":    "A",
    "Copa América":                                "A",
    "Africa Cup of Nations":                       "A",
    "AFC Asian Cup":                               "A",
    "CONCACAF Gold Cup":                           "A",
    "AFC Asian Cup qualification":                 "B",
    # Tier B — competitive Nations Leagues
    "UEFA Nations League A":                       "B",
    "UEFA Nations League B":                       "B",
    "UEFA Nations League C":                       "B",
    "UEFA Nations League D":                       "B",
    "UEFA Nations League":                         "B",
    "CONCACAF Nations League":                     "B",
    "CAF Africa Cup of Nations qualification":     "B",
    "CONMEBOL-UEFA Cup of Champions":              "B",
    "Intercontinental Playoff":                    "B",
    # Tier C — friendlies / low stakes
    "Friendly":                                    "C",
}

TIER_WEIGHTS = {"A": 0.85, "B": 0.70, "C": 0.35}
MIN_IMPORTANCE = {"A": ("A",), "B": ("A", "B"), "friendly": ("A", "B", "C")}

# Skip WC finals — already have them via StatsBomb with richer event data
SKIP_TOURNAMENTS = {"FIFA World Cup"}


def _fetch_csv() -> pd.DataFrame:
    print(f"Downloading international results from GitHub...")
    resp = requests.get(RESULTS_CSV_URL, timeout=60)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    print(f"  {len(df):,} total matches in dataset.")
    return df


def _filter(df: pd.DataFrame, months: int) -> pd.DataFrame:
    cutoff = date.today() - timedelta(days=months * 30)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df[df["date"].dt.date >= cutoff].copy()
    df = df[~df["tournament"].isin(SKIP_TOURNAMENTS)]
    df = df.dropna(subset=["home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    return df


def ingest(months: int = 24, min_importance: str = "friendly") -> int:
    init_db()
    allowed_tiers = MIN_IMPORTANCE.get(min_importance, ("A", "B"))

    df = _fetch_csv()
    df = _filter(df, months)

    # Keep only matches with at least tier C (all) or filtered by tier
    df["tier"] = df["tournament"].map(TOURNAMENT_TIERS).fillna("C")
    df = df[df["tier"].isin(allowed_tiers)]

    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM recent_results")

    rows = []
    for _, row in df.iterrows():
        rows.append((
            str(row["date"].date()),
            str(row["home_team"]),
            str(row["away_team"]),
            int(row["home_score"]),
            int(row["away_score"]),
            str(row.get("tournament", "")),
            int(bool(row.get("neutral", False))),
        ))

    conn.executemany(
        """
        INSERT INTO recent_results
          (match_date, home_team, away_team, home_score, away_score, tournament, neutral)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    conn.close()

    # Summary by tier
    tier_counts = df["tier"].value_counts()
    print(f"  Stored {len(rows)} matches (last {months} months):")
    for tier in ("A", "B", "C"):
        if tier in tier_counts:
            print(f"    Tier {tier}: {tier_counts[tier]:4d} matches")

    top_teams = (
        pd.concat([df["home_team"], df["away_team"]])
        .value_counts()
        .head(10)
        .index.tolist()
    )
    print(f"  Most active teams: {', '.join(top_teams)}")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest recent international results")
    parser.add_argument("--months", type=int, default=24,
                        help="How many months back to include (default 24)")
    parser.add_argument("--min-importance", choices=["A", "B", "friendly"],
                        default="friendly",
                        help="Minimum tournament tier: A=WCQ/Euros/Copa, "
                             "B=+Nations Leagues, friendly=everything (default friendly)")
    args = parser.parse_args()
    ingest(months=args.months, min_importance=args.min_importance)


if __name__ == "__main__":
    main()
