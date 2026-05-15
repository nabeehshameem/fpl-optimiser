"""
ingest_mini_league.py
Fetch and cache mini-league standings + entry histories.

Usage:
  python scripts/ingest_mini_league.py --league 123456
  python scripts/ingest_mini_league.py --league 123456,789012
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.mini_league import ingest_league


def main():
    parser = argparse.ArgumentParser(description="Ingest FPL mini-league data")
    parser.add_argument("--league", type=str, required=True,
                        help="Comma-separated league ID(s) to ingest")
    args = parser.parse_args()

    league_ids = [int(x.strip()) for x in args.league.split(",") if x.strip()]
    for lid in league_ids:
        print(f"\nIngesting league {lid}...")
        ingest_league(lid)
    print("\nDone.")


if __name__ == "__main__":
    main()
