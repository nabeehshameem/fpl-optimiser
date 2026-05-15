"""
analyse_mini_league.py
Print standings, GW rank history, and chips remaining for a mini-league.

Usage:
  python scripts/analyse_mini_league.py --league 123456
  python scripts/analyse_mini_league.py --league 123456 --history-gws 8
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.mini_league import league_summary, rank_history_table, chips_remaining


def _header(title: str) -> None:
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)


def main():
    parser = argparse.ArgumentParser(description="FPL mini-league analysis")
    parser.add_argument("--league", type=str, required=True,
                        help="League ID to analyse")
    parser.add_argument("--history-gws", type=int, default=6,
                        help="Recent GWs to show in rank history (default 6)")
    args = parser.parse_args()

    league_id = int(args.league.strip())

    _header(f"LEAGUE {league_id} — STANDINGS")
    try:
        summary = league_summary(league_id)
        print(summary.to_string(index=False))
    except ValueError as e:
        print(f"  [error] {e}")
        sys.exit(1)

    _header(f"RANK HISTORY  (last {args.history_gws} GWs, within-league position)")
    try:
        hist = rank_history_table(league_id, recent_gws=args.history_gws)
        print(hist.to_string(index=False))
    except Exception as e:
        print(f"  [error] {e}")

    _header("CHIPS REMAINING")
    try:
        cr = chips_remaining(league_id)
        print(cr.to_string(index=False))
    except Exception as e:
        print(f"  [error] {e}")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
