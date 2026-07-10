"""
manual_stats.py — Manually set per-player bonus stats not available from the API.

Used for: penalty_won (+2 each), big_chances_created (+1 per 2 for MID).

Usage:
  python wc/scripts/manual_stats.py --fixture 12813016 --player "Kylian Mbappé" --penalty_won 1
  python wc/scripts/manual_stats.py --fixture 12813016 --player "Ousmane Dembélé" --bcc 2

After running, re-run compute to refresh fantasy_pts:
  python wc/scripts/compute_wc2026_points.py

Or just run the full pipeline (run_pipeline.py handles this automatically for new matches).
"""

import argparse
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from wc.scripts.init_db import DB_PATH, init_db


def main():
    parser = argparse.ArgumentParser(description="Set manual bonus stats for a player in a fixture.")
    parser.add_argument("--fixture", type=int, required=True, help="api_fixture_id")
    parser.add_argument("--player",  type=str, required=True, help="Player name (exact match or partial)")
    parser.add_argument("--penalty_won", type=int, default=None, help="Penalties won")
    parser.add_argument("--bcc",         type=int, default=None, help="Big chances created")
    args = parser.parse_args()

    if args.penalty_won is None and args.bcc is None:
        parser.error("Specify at least one of --penalty_won or --bcc")

    init_db()
    conn = sqlite3.connect(DB_PATH)

    # Fuzzy-match player name in the fixture
    rows = conn.execute(
        "SELECT player_name, team_name, position FROM match_lineups "
        "WHERE api_fixture_id = ? AND player_name LIKE ?",
        (args.fixture, f"%{args.player}%"),
    ).fetchall()

    if not rows:
        print(f"[error] No player matching '{args.player}' in fixture {args.fixture}.")
        conn.close()
        sys.exit(1)
    if len(rows) > 1:
        print(f"[error] Multiple matches for '{args.player}': {[r[0] for r in rows]}")
        print("Use a more specific name.")
        conn.close()
        sys.exit(1)

    player_name, team, pos = rows[0]
    updates = []
    vals    = []
    if args.penalty_won is not None:
        updates.append("penalty_won = ?")
        vals.append(args.penalty_won)
    if args.bcc is not None:
        updates.append("big_chances_created = ?")
        vals.append(args.bcc)

    vals += [args.fixture, player_name]
    conn.execute(
        f"UPDATE match_lineups SET {', '.join(updates)} WHERE api_fixture_id = ? AND player_name = ?",
        vals,
    )
    conn.commit()
    conn.close()

    parts = []
    if args.penalty_won is not None:
        parts.append(f"penalty_won={args.penalty_won} (+{args.penalty_won * 2} pts)")
    if args.bcc is not None:
        parts.append(f"big_chances_created={args.bcc} (+{args.bcc // 2} pts)")
    print(f"Updated {player_name} ({team}, {pos}) in fixture {args.fixture}: {', '.join(parts)}")
    print("Run compute_wc2026_points.py to apply.")


if __name__ == "__main__":
    main()
