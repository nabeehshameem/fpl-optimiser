"""
manual_stats.py — Manually patch per-player stats that the API missed or doesn't provide.

Used for:
  --assist       Missing assist (injects into match_events)
  --penalty_won  Penalties won (+2 each, stored in match_lineups)
  --bcc          Big chances created (+1 per 2 for MID, stored in match_lineups)

Usage:
  python wc/scripts/manual_stats.py --fixture 12812994 --player "Cubarsí" --assist
  python wc/scripts/manual_stats.py --fixture 12813016 --player "Mbappé" --penalty_won 1
  python wc/scripts/manual_stats.py --fixture 12813016 --player "Dembélé" --bcc 2

After running, re-run compute to refresh fantasy_pts:
  python wc/scripts/compute_wc2026_points.py
"""

import argparse
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from wc.scripts.init_db import DB_PATH, init_db


def _resolve_player(conn, fixture_id, name_fragment):
    rows = conn.execute(
        "SELECT player_name, team_name, position, match_date FROM match_lineups "
        "WHERE api_fixture_id = ? AND player_name LIKE ?",
        (fixture_id, f"%{name_fragment}%"),
    ).fetchall()
    if not rows:
        print(f"[error] No player matching '{name_fragment}' in fixture {fixture_id}.")
        sys.exit(1)
    if len(rows) > 1:
        print(f"[error] Multiple matches: {[r[0] for r in rows]}. Use a more specific name.")
        sys.exit(1)
    return rows[0]  # player_name, team_name, position, match_date


def main():
    parser = argparse.ArgumentParser(description="Patch missing player stats for a fixture.")
    parser.add_argument("--fixture",     type=int, required=True)
    parser.add_argument("--player",      type=str, required=True)
    parser.add_argument("--assist",      action="store_true", help="Add a missing assist event")
    parser.add_argument("--penalty_won", type=int, default=None)
    parser.add_argument("--bcc",         type=int, default=None)
    args = parser.parse_args()

    if not args.assist and args.penalty_won is None and args.bcc is None:
        parser.error("Specify at least one of --assist, --penalty_won, or --bcc")

    init_db()
    conn = sqlite3.connect(DB_PATH)
    player_name, team, pos, match_date = _resolve_player(conn, args.fixture, args.player)
    parts = []

    # ── Assist: inject into match_events ──────────────────────────────
    if args.assist:
        existing = conn.execute(
            "SELECT COUNT(*) FROM match_events "
            "WHERE api_fixture_id=? AND player_name=? AND event_type='assist'",
            (args.fixture, player_name),
        ).fetchone()[0]
        new_count = existing + 1
        conn.execute(
            """INSERT INTO match_events
               (api_fixture_id, match_date, team_name, player_name,
                event_type, event_detail, minute, assist_player)
               VALUES (?, ?, ?, ?, 'assist', 'manual', NULL, NULL)""",
            (args.fixture, match_date, team, player_name),
        )
        parts.append(f"assist injected (total assists now: {new_count}, +3 pts)")

    # ── match_lineups bonus columns ────────────────────────────────────
    updates, vals = [], []
    if args.penalty_won is not None:
        updates.append("penalty_won = ?")
        vals.append(args.penalty_won)
        parts.append(f"penalty_won={args.penalty_won} (+{args.penalty_won * 2} pts)")
    if args.bcc is not None:
        updates.append("big_chances_created = ?")
        vals.append(args.bcc)
        parts.append(f"big_chances_created={args.bcc} (+{args.bcc // 2} pts)")

    if updates:
        vals += [args.fixture, player_name]
        conn.execute(
            f"UPDATE match_lineups SET {', '.join(updates)} "
            f"WHERE api_fixture_id=? AND player_name=?",
            vals,
        )

    conn.commit()
    conn.close()
    print(f"Patched {player_name} ({team}, {pos}) fixture {args.fixture}: {', '.join(parts)}")
    print("Run compute_wc2026_points.py to apply.")


if __name__ == "__main__":
    main()
