"""
manual_result.py
Manually insert a WC2026 match result when the API daily limit is hit.

Usage:
  python wc/scripts/manual_result.py "Spain" "Cabo Verde" 0 0 2026-06-15
  python wc/scripts/manual_result.py "France" "Germany" 2 1 2026-06-16

The script matches team names against the teams table (case-insensitive).
Only writes to match_stats — no xG/lineup data. DC model trains fine with just scores.

After inserting all results, run:
  python wc/scripts/run_pipeline.py --no-push  (to retrain locally)
  python wc/scripts/run_pipeline.py            (to retrain + push)
"""

import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from wc.scripts.init_db import DB_PATH, init_db


def _find_team(conn: sqlite3.Connection, name: str) -> tuple[int, str] | None:
    rows = conn.execute(
        "SELECT team_id, name FROM teams WHERE LOWER(name) LIKE ?",
        (f"%{name.lower()}%",),
    ).fetchall()
    if len(rows) == 1:
        return rows[0]
    if len(rows) > 1:
        print(f"  [warn] Multiple matches for '{name}': {[r[1] for r in rows]}")
        print(f"  Using: {rows[0][1]}")
        return rows[0]
    return None


def _find_fixture(conn: sqlite3.Connection, home_id: int, away_id: int) -> dict | None:
    row = conn.execute(
        "SELECT fixture_id, kickoff_time FROM fixtures WHERE home_team_id=? AND away_team_id=?",
        (home_id, away_id),
    ).fetchone()
    if row:
        return {"fixture_id": row[0], "kickoff_time": row[1]}
    return None


def insert_result(home_name: str, away_name: str, home_score: int, away_score: int, match_date: str) -> None:
    init_db()
    conn = sqlite3.connect(DB_PATH)

    home = _find_team(conn, home_name)
    away = _find_team(conn, away_name)

    if not home:
        print(f"[error] Team not found: '{home_name}'")
        conn.close()
        return
    if not away:
        print(f"[error] Team not found: '{away_name}'")
        conn.close()
        return

    home_id, home_resolved = home
    away_id, away_resolved = away
    print(f"  Match: {home_resolved} {home_score}-{away_score} {away_resolved}  ({match_date})")

    # Use fixture_id as a stable pseudo-API ID (negative to avoid clashing with real API IDs)
    fixture = _find_fixture(conn, home_id, away_id)
    pseudo_id = -(fixture["fixture_id"]) if fixture else -(hash(f"{home_id}{away_id}{match_date}") % 900000)

    existing = conn.execute(
        "SELECT home_score, away_score FROM match_stats WHERE api_fixture_id=?", (pseudo_id,)
    ).fetchone()
    if existing:
        print(f"  Already exists: {existing[0]}-{existing[1]}. Overwriting...")

    conn.execute(
        """
        INSERT OR REPLACE INTO match_stats
          (api_fixture_id, match_date, home_team, away_team, home_score, away_score)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (pseudo_id, match_date, home_resolved, away_resolved, home_score, away_score),
    )
    conn.commit()
    conn.close()
    print(f"  Inserted (api_fixture_id={pseudo_id}). Run pipeline to retrain.")


def main() -> None:
    if len(sys.argv) != 6:
        print("Usage: python wc/scripts/manual_result.py <home> <away> <home_score> <away_score> <YYYY-MM-DD>")
        print('Example: python wc/scripts/manual_result.py "Spain" "Cabo Verde" 0 0 2026-06-15')
        sys.exit(1)

    home_name  = sys.argv[1]
    away_name  = sys.argv[2]
    home_score = int(sys.argv[3])
    away_score = int(sys.argv[4])
    match_date = sys.argv[5]

    insert_result(home_name, away_name, home_score, away_score, match_date)


if __name__ == "__main__":
    main()
