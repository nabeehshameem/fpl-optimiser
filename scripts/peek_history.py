"""
peek_history.py
Sanity-check the player_gameweek_history table after backfill.
"""

import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "fpl.db"


def peek():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    print("Total history rows per gameweek:")
    cursor.execute("""
        SELECT gameweek_id, COUNT(*) AS rows
        FROM player_gameweek_history
        GROUP BY gameweek_id
        ORDER BY gameweek_id
    """)
    for row in cursor.fetchall():
        print(f"  GW {row[0]}: {row[1]} player-rows")

    print("\nTop 10 single-gameweek hauls this season:")
    cursor.execute("""
        SELECT
            p.web_name,
            t.short_name AS team,
            h.gameweek_id,
            h.minutes,
            h.goals_scored,
            h.assists,
            h.total_points
        FROM player_gameweek_history h
        JOIN players p ON h.player_id = p.player_id
        JOIN teams t ON p.team_id = t.team_id
        ORDER BY h.total_points DESC
        LIMIT 10
    """)
    for row in cursor.fetchall():
        print(f"  {row[0]:20s} ({row[1]}) GW{row[2]:2d}: {row[3]} min, {row[4]}g {row[5]}a -> {row[6]} pts")

    conn.close()


if __name__ == "__main__":
    peek()