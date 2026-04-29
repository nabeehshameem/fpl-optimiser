"""
peek_fixtures.py
Sanity-check the fixtures table after ingest.
"""

import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "fpl.db"


def peek():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    print("Fixtures per gameweek:")
    cursor.execute("""
        SELECT gameweek_id, COUNT(*) AS num_fixtures
        FROM fixtures
        GROUP BY gameweek_id
        ORDER BY gameweek_id
    """)
    for row in cursor.fetchall():
        print(f"  GW {row[0]}: {row[1]} fixtures")

    print("\nNext 5 fixtures (chronologically):")
    cursor.execute("""
        SELECT
            f.gameweek_id,
            f.kickoff_time,
            home.short_name AS home,
            away.short_name AS away,
            f.home_team_difficulty AS h_fdr,
            f.away_team_difficulty AS a_fdr
        FROM fixtures f
        JOIN teams home ON f.home_team_id = home.team_id
        JOIN teams away ON f.away_team_id = away.team_id
        WHERE f.finished = 0
        ORDER BY f.kickoff_time
        LIMIT 5
    """)
    for row in cursor.fetchall():
        print(f"  GW{row[0]} {row[1]}: {row[2]} (FDR {row[4]}) vs {row[3]} (FDR {row[5]})")

    conn.close()


if __name__ == "__main__":
    peek()