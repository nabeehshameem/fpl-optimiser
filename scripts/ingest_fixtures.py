"""
ingest_fixtures.py
Fetches the FPL fixtures endpoint and upserts every fixture into the database.

Run this every gameweek to capture:
  - rescheduling (gameweek_id changes for postponed matches)
  - score updates (post-match)
  - any FDR adjustments
"""

import sqlite3
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "fpl.db"

FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"


def fetch_fixtures():
    print(f"Fetching {FIXTURES_URL}")
    response = requests.get(FIXTURES_URL, timeout=30)
    response.raise_for_status()
    return response.json()


def upsert_fixtures(cursor, fixtures):
    sql = """
        INSERT INTO fixtures (
            fixture_id, gameweek_id, home_team_id, away_team_id,
            kickoff_time, home_team_difficulty, away_team_difficulty,
            finished, home_score, away_score
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(fixture_id) DO UPDATE SET
            gameweek_id = excluded.gameweek_id,
            home_team_id = excluded.home_team_id,
            away_team_id = excluded.away_team_id,
            kickoff_time = excluded.kickoff_time,
            home_team_difficulty = excluded.home_team_difficulty,
            away_team_difficulty = excluded.away_team_difficulty,
            finished = excluded.finished,
            home_score = excluded.home_score,
            away_score = excluded.away_score
    """
    rows = [
        (
            f["id"],
            f.get("event"),  # gameweek_id; can be NULL for postponed
            f["team_h"],
            f["team_a"],
            f.get("kickoff_time"),
            f.get("team_h_difficulty"),
            f.get("team_a_difficulty"),
            int(f.get("finished", False)),
            f.get("team_h_score"),
            f.get("team_a_score"),
        )
        for f in fixtures
    ]
    cursor.executemany(sql, rows)
    print(f"  Upserted {len(rows)} fixtures")


def main():
    fixtures = fetch_fixtures()

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    cursor = conn.cursor()

    print("Ingesting:")
    upsert_fixtures(cursor, fixtures)

    conn.commit()
    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()