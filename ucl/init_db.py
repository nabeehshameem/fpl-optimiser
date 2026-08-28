"""
ucl/init_db.py
Creates data/ucl.db with all tables required for UCL analysis.
Idempotent — safe to re-run. Separate from fpl.db (Rule 3: different team IDs).

    python ucl/init_db.py
"""

import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "ucl.db"

SCHEMA = [
    # Teams as returned by API-Football (numeric IDs differ from FPL team IDs).
    # short_name is the 3-letter key used throughout DC model to avoid ID drift.
    """
    CREATE TABLE IF NOT EXISTS teams (
        team_id     INTEGER PRIMARY KEY,
        name        TEXT NOT NULL,
        short_name  TEXT NOT NULL
    )
    """,
    # One row per UCL fixture. round_name covers league phase ("League Phase")
    # and knockout rounds ("Round of 16", "Quarter-finals", etc.).
    """
    CREATE TABLE IF NOT EXISTS fixtures (
        fixture_id      INTEGER PRIMARY KEY,
        round_name      TEXT,
        home_team_id    INTEGER NOT NULL,
        away_team_id    INTEGER NOT NULL,
        kickoff_utc     TEXT,
        status          TEXT,
        home_score      INTEGER,
        away_score      INTEGER,
        fetched_at_utc  TEXT NOT NULL,
        FOREIGN KEY (home_team_id) REFERENCES teams(team_id),
        FOREIGN KEY (away_team_id) REFERENCES teams(team_id)
    )
    """,
    # Per-match stats ingested after FT. One row per fixture.
    # xg_home / xg_away populated where API-Football provides them.
    """
    CREATE TABLE IF NOT EXISTS match_stats (
        fixture_id      INTEGER PRIMARY KEY,
        shots_home      INTEGER,
        shots_away      INTEGER,
        on_target_home  INTEGER,
        on_target_away  INTEGER,
        possession_home INTEGER,
        possession_away INTEGER,
        xg_home         REAL,
        xg_away         REAL,
        fetched_at_utc  TEXT NOT NULL,
        FOREIGN KEY (fixture_id) REFERENCES fixtures(fixture_id)
    )
    """,
    # Committed DC parameters for the UCL model.
    # Mirrors the structure in models/fpl_dc_params.json but stored per-season.
    """
    CREATE TABLE IF NOT EXISTS dc_params (
        season      TEXT NOT NULL,
        team        TEXT NOT NULL,
        attack      REAL NOT NULL,
        defence     REAL NOT NULL,
        PRIMARY KEY (season, team)
    )
    """,
    # League-phase standings (36-team format from 2024/25 onward).
    """
    CREATE TABLE IF NOT EXISTS league_standings (
        team_id     INTEGER NOT NULL,
        season      TEXT NOT NULL,
        played      INTEGER DEFAULT 0,
        won         INTEGER DEFAULT 0,
        drawn       INTEGER DEFAULT 0,
        lost        INTEGER DEFAULT 0,
        goals_for   INTEGER DEFAULT 0,
        goals_against INTEGER DEFAULT 0,
        points      INTEGER DEFAULT 0,
        fetched_at_utc TEXT NOT NULL,
        PRIMARY KEY (team_id, season),
        FOREIGN KEY (team_id) REFERENCES teams(team_id)
    )
    """,
]

INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_fix_round   ON fixtures(round_name)",
    "CREATE INDEX IF NOT EXISTS idx_fix_home     ON fixtures(home_team_id)",
    "CREATE INDEX IF NOT EXISTS idx_fix_away     ON fixtures(away_team_id)",
    "CREATE INDEX IF NOT EXISTS idx_fix_status   ON fixtures(status)",
]


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    cur = conn.cursor()
    for stmt in SCHEMA:
        cur.execute(stmt)
    for idx in INDEXES:
        cur.execute(idx)
    conn.commit()
    conn.close()
    print(f"UCL database initialised at: {DB_PATH}")


if __name__ == "__main__":
    init_db()
