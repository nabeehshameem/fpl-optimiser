"""
init_db.py — World Cup Fantasy DB initialisation.

Run once before any ingest scripts. Safe to re-run (IF NOT EXISTS throughout).
"""

import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "wc.db"

SCHEMA = [
    # National teams — populated by ingest_sb.py and ingest_fixtures.py
    """
    CREATE TABLE IF NOT EXISTS teams (
        team_id    INTEGER PRIMARY KEY,
        name       TEXT NOT NULL,
        short_name TEXT,
        fifa_rank  INTEGER,
        group_id   TEXT,
        source     TEXT DEFAULT 'statsbomb'
    )
    """,
    # Historical + upcoming matches
    """
    CREATE TABLE IF NOT EXISTS matches (
        match_id     INTEGER PRIMARY KEY,
        tournament   TEXT NOT NULL,
        stage        TEXT,
        matchday     INTEGER,
        home_team_id INTEGER,
        away_team_id INTEGER,
        home_score   INTEGER,
        away_score   INTEGER,
        match_date   TEXT,
        FOREIGN KEY (home_team_id) REFERENCES teams(team_id),
        FOREIGN KEY (away_team_id) REFERENCES teams(team_id)
    )
    """,
    # Players (StatsBomb IDs as primary key for training data)
    """
    CREATE TABLE IF NOT EXISTS players (
        player_id   INTEGER PRIMARY KEY,
        name        TEXT NOT NULL,
        name_norm   TEXT,
        team_id     INTEGER,
        position    TEXT,
        nationality TEXT,
        FOREIGN KEY (team_id) REFERENCES teams(team_id)
    )
    """,
    # Per-player per-match stats (from StatsBomb events)
    """
    CREATE TABLE IF NOT EXISTS player_match_stats (
        player_id    INTEGER NOT NULL,
        match_id     INTEGER NOT NULL,
        team_id      INTEGER,
        minutes      INTEGER DEFAULT 0,
        goals        INTEGER DEFAULT 0,
        assists      INTEGER DEFAULT 0,
        xg           REAL DEFAULT 0.0,
        xa           REAL DEFAULT 0.0,
        shots        INTEGER DEFAULT 0,
        key_passes   INTEGER DEFAULT 0,
        yellow_cards INTEGER DEFAULT 0,
        red_cards    INTEGER DEFAULT 0,
        saves        INTEGER DEFAULT 0,
        clean_sheet  INTEGER DEFAULT 0,
        fantasy_pts  REAL DEFAULT 0.0,
        PRIMARY KEY (player_id, match_id),
        FOREIGN KEY (player_id) REFERENCES players(player_id),
        FOREIGN KEY (match_id) REFERENCES matches(match_id)
    )
    """,
    # Fantasy game player roster (once the WC Fantasy game launches)
    """
    CREATE TABLE IF NOT EXISTS fantasy_players (
        fantasy_id   INTEGER PRIMARY KEY,
        player_id    INTEGER,
        name         TEXT NOT NULL,
        name_norm    TEXT,
        team_id      INTEGER,
        position     TEXT,
        price        INTEGER,
        ownership    REAL DEFAULT 0.0,
        FOREIGN KEY (player_id) REFERENCES players(player_id),
        FOREIGN KEY (team_id) REFERENCES teams(team_id)
    )
    """,
    # Upcoming matchday fixtures
    """
    CREATE TABLE IF NOT EXISTS fixtures (
        fixture_id   INTEGER PRIMARY KEY AUTOINCREMENT,
        matchday     INTEGER NOT NULL,
        home_team_id INTEGER NOT NULL,
        away_team_id INTEGER NOT NULL,
        kickoff_time TEXT,
        group_id     TEXT,
        FOREIGN KEY (home_team_id) REFERENCES teams(team_id),
        FOREIGN KEY (away_team_id) REFERENCES teams(team_id)
    )
    """,
    # Qualifying campaign stats (FBRef — used as pre-tournament features)
    """
    CREATE TABLE IF NOT EXISTS qualifying_stats (
        player_id    INTEGER NOT NULL,
        tournament   TEXT NOT NULL,
        matches      INTEGER DEFAULT 0,
        minutes      INTEGER DEFAULT 0,
        goals        INTEGER DEFAULT 0,
        assists      INTEGER DEFAULT 0,
        xg           REAL DEFAULT 0.0,
        xa           REAL DEFAULT 0.0,
        shots        INTEGER DEFAULT 0,
        PRIMARY KEY (player_id, tournament),
        FOREIGN KEY (player_id) REFERENCES players(player_id)
    )
    """,
    # Ownership snapshots
    """
    CREATE TABLE IF NOT EXISTS ownership_snapshots (
        snapshot_id   INTEGER PRIMARY KEY AUTOINCREMENT,
        fantasy_id    INTEGER NOT NULL,
        matchday      INTEGER,
        snapshot_time TEXT,
        ownership     REAL,
        price         INTEGER,
        FOREIGN KEY (fantasy_id) REFERENCES fantasy_players(fantasy_id)
    )
    """,
    # Predictions
    """
    CREATE TABLE IF NOT EXISTS predictions (
        prediction_id   INTEGER PRIMARY KEY AUTOINCREMENT,
        fantasy_id      INTEGER NOT NULL,
        matchday        INTEGER NOT NULL,
        predicted_pts   REAL,
        model_version   TEXT,
        prediction_time TEXT,
        FOREIGN KEY (fantasy_id) REFERENCES fantasy_players(fantasy_id)
    )
    """,
    # Recent international results (last ~2 years from martj42/international_results)
    # Used to give the Dixon-Coles model current team form beyond WC2018/2022 history.
    """
    CREATE TABLE IF NOT EXISTS recent_results (
        result_id  INTEGER PRIMARY KEY AUTOINCREMENT,
        match_date TEXT NOT NULL,
        home_team  TEXT NOT NULL,
        away_team  TEXT NOT NULL,
        home_score INTEGER NOT NULL,
        away_score INTEGER NOT NULL,
        tournament TEXT DEFAULT '',
        neutral    INTEGER DEFAULT 0
    )
    """,
]

INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_pms_match  ON player_match_stats(match_id)",
    "CREATE INDEX IF NOT EXISTS idx_pms_player ON player_match_stats(player_id)",
    "CREATE INDEX IF NOT EXISTS idx_fix_md     ON fixtures(matchday)",
    "CREATE INDEX IF NOT EXISTS idx_pred_fmd   ON predictions(fantasy_id, matchday)",
    "CREATE INDEX IF NOT EXISTS idx_rr_date    ON recent_results(match_date)",
]


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    for stmt in SCHEMA:
        conn.execute(stmt)
    for stmt in INDEXES:
        conn.execute(stmt)
    conn.commit()
    conn.close()
    print(f"DB initialised: {DB_PATH}")


if __name__ == "__main__":
    init_db()
