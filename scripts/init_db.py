"""
init_db.py
Creates the SQLite database with all tables required for the FPL optimiser.
Idempotent: safe to run multiple times. Will not overwrite existing data
unless you delete the .db file first.
"""

import sqlite3
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "fpl.db"

# Schema definitions
SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS teams (
        team_id     INTEGER PRIMARY KEY,
        name        TEXT NOT NULL,
        short_name  TEXT NOT NULL,
        strength    INTEGER
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS gameweeks (
        gameweek_id     INTEGER PRIMARY KEY,
        deadline_time   TEXT,
        is_current      INTEGER DEFAULT 0,
        is_next         INTEGER DEFAULT 0,
        finished        INTEGER DEFAULT 0,
        average_score   INTEGER
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS players (
        player_id          INTEGER PRIMARY KEY,
        first_name         TEXT,
        second_name        TEXT,
        web_name           TEXT,
        team_id            INTEGER,
        position           INTEGER,
        current_cost       INTEGER,
        last_updated       TEXT,
        corners_order      INTEGER,
        freekicks_order    INTEGER,
        penalties_order    INTEGER,
        FOREIGN KEY (team_id) REFERENCES teams(team_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS fixtures (
        fixture_id              INTEGER PRIMARY KEY,
        gameweek_id             INTEGER,
        home_team_id            INTEGER NOT NULL,
        away_team_id            INTEGER NOT NULL,
        kickoff_time            TEXT,
        home_team_difficulty    INTEGER,
        away_team_difficulty    INTEGER,
        finished                INTEGER DEFAULT 0,
        home_score              INTEGER,
        away_score              INTEGER,
        FOREIGN KEY (gameweek_id) REFERENCES gameweeks(gameweek_id),
        FOREIGN KEY (home_team_id) REFERENCES teams(team_id),
        FOREIGN KEY (away_team_id) REFERENCES teams(team_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS player_gameweek_history (
        player_id               INTEGER NOT NULL,
        gameweek_id             INTEGER NOT NULL,
        fixture_id              INTEGER,
        minutes                 INTEGER,
        goals_scored            INTEGER,
        assists                 INTEGER,
        clean_sheets            INTEGER,
        total_points            INTEGER,
        bonus                   INTEGER,
        bps                     INTEGER,
        expected_goals          REAL,
        expected_assists        REAL,
        defensive_contribution  INTEGER,
        value                   INTEGER,
        selected                INTEGER,
        PRIMARY KEY (player_id, gameweek_id),
        FOREIGN KEY (player_id) REFERENCES players(player_id),
        FOREIGN KEY (gameweek_id) REFERENCES gameweeks(gameweek_id),
        FOREIGN KEY (fixture_id) REFERENCES fixtures(fixture_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS player_snapshots (
        snapshot_id             INTEGER PRIMARY KEY AUTOINCREMENT,
        player_id               INTEGER NOT NULL,
        gameweek_id             INTEGER NOT NULL,
        snapshot_time           TEXT NOT NULL,
        form                    REAL,
        points_per_game         REAL,
        selected_by_percent     REAL,
        now_cost                INTEGER,
        chance_of_playing_next  INTEGER,
        news                    TEXT,
        ep_next                 REAL,
        FOREIGN KEY (player_id) REFERENCES players(player_id),
        FOREIGN KEY (gameweek_id) REFERENCES gameweeks(gameweek_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS predictions (
        prediction_id     INTEGER PRIMARY KEY AUTOINCREMENT,
        player_id         INTEGER NOT NULL,
        gameweek_id       INTEGER NOT NULL,
        model_name        TEXT NOT NULL,
        predicted_points  REAL,
        prediction_time   TEXT NOT NULL,
        FOREIGN KEY (player_id) REFERENCES players(player_id),
        FOREIGN KEY (gameweek_id) REFERENCES gameweeks(gameweek_id)
    )
    """,
]

# Useful indexes for query performance
INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_pgh_gameweek ON player_gameweek_history(gameweek_id)",
    "CREATE INDEX IF NOT EXISTS idx_pgh_player   ON player_gameweek_history(player_id)",
    "CREATE INDEX IF NOT EXISTS idx_snap_pg      ON player_snapshots(player_id, gameweek_id)",
    "CREATE INDEX IF NOT EXISTS idx_fix_gameweek ON fixtures(gameweek_id)",
    "CREATE INDEX IF NOT EXISTS idx_pred_pgm ON predictions(player_id, gameweek_id, model_name)",
]


def _migrate(conn: sqlite3.Connection) -> None:
    """Add new columns / tables to existing databases."""
    new_player_cols = [
        ("corners_order",   "INTEGER"),
        ("freekicks_order",  "INTEGER"),
        ("penalties_order",  "INTEGER"),
    ]
    for col, col_type in new_player_cols:
        try:
            conn.execute(f"ALTER TABLE players ADD COLUMN {col} {col_type}")
        except sqlite3.OperationalError:
            pass  # column already exists

    # Mini-league tables (idempotent — IF NOT EXISTS)
    mini_league_tables = [
        """
        CREATE TABLE IF NOT EXISTS mini_leagues (
            league_id    INTEGER PRIMARY KEY,
            name         TEXT,
            last_updated TEXT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS league_standings (
            league_id    INTEGER NOT NULL,
            entry_id     INTEGER NOT NULL,
            entry_name   TEXT,
            player_name  TEXT,
            rank         INTEGER,
            last_rank    INTEGER,
            total_points INTEGER,
            event_total  INTEGER,
            fetched_gw   INTEGER,
            PRIMARY KEY (league_id, entry_id),
            FOREIGN KEY (league_id) REFERENCES mini_leagues(league_id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS league_entry_history (
            entry_id               INTEGER NOT NULL,
            gameweek_id            INTEGER NOT NULL,
            points                 INTEGER,
            total_points           INTEGER,
            overall_rank           INTEGER,
            bank                   INTEGER,
            value                  INTEGER,
            event_transfers        INTEGER,
            event_transfers_cost   INTEGER,
            points_on_bench        INTEGER,
            PRIMARY KEY (entry_id, gameweek_id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS league_entry_chips (
            entry_id     INTEGER NOT NULL,
            chip_name    TEXT NOT NULL,
            gameweek_id  INTEGER,
            PRIMARY KEY (entry_id, chip_name)
        )
        """,
    ]
    for stmt in mini_league_tables:
        conn.execute(stmt)


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    cursor = conn.cursor()

    for statement in SCHEMA:
        cursor.execute(statement)

    for statement in INDEXES:
        cursor.execute(statement)

    _migrate(conn)

    conn.commit()
    conn.close()
    print(f"Database initialised at: {DB_PATH}")


if __name__ == "__main__":
    init_db()