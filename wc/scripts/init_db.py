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
    # Team-level match statistics for WC2026 (from API-Football)
    # xg_home/xg_away: expected goals per team. shots_on_target, possession included
    # for future model enhancements. api_fixture_id links back to API-Football's ID.
    """
    CREATE TABLE IF NOT EXISTS match_stats (
        stat_id         INTEGER PRIMARY KEY AUTOINCREMENT,
        api_fixture_id  INTEGER UNIQUE,
        match_date      TEXT NOT NULL,
        home_team       TEXT NOT NULL,
        away_team       TEXT NOT NULL,
        home_score      INTEGER,
        away_score      INTEGER,
        xg_home         REAL,
        xg_away         REAL,
        shots_home      INTEGER,
        shots_away      INTEGER,
        shots_on_home   INTEGER,
        shots_on_away   INTEGER,
        possession_home INTEGER,
        possession_away INTEGER,
        tournament      TEXT DEFAULT 'FIFA World Cup'
    )
    """,
    # Player events for WC2026 (goals, assists, yellow/red cards, substitutions)
    # Keyed by player name + fixture since we don't have StatsBomb IDs for WC2026 players yet.
    """
    CREATE TABLE IF NOT EXISTS match_events (
        event_id        INTEGER PRIMARY KEY AUTOINCREMENT,
        api_fixture_id  INTEGER NOT NULL,
        match_date      TEXT NOT NULL,
        team_name       TEXT NOT NULL,
        player_name     TEXT NOT NULL,
        event_type      TEXT NOT NULL,
        event_detail    TEXT,
        minute          INTEGER,
        assist_player   TEXT,
        UNIQUE (api_fixture_id, player_name, event_type, minute)
    )
    """,
    # Player appearances (minutes played per WC2026 match)
    # Separate from match_events since appearance data comes from lineups endpoint.
    """
    CREATE TABLE IF NOT EXISTS match_lineups (
        lineup_id       INTEGER PRIMARY KEY AUTOINCREMENT,
        api_fixture_id  INTEGER NOT NULL,
        match_date      TEXT NOT NULL,
        team_name       TEXT NOT NULL,
        player_name     TEXT NOT NULL,
        position        TEXT,
        minutes_played  INTEGER DEFAULT 0,
        is_starter      INTEGER DEFAULT 1,
        UNIQUE (api_fixture_id, player_name)
    )
    """,
    # Computed WC2026 fantasy points per player per match
    # Derived from match_lineups (minutes, position) + match_events (goals/assists/cards)
    # + match_stats (goals conceded for CS / conceded-penalty calculation).
    """
    CREATE TABLE IF NOT EXISTS wc2026_player_points (
        points_id      INTEGER PRIMARY KEY AUTOINCREMENT,
        api_fixture_id INTEGER NOT NULL,
        match_date     TEXT NOT NULL,
        team_name      TEXT NOT NULL,
        player_name    TEXT NOT NULL,
        position       TEXT,
        minutes        INTEGER DEFAULT 0,
        goals          INTEGER DEFAULT 0,
        assists        INTEGER DEFAULT 0,
        yellow_cards   INTEGER DEFAULT 0,
        red_cards      INTEGER DEFAULT 0,
        clean_sheet    INTEGER DEFAULT 0,
        goals_conceded INTEGER DEFAULT 0,
        fantasy_pts    REAL DEFAULT 0.0,
        UNIQUE (api_fixture_id, player_name)
    )
    """,
]

INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_pms_match  ON player_match_stats(match_id)",
    "CREATE INDEX IF NOT EXISTS idx_pms_player ON player_match_stats(player_id)",
    "CREATE INDEX IF NOT EXISTS idx_fix_md     ON fixtures(matchday)",
    "CREATE INDEX IF NOT EXISTS idx_pred_fmd   ON predictions(fantasy_id, matchday)",
    "CREATE INDEX IF NOT EXISTS idx_rr_date    ON recent_results(match_date)",
    "CREATE INDEX IF NOT EXISTS idx_ms_fixture ON match_stats(api_fixture_id)",
    "CREATE INDEX IF NOT EXISTS idx_me_fixture ON match_events(api_fixture_id)",
    "CREATE INDEX IF NOT EXISTS idx_ml_fixture ON match_lineups(api_fixture_id)",
    "CREATE INDEX IF NOT EXISTS idx_me_player  ON match_events(player_name)",
    "CREATE INDEX IF NOT EXISTS idx_ml_player  ON match_lineups(player_name)",
    "CREATE INDEX IF NOT EXISTS idx_pp_fixture ON wc2026_player_points(api_fixture_id)",
    "CREATE INDEX IF NOT EXISTS idx_pp_player  ON wc2026_player_points(player_name)",
    "CREATE INDEX IF NOT EXISTS idx_pp_team    ON wc2026_player_points(team_name)",
]


MIGRATIONS = [
    # match_lineups: bonus stat columns added after initial deploy
    "ALTER TABLE match_lineups ADD COLUMN saves INTEGER DEFAULT 0",
    "ALTER TABLE match_lineups ADD COLUMN shots_on_target INTEGER DEFAULT 0",
    "ALTER TABLE match_lineups ADD COLUMN tackles INTEGER DEFAULT 0",
    "ALTER TABLE match_lineups ADD COLUMN big_chances_created INTEGER DEFAULT 0",
    "ALTER TABLE match_lineups ADD COLUMN penalty_saves INTEGER DEFAULT 0",
    "ALTER TABLE match_lineups ADD COLUMN penalty_conceded INTEGER DEFAULT 0",
    # wc2026_player_points: own_goals + bonus stat columns
    "ALTER TABLE wc2026_player_points ADD COLUMN own_goals INTEGER DEFAULT 0",
    "ALTER TABLE wc2026_player_points ADD COLUMN penalty_conceded INTEGER DEFAULT 0",
    "ALTER TABLE wc2026_player_points ADD COLUMN saves INTEGER DEFAULT 0",
    "ALTER TABLE wc2026_player_points ADD COLUMN shots_on_target INTEGER DEFAULT 0",
    "ALTER TABLE wc2026_player_points ADD COLUMN tackles INTEGER DEFAULT 0",
    "ALTER TABLE wc2026_player_points ADD COLUMN big_chances_created INTEGER DEFAULT 0",
    "ALTER TABLE wc2026_player_points ADD COLUMN penalty_saves INTEGER DEFAULT 0",
]


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    for stmt in SCHEMA:
        conn.execute(stmt)
    for stmt in INDEXES:
        conn.execute(stmt)
    for stmt in MIGRATIONS:
        try:
            conn.execute(stmt)
        except sqlite3.OperationalError:
            pass  # column already exists
    conn.commit()
    conn.close()
    print(f"DB initialised: {DB_PATH}")


if __name__ == "__main__":
    init_db()
