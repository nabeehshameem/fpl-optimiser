"""
ingest_bootstrap.py
Fetches the FPL bootstrap-static endpoint and populates:
  - teams (upsert)
  - gameweeks (upsert)
  - players (upsert)
  - player_snapshots (insert-only, one row per player per run)

Run this every gameweek (before the deadline ideally) to capture state.
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import requests

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "fpl.db"

# --- API ---
BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"


def fetch_bootstrap():
    """Pull the bootstrap-static endpoint and return parsed JSON."""
    print(f"Fetching {BOOTSTRAP_URL}")
    response = requests.get(BOOTSTRAP_URL, timeout=30)
    response.raise_for_status()
    return response.json()


def upsert_teams(cursor, teams):
    """Insert or update each team."""
    sql = """
        INSERT INTO teams (team_id, name, short_name, strength)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(team_id) DO UPDATE SET
            name = excluded.name,
            short_name = excluded.short_name,
            strength = excluded.strength
    """
    rows = [
        (t["id"], t["name"], t["short_name"], t.get("strength"))
        for t in teams
    ]
    cursor.executemany(sql, rows)
    print(f"  Upserted {len(rows)} teams")


def upsert_gameweeks(cursor, events):
    """Insert or update each gameweek."""
    sql = """
        INSERT INTO gameweeks (gameweek_id, deadline_time, is_current, is_next, finished, average_score)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(gameweek_id) DO UPDATE SET
            deadline_time = excluded.deadline_time,
            is_current = excluded.is_current,
            is_next = excluded.is_next,
            finished = excluded.finished,
            average_score = excluded.average_score
    """
    rows = [
        (
            e["id"],
            e.get("deadline_time"),
            int(e.get("is_current", False)),
            int(e.get("is_next", False)),
            int(e.get("finished", False)),
            e.get("average_entry_score"),
        )
        for e in events
    ]
    cursor.executemany(sql, rows)
    print(f"  Upserted {len(rows)} gameweeks")


def upsert_players(cursor, elements, now_iso):
    """Insert or update each player's current dimension data."""
    sql = """
        INSERT INTO players (
            player_id, first_name, second_name, web_name,
            team_id, position, current_cost, last_updated
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(player_id) DO UPDATE SET
            first_name = excluded.first_name,
            second_name = excluded.second_name,
            web_name = excluded.web_name,
            team_id = excluded.team_id,
            position = excluded.position,
            current_cost = excluded.current_cost,
            last_updated = excluded.last_updated
    """
    rows = [
        (
            p["id"],
            p.get("first_name"),
            p.get("second_name"),
            p.get("web_name"),
            p.get("team"),
            p.get("element_type"),
            p.get("now_cost"),
            now_iso,
        )
        for p in elements
    ]
    cursor.executemany(sql, rows)
    print(f"  Upserted {len(rows)} players")


def get_next_gameweek_id(events):
    """Return the gameweek_id of the next upcoming gameweek, or current if none next."""
    for e in events:
        if e.get("is_next"):
            return e["id"]
    for e in events:
        if e.get("is_current"):
            return e["id"]
    # Fallback: first unfinished
    for e in events:
        if not e.get("finished"):
            return e["id"]
    return None


def insert_snapshots(cursor, elements, gameweek_id, now_iso):
    """Append one snapshot row per player for the given gameweek."""
    sql = """
        INSERT INTO player_snapshots (
            player_id, gameweek_id, snapshot_time,
            form, points_per_game, selected_by_percent,
            now_cost, chance_of_playing_next, news, ep_next
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    def to_float(value):
        # FPL returns some numerics as strings (e.g. "5.4"). Coerce safely.
        if value is None or value == "":
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    rows = [
        (
            p["id"],
            gameweek_id,
            now_iso,
            to_float(p.get("form")),
            to_float(p.get("points_per_game")),
            to_float(p.get("selected_by_percent")),
            p.get("now_cost"),
            p.get("chance_of_playing_next_round"),
            p.get("news"),
            to_float(p.get("ep_next")),
        )
        for p in elements
    ]
    cursor.executemany(sql, rows)
    print(f"  Inserted {len(rows)} player snapshots for gameweek {gameweek_id}")


def main():
    data = fetch_bootstrap()

    now_iso = datetime.now(timezone.utc).isoformat()
    next_gw = get_next_gameweek_id(data["events"])
    if next_gw is None:
        raise RuntimeError("No upcoming gameweek found in bootstrap response.")

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    cursor = conn.cursor()

    print("Ingesting:")
    upsert_teams(cursor, data["teams"])
    upsert_gameweeks(cursor, data["events"])
    upsert_players(cursor, data["elements"], now_iso)
    insert_snapshots(cursor, data["elements"], next_gw, now_iso)

    conn.commit()
    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()