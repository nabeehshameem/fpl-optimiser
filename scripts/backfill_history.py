"""
backfill_history.py
For every player in the players table, fetches their per-gameweek history
from the FPL element-summary endpoint and upserts into player_gameweek_history.

Slow (~5-10 minutes for ~830 players). Idempotent: safe to re-run.
"""

import sqlite3
import time
from pathlib import Path

import requests
from requests.exceptions import RequestException

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "fpl.db"

ELEMENT_SUMMARY_URL = "https://fantasy.premierleague.com/api/element-summary/{player_id}/"

# Politeness: sleep this many seconds between API calls
REQUEST_DELAY_SECONDS = 0.3
# Retry settings
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2.0


def fetch_player_history(player_id):
    """Fetch element-summary for one player, with retry on transient failures."""
    url = ELEMENT_SUMMARY_URL.format(player_id=player_id)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.json()
        except RequestException as e:
            if attempt == MAX_RETRIES:
                print(f"  [FAIL] player {player_id}: {e}")
                return None
            sleep_s = RETRY_BACKOFF_SECONDS * attempt
            print(f"  [retry {attempt}] player {player_id}: {e} (sleeping {sleep_s}s)")
            time.sleep(sleep_s)
    return None


def upsert_history_rows(cursor, player_id, history):
    """Re-insert all gameweek history rows for one player.

    Strategy: DELETE all existing rows for this player first, then INSERT fresh.
    This is idempotent for re-runs AND correctly handles DGW weeks: the API returns
    two fixture entries for the same gameweek_id, and the second INSERT will hit
    ON CONFLICT and ADD stats to the first — giving the correct per-GW aggregate.
    """
    cursor.execute(
        "DELETE FROM player_gameweek_history WHERE player_id = ?", (player_id,)
    )

    insert_sql = """
        INSERT INTO player_gameweek_history (
            player_id, gameweek_id, fixture_id,
            minutes, goals_scored, assists, clean_sheets, total_points,
            bonus, bps, expected_goals, expected_assists,
            defensive_contribution, value, selected
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(player_id, gameweek_id) DO UPDATE SET
            fixture_id      = excluded.fixture_id,
            minutes         = minutes         + excluded.minutes,
            goals_scored    = goals_scored    + excluded.goals_scored,
            assists         = assists         + excluded.assists,
            clean_sheets    = clean_sheets    + excluded.clean_sheets,
            total_points    = total_points    + excluded.total_points,
            bonus           = bonus           + excluded.bonus,
            bps             = bps             + excluded.bps,
            expected_goals  = expected_goals  + excluded.expected_goals,
            expected_assists= expected_assists+ excluded.expected_assists,
            defensive_contribution = defensive_contribution + excluded.defensive_contribution,
            value    = excluded.value,
            selected = excluded.selected
    """

    def to_float(value):
        if value is None or value == "":
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    rows = [
        (
            player_id,
            h.get("round"),  # gameweek
            h.get("fixture"),
            h.get("minutes"),
            h.get("goals_scored"),
            h.get("assists"),
            h.get("clean_sheets"),
            h.get("total_points"),
            h.get("bonus"),
            h.get("bps"),
            to_float(h.get("expected_goals")),
            to_float(h.get("expected_assists")),
            h.get("defensive_contribution"),
            h.get("value"),
            h.get("selected"),
        )
        for h in history
    ]
    if rows:
        cursor.executemany(insert_sql, rows)
    return len(rows)


def main():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    cursor = conn.cursor()

    cursor.execute("SELECT player_id FROM players ORDER BY player_id")
    player_ids = [row[0] for row in cursor.fetchall()]
    total = len(player_ids)
    print(f"Backfilling history for {total} players. ETA ~{int(total * (REQUEST_DELAY_SECONDS + 0.2) / 60)} min.")

    total_rows_written = 0
    failed = 0

    for i, player_id in enumerate(player_ids, start=1):
        data = fetch_player_history(player_id)
        if data is None:
            failed += 1
            continue

        history = data.get("history", [])
        n = upsert_history_rows(cursor, player_id, history)
        total_rows_written += n

        if i % 50 == 0 or i == total:
            conn.commit()  # commit periodically so you don't lose progress on crash
            print(f"  Progress: {i}/{total} players, {total_rows_written} history rows written, {failed} failed")

        time.sleep(REQUEST_DELAY_SECONDS)

    conn.commit()
    conn.close()
    print(f"Done. {total_rows_written} history rows written across {total - failed} players ({failed} failed).")


if __name__ == "__main__":
    main()