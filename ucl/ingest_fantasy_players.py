"""
ucl/ingest_fantasy_players.py  (B4)
Fetch UCL Fantasy player data from the UEFA Fantasy Football API and write
to data/ucl.db. Used to build a UCL fantasy squad optimiser analogous to FPL.

IMPORTANT: These are UNOFFICIAL endpoints — verify URLs in browser DevTools
before each new season (season ID changes). See ucl/docs/data_sources.md.

    python ucl/ingest_fantasy_players.py                # current season
    python ucl/ingest_fantasy_players.py --season 2026  # explicit

No API key required — the UEFA Fantasy API is unauthenticated.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DB_PATH = PROJECT_ROOT / "data" / "ucl.db"

# Base URL template — {season} is the 4-digit start year (e.g. 2026).
# The "2" in the URL is UEFA's competition ID for the UCL.
# IMPORTANT: Verify these URLs each season before running.
_PARTICIPANTS_URL = (
    "https://gaming.uefa.com/en/uclfantasy/services/feeds/"
    "participants/participants_2_{season}.json"
)
_FIXTURES_URL = (
    "https://gaming.uefa.com/en/uclfantasy/services/feeds/"
    "fixtures/fixtures_2_{season}.json"
)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; fpl-optimiser/1.0)",
    "Accept": "application/json",
}


def _current_ucl_season() -> int:
    now = datetime.now(timezone.utc)
    return now.year if now.month >= 7 else now.year - 1


def _get(url: str) -> dict:
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()


def _ensure_ucl_players_table(conn: sqlite3.Connection) -> None:
    """Create ucl_fantasy_players table if it doesn't exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ucl_fantasy_players (
            player_id       INTEGER PRIMARY KEY,
            first_name      TEXT,
            last_name       TEXT,
            display_name    TEXT,
            team_short_name TEXT,
            position        TEXT,
            price           REAL,
            total_points    INTEGER,
            selected_by_pct REAL,
            status          TEXT,
            fetched_at_utc  TEXT NOT NULL
        )
    """)


def fetch_participants(season: int) -> list[dict]:
    url = _PARTICIPANTS_URL.format(season=season)
    print(f"Fetching UCL Fantasy participants from:\n  {url}")
    try:
        data = _get(url)
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            print(
                f"\n[ERROR] 404 — season {season} URL not found.\n"
                "The UEFA Fantasy API URL format changes each season.\n"
                "Verify the correct URL by opening:\n"
                "  https://gaming.uefa.com/en/uclfantasy\n"
                "and inspecting Network requests in your browser DevTools.\n"
                "Update _PARTICIPANTS_URL in ucl/ingest_fantasy_players.py accordingly."
            )
            sys.exit(1)
        raise
    # UEFA returns {"participants": [...]} or {"data": {"participants": [...]}}
    if "participants" in data:
        return data["participants"]
    return data.get("data", {}).get("participants", [])


def upsert_players(conn: sqlite3.Connection, players: list[dict]) -> None:
    """Normalise UEFA Fantasy player JSON to our schema and upsert."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    # UEFA Fantasy schema (field names vary by season; we probe for common keys):
    POSITION_MAP = {"1": "GK", "2": "DEF", "3": "MID", "4": "FWD"}

    rows = []
    for p in players:
        # Try common key names across seasons
        pid    = p.get("id") or p.get("pId") or p.get("participantId")
        fname  = p.get("firstName") or p.get("first_name") or ""
        lname  = p.get("lastName") or p.get("last_name") or p.get("name") or ""
        dname  = p.get("displayName") or p.get("display_name") or f"{fname} {lname}".strip()
        team   = (p.get("teamShortName") or p.get("team_short_name") or "?")[:5]
        pos_id = str(p.get("positionId") or p.get("position_id") or "0")
        pos    = POSITION_MAP.get(pos_id, "?")
        price  = float(p.get("value") or p.get("price") or 0) / 10.0
        pts    = p.get("totalPoints") or p.get("total_points") or 0
        sel    = float(p.get("selectedByPercent") or p.get("selected_by_pct") or 0)
        status = p.get("status") or "a"
        if pid is None:
            continue
        rows.append((int(pid), fname, lname, dname, team, pos, price, pts, sel, status, now))

    conn.executemany("""
        INSERT INTO ucl_fantasy_players (
            player_id, first_name, last_name, display_name,
            team_short_name, position, price, total_points,
            selected_by_pct, status, fetched_at_utc
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(player_id) DO UPDATE SET
            display_name    = excluded.display_name,
            team_short_name = excluded.team_short_name,
            position        = excluded.position,
            price           = excluded.price,
            total_points    = excluded.total_points,
            selected_by_pct = excluded.selected_by_pct,
            status          = excluded.status,
            fetched_at_utc  = excluded.fetched_at_utc
    """, rows)
    print(f"  Upserted {len(rows)} UCL Fantasy players")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=None)
    args = ap.parse_args()

    season = args.season or _current_ucl_season()
    print(f"=== UCL Fantasy player ingest — season {season}/{season % 100 + 1} ===")

    from ucl.init_db import init_db
    init_db()

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        _ensure_ucl_players_table(conn)
        players = fetch_participants(season)
        print(f"  {len(players)} players returned")
        upsert_players(conn, players)
        conn.commit()
    finally:
        conn.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
