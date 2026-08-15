"""
ucl/ingest_fixtures.py
Fetch all UCL fixtures and teams for a season from API-Football and write
them to data/ucl.db. Run once at the start of each season and again after
any rescheduling.

    python ucl/ingest_fixtures.py                # current season (inferred)
    python ucl/ingest_fixtures.py --season 2026  # explicit: 2026/27 UCL

Requires RAPIDAPI_KEY in environment (or .env at project root).
API-Football league ID for UCL: 2.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DB_PATH = PROJECT_ROOT / "data" / "ucl.db"
UCL_LEAGUE_ID = 2
RAPIDAPI_HOST = "api-football-v1.p.rapidapi.com"


def _load_env() -> None:
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())


def _headers() -> dict:
    key = os.environ.get("RAPIDAPI_KEY")
    if not key:
        raise RuntimeError(
            "RAPIDAPI_KEY not set — add it to .env or the environment."
        )
    return {
        "X-RapidAPI-Key": key,
        "X-RapidAPI-Host": RAPIDAPI_HOST,
    }


def _get(path: str, params: dict) -> dict:
    url = f"https://{RAPIDAPI_HOST}{path}"
    r = requests.get(url, headers=_headers(), params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    errors = data.get("errors")
    if errors:
        raise RuntimeError(f"API-Football error: {errors}")
    return data


def _current_ucl_season() -> int:
    """Infer the UCL season from the current date.
    UCL 2026/27 starts in the latter half of 2026, so season = current_year
    when month >= 7, else current_year - 1.
    """
    now = datetime.now(timezone.utc)
    return now.year if now.month >= 7 else now.year - 1


# ── team ingest ───────────────────────────────────────────────────────────────

def fetch_teams(season: int) -> list[dict]:
    print(f"Fetching UCL teams for season {season}…")
    data = _get("/teams", {"league": UCL_LEAGUE_ID, "season": season})
    teams = data.get("response", [])
    print(f"  {len(teams)} teams returned")
    return teams


def upsert_teams(conn: sqlite3.Connection, teams: list[dict]) -> None:
    sql = """
        INSERT INTO teams (team_id, name, short_name)
        VALUES (?, ?, ?)
        ON CONFLICT(team_id) DO UPDATE SET
            name = excluded.name,
            short_name = excluded.short_name
    """
    # API-Football returns {"team": {"id":..., "name":..., "code":...}, ...}
    # "code" is the 3-letter code (e.g. "MCI", "REA"); use it as short_name.
    rows = [
        (
            t["team"]["id"],
            t["team"]["name"],
            t["team"].get("code") or t["team"]["name"][:3].upper(),
        )
        for t in teams
    ]
    conn.executemany(sql, rows)
    print(f"  Upserted {len(rows)} teams")


# ── fixture ingest ────────────────────────────────────────────────────────────

def fetch_fixtures(season: int) -> list[dict]:
    print(f"Fetching UCL fixtures for season {season}…")
    data = _get("/fixtures", {"league": UCL_LEAGUE_ID, "season": season})
    fixtures = data.get("response", [])
    print(f"  {len(fixtures)} fixtures returned")
    return fixtures


def upsert_fixtures(conn: sqlite3.Connection, fixtures: list[dict]) -> None:
    sql = """
        INSERT INTO fixtures (
            fixture_id, round_name,
            home_team_id, away_team_id,
            kickoff_utc, status,
            home_score, away_score, fetched_at_utc
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(fixture_id) DO UPDATE SET
            round_name   = excluded.round_name,
            kickoff_utc  = excluded.kickoff_utc,
            status       = excluded.status,
            home_score   = excluded.home_score,
            away_score   = excluded.away_score,
            fetched_at_utc = excluded.fetched_at_utc
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = []
    for f in fixtures:
        fix = f.get("fixture", {})
        teams = f.get("teams", {})
        goals = f.get("goals", {})
        rows.append((
            fix["id"],
            f.get("league", {}).get("round"),
            teams.get("home", {}).get("id"),
            teams.get("away", {}).get("id"),
            fix.get("date"),
            fix.get("status", {}).get("short"),
            goals.get("home"),
            goals.get("away"),
            now,
        ))
    conn.executemany(sql, rows)
    print(f"  Upserted {len(rows)} fixtures")


# ── standings ingest ──────────────────────────────────────────────────────────

def fetch_standings(season: int) -> list[dict]:
    print(f"Fetching UCL standings for season {season}…")
    data = _get("/standings", {"league": UCL_LEAGUE_ID, "season": season})
    # API-Football nests standings as response[0].league.standings[0][...]
    try:
        groups = data["response"][0]["league"]["standings"]
        # UCL league phase is a single 36-team group → standings[0]
        return groups[0] if groups else []
    except (IndexError, KeyError):
        print("  No standings available yet (pre-season?)")
        return []


def upsert_standings(
    conn: sqlite3.Connection, rows: list[dict], season: int
) -> None:
    sql = """
        INSERT INTO league_standings (
            team_id, season, played, won, drawn, lost,
            goals_for, goals_against, points, fetched_at_utc
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(team_id, season) DO UPDATE SET
            played        = excluded.played,
            won           = excluded.won,
            drawn         = excluded.drawn,
            lost          = excluded.lost,
            goals_for     = excluded.goals_for,
            goals_against = excluded.goals_against,
            points        = excluded.points,
            fetched_at_utc = excluded.fetched_at_utc
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    db_rows = [
        (
            r["team"]["id"],
            str(season),
            r["all"]["played"],
            r["all"]["win"],
            r["all"]["draw"],
            r["all"]["lose"],
            r["all"]["goals"]["for"],
            r["all"]["goals"]["against"],
            r["points"],
            now,
        )
        for r in rows
    ]
    if db_rows:
        conn.executemany(sql, db_rows)
        print(f"  Upserted {len(db_rows)} standing rows")
    else:
        print("  No standing rows to upsert")


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    _load_env()

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--season", type=int, default=None,
        help="UCL season start year (e.g. 2026 for 2026/27). Default: inferred.",
    )
    ap.add_argument(
        "--skip-standings", action="store_true",
        help="Skip standings ingest (pre-season, before any results).",
    )
    args = ap.parse_args()

    season = args.season or _current_ucl_season()
    print(f"=== UCL ingest — season {season}/{season % 100 + 1} ===")

    # Ensure schema exists (idempotent)
    from ucl.init_db import init_db
    init_db()

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        teams = fetch_teams(season)
        upsert_teams(conn, teams)
        conn.commit()

        fixtures = fetch_fixtures(season)
        upsert_fixtures(conn, fixtures)
        conn.commit()

        if not args.skip_standings:
            standing_rows = fetch_standings(season)
            upsert_standings(conn, standing_rows, season)
            conn.commit()
    finally:
        conn.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
