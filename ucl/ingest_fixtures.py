"""
ucl/ingest_fixtures.py
Fetch UCL teams and fixtures for a season from SportAPI7 (SofaScore) and
write them to data/ucl.db.

    python ucl/ingest_fixtures.py                # current season (inferred)
    python ucl/ingest_fixtures.py --season 2026  # explicit: 2026/27 UCL

Requires RAPIDAPI_KEY in environment (or .env at project root).
SportAPI7 tournament ID for UCL: 7.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DB_PATH = PROJECT_ROOT / "data" / "ucl.db"
UCL_TOURNAMENT_ID = 7
RAPIDAPI_HOST = "sportapi7.p.rapidapi.com"
API_BASE = f"https://{RAPIDAPI_HOST}/api/v1"


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
        raise RuntimeError("RAPIDAPI_KEY not set — add it to .env or the environment.")
    return {
        "X-RapidAPI-Key": key,
        "X-RapidAPI-Host": RAPIDAPI_HOST,
        "Content-Type": "application/json",
    }


def _get(path: str, retries: int = 4) -> dict:
    for attempt in range(retries):
        r = requests.get(f"{API_BASE}/{path}", headers=_headers(), timeout=30)
        if r.status_code == 429:
            wait = 15 * (2 ** attempt)
            print(f"  [rate-limited] waiting {wait}s before retry {attempt + 1}/{retries}…")
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"Still rate-limited after {retries} retries for {path}")


def _current_ucl_season() -> int:
    now = datetime.now(timezone.utc)
    return now.year if now.month >= 7 else now.year - 1


def _ts_to_utc(ts: int | None) -> str | None:
    if not ts:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── season lookup ─────────────────────────────────────────────────────────────

def find_season_id(season_year: int) -> int:
    """Return SportAPI7 season ID for the UCL season starting in season_year."""
    data = _get(f"unique-tournament/{UCL_TOURNAMENT_ID}/seasons")
    target = f"{season_year % 100}/{season_year % 100 + 1}"
    for s in data.get("seasons", []):
        if s.get("year") == target:
            return int(s["id"])
    raise RuntimeError(
        f"UCL season {season_year}/{season_year % 100 + 1} not found in SportAPI7. "
        f"Available: {[s.get('year') for s in data.get('seasons', [])]}"
    )


# ── team ingest ───────────────────────────────────────────────────────────────

def upsert_teams(conn: sqlite3.Connection, events: list[dict]) -> None:
    seen: dict[int, tuple[str, str]] = {}
    for e in events:
        for side in ("homeTeam", "awayTeam"):
            t = e[side]
            tid = int(t["id"])
            if tid not in seen:
                name = t["name"]
                short = t.get("nameCode") or t.get("shortName", name[:3].upper())
                seen[tid] = (name, short)

    # Resolve short_name collisions: if two teams share a code, give the
    # later one a 4-char fallback so the UNIQUE constraint is satisfied.
    used_codes: dict[str, int] = {}
    rows: list[tuple] = []
    for tid, (name, short) in seen.items():
        if short in used_codes:
            short = name[:4].upper()
            if short in used_codes:
                short = f"{name[:3].upper()}{tid % 10}"
        used_codes[short] = tid
        rows.append((tid, name, short))

    sql = """
        INSERT INTO teams (team_id, name, short_name)
        VALUES (?, ?, ?)
        ON CONFLICT(team_id) DO UPDATE SET
            name = excluded.name,
            short_name = excluded.short_name
    """
    conn.executemany(sql, rows)
    print(f"  Upserted {len(rows)} teams")


# ── fixture ingest ────────────────────────────────────────────────────────────

def fetch_all_events(
    season_id: int, include_past: bool = True, past_only: bool = False
) -> list[dict]:
    """Fetch fixtures for the season.

    past_only=True  — only completed fixtures (use for fully-finished seasons).
    include_past=False — only upcoming fixtures (use at season start to avoid
        qualifying-round teams whose nameCodes collide with main-competition sides).
    include_past=True — both upcoming and completed (once league phase is under way).
    """
    all_events: list[dict] = []

    if not past_only:
        for page in range(0, 20):
            data = _get(
                f"unique-tournament/{UCL_TOURNAMENT_ID}/season/{season_id}/events/next/{page}"
            )
            events = data.get("events", [])
            all_events.extend(events)
            if not data.get("hasNextPage", False) or not events:
                break

    if past_only or include_past:
        for page in range(0, 20):
            data = _get(
                f"unique-tournament/{UCL_TOURNAMENT_ID}/season/{season_id}/events/last/{page}"
            )
            events = data.get("events", [])
            all_events.extend(events)
            if not data.get("hasNextPage", False) or not events:
                break

    seen: set[int] = set()
    unique: list[dict] = []
    for e in all_events:
        eid = e["id"]
        if eid not in seen:
            seen.add(eid)
            unique.append(e)
    return unique


def upsert_fixtures(conn: sqlite3.Connection, events: list[dict]) -> None:
    sql = """
        INSERT INTO fixtures (
            fixture_id, round_name,
            home_team_id, away_team_id,
            kickoff_utc, status,
            home_score, away_score, fetched_at_utc
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(fixture_id) DO UPDATE SET
            round_name     = excluded.round_name,
            kickoff_utc    = excluded.kickoff_utc,
            status         = excluded.status,
            home_score     = excluded.home_score,
            away_score     = excluded.away_score,
            fetched_at_utc = excluded.fetched_at_utc
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = []
    for e in events:
        ri = e.get("roundInfo", {})
        round_name = ri.get("name") or f"Round {ri.get('round', '?')}"
        home_score = e.get("homeScore", {}).get("current")
        away_score = e.get("awayScore", {}).get("current")
        raw_status = e.get("status", {}).get("description", "Not started")
        # Normalise SofaScore status to the canonical value train_dc expects
        status = "FT" if raw_status == "Ended" else raw_status
        rows.append((
            e["id"],
            round_name,
            int(e["homeTeam"]["id"]),
            int(e["awayTeam"]["id"]),
            _ts_to_utc(e.get("startTimestamp")),
            status,
            home_score,
            away_score,
            now,
        ))
    conn.executemany(sql, rows)
    print(f"  Upserted {len(rows)} fixtures")


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    _load_env()

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--season", type=int, default=None,
        help="UCL season start year (e.g. 2026 for 2026/27). Default: inferred.",
    )
    ap.add_argument(
        "--season-id", type=int, default=None,
        help="SportAPI7 season ID (skips the seasons lookup API call). "
             "Known IDs: 96518=26/27, 76953=25/26.",
    )
    ap.add_argument(
        "--include-past", action="store_true",
        help="Also fetch completed fixtures (use once league phase has started).",
    )
    ap.add_argument(
        "--past-only", action="store_true",
        help="Only fetch completed fixtures (use for fully-finished seasons).",
    )
    args = ap.parse_args()

    season = args.season or _current_ucl_season()
    print(f"=== UCL ingest — season {season}/{season % 100 + 1} ===")

    from ucl.init_db import init_db
    init_db()

    if args.season_id:
        season_id = args.season_id
        print(f"  Season ID: {season_id} (provided)")
    else:
        print(f"Looking up SportAPI7 season ID for {season}/{season % 100 + 1}…")
        season_id = find_season_id(season)
        print(f"  Season ID: {season_id}")

    print("Fetching fixtures…")
    events = fetch_all_events(
        season_id,
        include_past=args.include_past,
        past_only=args.past_only,
    )
    print(f"  {len(events)} fixtures fetched")

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        upsert_teams(conn, events)
        conn.commit()
        upsert_fixtures(conn, events)
        conn.commit()
    finally:
        conn.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
