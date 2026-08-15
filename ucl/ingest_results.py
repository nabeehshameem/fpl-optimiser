"""
ucl/ingest_results.py
After each match day, fetch per-match statistics from API-Football and write
them to the match_stats table in data/ucl.db. These stats feed DC parameter
retraining in ucl/train_dc.py.

    python ucl/ingest_results.py                # all FT fixtures this season
    python ucl/ingest_results.py --season 2026  # explicit season

Only fetches stats for fixtures already in the DB with status FT (full time).
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
        raise RuntimeError("RAPIDAPI_KEY not set — add it to .env or the environment.")
    return {
        "X-RapidAPI-Key": key,
        "X-RapidAPI-Host": RAPIDAPI_HOST,
    }


def _get(path: str, params: dict) -> dict:
    url = f"https://{RAPIDAPI_HOST}{path}"
    r = requests.get(url, headers=_headers(), params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    if data.get("errors"):
        raise RuntimeError(f"API-Football error: {data['errors']}")
    return data


def _current_ucl_season() -> int:
    now = datetime.now(timezone.utc)
    return now.year if now.month >= 7 else now.year - 1


def _finished_without_stats(conn: sqlite3.Connection) -> list[int]:
    """Return fixture IDs that are FT in fixtures but missing from match_stats."""
    rows = conn.execute("""
        SELECT f.fixture_id FROM fixtures f
        WHERE f.status = 'FT'
          AND f.fixture_id NOT IN (SELECT fixture_id FROM match_stats)
        ORDER BY f.kickoff_utc
    """).fetchall()
    return [r[0] for r in rows]


def _parse_stat(stats: list[dict], team_type: str, stat_name: str):
    """Extract a single stat value from the API-Football statistics response."""
    for entry in stats:
        if entry.get("team", {}).get("home_away") == team_type:
            for s in entry.get("statistics", []):
                if s["type"] == stat_name:
                    v = s.get("value")
                    if isinstance(v, str) and v.endswith("%"):
                        return int(v.rstrip("%"))
                    return v if v is not None else None
    return None


def _parse_stats_v2(response: list[dict], stat_name: str, idx: int):
    """API-Football /fixtures/statistics returns a list[team_stats].
    idx=0 → home, idx=1 → away.
    """
    if idx >= len(response):
        return None
    for s in response[idx].get("statistics", []):
        if s["type"] == stat_name:
            v = s.get("value")
            if isinstance(v, str) and v.endswith("%"):
                return int(v.rstrip("%"))
            return v if v is not None else None
    return None


def fetch_match_stats(fixture_id: int) -> dict | None:
    """Fetch and parse per-match stats. Returns None on API error."""
    try:
        data = _get("/fixtures/statistics", {"fixture": fixture_id})
        resp = data.get("response", [])
        if not resp:
            return None

        def stat(name: str, idx: int):
            return _parse_stats_v2(resp, name, idx)

        return {
            "shots_home":      stat("Total Shots", 0),
            "shots_away":      stat("Total Shots", 1),
            "on_target_home":  stat("Shots on Goal", 0),
            "on_target_away":  stat("Shots on Goal", 1),
            "possession_home": stat("Ball Possession", 0),
            "possession_away": stat("Ball Possession", 1),
            "xg_home":         stat("expected_goals", 0),
            "xg_away":         stat("expected_goals", 1),
        }
    except Exception as exc:
        print(f"  [WARN] stats fetch failed for fixture {fixture_id}: {exc}",
              file=sys.stderr)
        return None


def upsert_match_stats(
    conn: sqlite3.Connection, fixture_id: int, stats: dict
) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    conn.execute("""
        INSERT INTO match_stats (
            fixture_id, shots_home, shots_away,
            on_target_home, on_target_away,
            possession_home, possession_away,
            xg_home, xg_away, fetched_at_utc
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(fixture_id) DO UPDATE SET
            shots_home      = excluded.shots_home,
            shots_away      = excluded.shots_away,
            on_target_home  = excluded.on_target_home,
            on_target_away  = excluded.on_target_away,
            possession_home = excluded.possession_home,
            possession_away = excluded.possession_away,
            xg_home         = excluded.xg_home,
            xg_away         = excluded.xg_away,
            fetched_at_utc  = excluded.fetched_at_utc
    """, (
        fixture_id,
        stats.get("shots_home"), stats.get("shots_away"),
        stats.get("on_target_home"), stats.get("on_target_away"),
        stats.get("possession_home"), stats.get("possession_away"),
        stats.get("xg_home"), stats.get("xg_away"),
        now,
    ))


def main() -> None:
    _load_env()

    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=None)
    args = ap.parse_args()

    season = args.season or _current_ucl_season()
    print(f"=== UCL result ingest — season {season}/{season % 100 + 1} ===")

    # Refresh fixture statuses first so we know which are FT
    from ucl.ingest_fixtures import fetch_fixtures, upsert_fixtures
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        fixtures = fetch_fixtures(season)
        upsert_fixtures(conn, fixtures)
        conn.commit()

        pending = _finished_without_stats(conn)
        print(f"{len(pending)} finished fixtures need stats")

        for i, fid in enumerate(pending, 1):
            print(f"  [{i}/{len(pending)}] fixture {fid}")
            stats = fetch_match_stats(fid)
            if stats:
                upsert_match_stats(conn, fid, stats)
                conn.commit()
            # Be polite to the API — 1 req/sec is safe on Pro tier
            if i < len(pending):
                time.sleep(1.1)
    finally:
        conn.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
