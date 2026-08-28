"""
ucl/ingest_fd.py
Fetch UCL fixtures and results from football-data.org and write them to
data/ucl.db.  Uses the same schema as ingest_fixtures.py so train_dc.py
and run_predictions.py work unchanged.

    python ucl/ingest_fd.py --season 2025   # 25/26 results (DC training)
    python ucl/ingest_fd.py --season 2026   # 26/27 fixtures (once available)

Requires FOOTBALL_DATA_TOKEN in environment (or .env at project root).
Register free at football-data.org — no card required.
Competition code for UCL: CL.  Free tier: 10 req/min.
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
API_BASE = "https://api.football-data.org/v4"
COMPETITION = "CL"

STAGE_LABELS = {
    "LEAGUE_PHASE":        "League Phase",
    "ROUND_OF_16":         "Round of 16",
    "QUARTER_FINALS":      "Quarter-finals",
    "SEMI_FINALS":         "Semi-finals",
    "FINAL":               "Final",
    "PRELIMINARY_ROUND":   "Preliminary Round",
    "QUALIFYING_ROUNDS":   "Qualifying Round",
    "PLAY_OFF_ROUND":      "Play-off Round",
}


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
    token = os.environ.get("FOOTBALL_DATA_TOKEN")
    if not token:
        raise RuntimeError(
            "FOOTBALL_DATA_TOKEN not set — add it to .env. "
            "Register free at football-data.org."
        )
    return {"X-Auth-Token": token}


def _get(path: str, params: dict | None = None) -> dict:
    r = requests.get(f"{API_BASE}/{path}", headers=_headers(),
                     params=params or {}, timeout=30)
    if r.status_code == 404:
        return {}
    r.raise_for_status()
    return r.json()


def _current_ucl_season() -> int:
    now = datetime.now(timezone.utc)
    return now.year if now.month >= 7 else now.year - 1


# ── team + fixture ingest ─────────────────────────────────────────────────────

def upsert_teams(conn: sqlite3.Connection, matches: list[dict]) -> None:
    seen: dict[int, tuple[str, str]] = {}
    for m in matches:
        for side in ("homeTeam", "awayTeam"):
            t = m[side]
            tid = int(t["id"])
            if tid not in seen:
                name = t["name"]
                short = t.get("tla") or t.get("shortName", name[:3].upper())
                seen[tid] = (name, short)

    sql = """
        INSERT INTO teams (team_id, name, short_name)
        VALUES (?, ?, ?)
        ON CONFLICT(team_id) DO UPDATE SET
            name = excluded.name,
            short_name = excluded.short_name
    """
    rows = [(tid, name, short) for tid, (name, short) in seen.items()]
    conn.executemany(sql, rows)
    print(f"  Upserted {len(rows)} teams")


def upsert_fixtures(conn: sqlite3.Connection, matches: list[dict]) -> None:
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
    for m in matches:
        stage = m.get("stage", "")
        md = m.get("matchday")
        round_name = STAGE_LABELS.get(stage, stage)
        if stage == "LEAGUE_PHASE" and md:
            round_name = f"Matchday {md}"

        raw_status = m.get("status", "SCHEDULED")
        status = "FT" if raw_status == "FINISHED" else raw_status

        score = m.get("score", {})
        ft = score.get("fullTime", {})
        home_score = ft.get("home")
        away_score = ft.get("away")

        rows.append((
            m["id"],
            round_name,
            int(m["homeTeam"]["id"]),
            int(m["awayTeam"]["id"]),
            m.get("utcDate"),
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
    ap.add_argument("--season", type=int, default=None,
                    help="UCL season start year (e.g. 2025 for 25/26). Default: inferred.")
    args = ap.parse_args()

    season = args.season or _current_ucl_season()
    print(f"=== UCL ingest (football-data.org) — season {season}/{season % 100 + 1} ===")

    from ucl.init_db import init_db
    init_db()

    print(f"Fetching CL matches for season {season}…")
    # football-data.org free tier: the current active season needs no season
    # param (specifying it returns 400).  For future seasons that aren't
    # in the API yet, the endpoint returns 404 — print a clear message.
    current_year = _current_ucl_season()
    params: dict = {} if season == current_year else {"season": season}
    data = _get(f"competitions/{COMPETITION}/matches", params)
    if not data:
        print(f"  No data returned — season {season} may not be available yet.")
        return

    matches = data.get("matches", [])
    print(f"  {len(matches)} matches returned")

    if not matches:
        return

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        upsert_teams(conn, matches)
        conn.commit()
        upsert_fixtures(conn, matches)
        conn.commit()
    finally:
        conn.close()

    finished = sum(1 for m in matches if m.get("status") == "FINISHED")
    print(f"  {finished} finished (FT), {len(matches) - finished} scheduled")
    print("\nDone.")


if __name__ == "__main__":
    main()
