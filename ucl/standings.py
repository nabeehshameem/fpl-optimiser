"""
ucl/standings.py
Derive the UCL league-phase table from finished fixtures.

Derived rather than stored: football-data.org's ingest gives us results, and a
second synced copy of the table is one more thing that can go stale. This is
the single implementation, shared by the exporter and any caller that has the
database to hand.

Railway has no database (Rule 8) — the API reads git-committed artifacts only,
and data/ is gitignored. So the table is exported to predictions/ucl at ingest
time and the API serves that file, never the DB.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "ucl.db"
EXPORT_DIR = PROJECT_ROOT / "predictions" / "ucl"
EXPORT_NAME = "standings.json"

FINISHED_STATUSES = ("FT", "AET", "PEN")


def season_bounds(season: int) -> tuple[str, str]:
    """A UCL season runs July→June, so it straddles two calendar years.

    The database holds more than one season; any query over fixtures must scope
    to one or a finished season bleeds into the running one.
    """
    return f"{season}-07-01", f"{season + 1}-07-01"


def derive(conn: sqlite3.Connection, season: int) -> dict:
    lo, hi = season_bounds(season)

    entrants = conn.execute(
        """
        SELECT DISTINCT t.team_id, t.short_name, t.name
        FROM fixtures f
        JOIN teams t ON t.team_id IN (f.home_team_id, f.away_team_id)
        WHERE f.kickoff_utc >= ? AND f.kickoff_utc < ?
        """,
        (lo, hi),
    ).fetchall()

    played = conn.execute(
        f"""
        SELECT home_team_id, away_team_id, home_score, away_score
        FROM fixtures
        WHERE kickoff_utc >= ? AND kickoff_utc < ?
          AND status IN ({','.join('?' * len(FINISHED_STATUSES))})
          AND home_score IS NOT NULL AND away_score IS NOT NULL
        """,
        (lo, hi, *FINISHED_STATUSES),
    ).fetchall()

    table = {
        tid: {"team": sn, "team_name": name, "played": 0, "won": 0, "drawn": 0,
              "lost": 0, "goals_for": 0, "goals_against": 0, "points": 0}
        for tid, sn, name in entrants
    }

    for home_id, away_id, hs, as_ in played:
        for tid, gf, ga in ((home_id, hs, as_), (away_id, as_, hs)):
            row = table.get(tid)
            if row is None:
                continue
            row["played"] += 1
            row["goals_for"] += gf
            row["goals_against"] += ga
            if gf > ga:
                row["won"] += 1
                row["points"] += 3
            elif gf == ga:
                row["drawn"] += 1
                row["points"] += 1
            else:
                row["lost"] += 1

    rows = sorted(
        table.values(),
        key=lambda r: (-r["points"],
                       -(r["goals_for"] - r["goals_against"]),
                       -r["goals_for"],
                       r["team_name"]),
    )
    for i, r in enumerate(rows, 1):
        r["rank"] = i
        r["goal_difference"] = r["goals_for"] - r["goals_against"]

    return {
        "season": season,
        "matches_played": len(played),
        "season_started": bool(played),
        "standings": rows,
    }


def export(season: int, db_path: Path = DB_PATH) -> Path:
    conn = sqlite3.connect(db_path)
    try:
        payload = derive(conn, season)
    finally:
        conn.close()

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = EXPORT_DIR / EXPORT_NAME
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    return out
