"""
Tests for src/ucl_api.py router functions.
No live server required — router functions are called directly.
Run: python ucl/test_ucl_api.py

A1  /api/ucl/standings returns 404 (not crash) when DB empty
A2  /api/ucl/league-phase/sim returns 404 when file missing
A3  /api/ucl/bracket returns 404 when file missing
A4  /api/ucl/standings returns correct structure when standings are present
A5  /api/ucl/predictions/upcoming returns 404 when file missing
A6  _slug() correctly slugifies round names
"""

from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import HTTPException


def check(label, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {label}" + (f"  ({detail})" if detail else ""))
    return cond


def _make_db_with_standings(tmp: Path) -> Path:
    db = tmp / "ucl.db"
    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE TABLE teams (team_id INT PRIMARY KEY, name TEXT, short_name TEXT UNIQUE);
        CREATE TABLE league_standings (
            team_id INT NOT NULL, season TEXT NOT NULL,
            played INT DEFAULT 0, won INT DEFAULT 0, drawn INT DEFAULT 0,
            lost INT DEFAULT 0, goals_for INT DEFAULT 0, goals_against INT DEFAULT 0,
            points INT DEFAULT 0, fetched_at_utc TEXT NOT NULL,
            PRIMARY KEY (team_id, season)
        );
    """)
    conn.execute("INSERT INTO teams VALUES (1, 'Real Madrid', 'RMA')")
    conn.execute("INSERT INTO teams VALUES (2, 'Arsenal', 'ARS')")
    conn.execute("INSERT INTO league_standings VALUES (1,'2026',3,2,1,0,7,2,7,'2026-10-01T00:00:00Z')")
    conn.execute("INSERT INTO league_standings VALUES (2,'2026',3,1,0,2,3,5,3,'2026-10-01T00:00:00Z')")
    conn.commit()
    conn.close()
    return db


def main():
    import src.ucl_api as api
    ok = True
    tmp = Path(tempfile.mkdtemp())

    # A1: standings 404 when DB doesn't exist
    with patch.object(api, "UCL_DB", tmp / "nonexistent.db"):
        try:
            api.standings()
            ok &= check("A1 standings 404 when DB missing", False)
        except HTTPException as e:
            ok &= check("A1 standings 404 when DB missing", e.status_code == 404,
                        f"status={e.status_code}")

    # A2: league-phase sim 404 when file missing
    with patch.object(api, "UCL_PREDS", tmp / "empty"):
        try:
            api.league_phase_sim()
            ok &= check("A2 league-phase/sim 404 when file missing", False)
        except HTTPException as e:
            ok &= check("A2 league-phase/sim 404 when file missing", e.status_code == 404)

    # A3: bracket 404 when file missing
    with patch.object(api, "UCL_PREDS", tmp / "empty"):
        try:
            api.bracket()
            ok &= check("A3 bracket 404 when file missing", False)
        except HTTPException as e:
            ok &= check("A3 bracket 404 when file missing", e.status_code == 404)

    # A4: standings returns correct structure when DB has data
    db = _make_db_with_standings(tmp)
    with patch.object(api, "UCL_DB", db):
        result = api.standings()
    ok &= check("A4 standings has 'standings' key", "standings" in result)
    rows = result.get("standings", [])
    ok &= check("A4 standings has 2 rows", len(rows) == 2, str(len(rows)))
    ok &= check("A4 first place is RMA",
                rows[0]["team"] == "RMA", str(rows[0].get("team")))
    ok &= check("A4 standings row has required keys",
                {"rank", "team", "team_name", "played", "won", "drawn", "lost",
                 "goals_for", "goals_against", "goal_difference", "points"}.issubset(rows[0]),
                str(set(rows[0].keys())))

    # A5: predictions/upcoming 404 when file missing
    with patch.object(api, "UCL_PREDS", tmp / "empty"):
        try:
            api.predictions_upcoming()
            ok &= check("A5 predictions/upcoming 404 when missing", False)
        except HTTPException as e:
            ok &= check("A5 predictions/upcoming 404 when missing", e.status_code == 404)

    # A6: _slug produces correct output
    cases = [
        ("League Phase", "league_phase"),
        ("Round of 16", "round_of_16"),
        ("Quarter-finals", "quarter_finals"),
        ("Semi-finals", "semi_finals"),
    ]
    for raw, expected in cases:
        got = api._slug(raw)
        ok &= check(f"A6 _slug({raw!r})", got == expected, f"got {got!r}")

    print("\n" + ("ALL PASS" if ok else "FAILURES PRESENT"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
