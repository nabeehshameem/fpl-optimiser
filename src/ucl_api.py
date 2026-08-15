"""
ucl_api.py
Read-only UCL endpoints. Mounted into api.py:

    from src.ucl_api import router as ucl_router
    app.include_router(ucl_router)

Data sources (Rule 8 — Railway has no DB for UCL yet):
  predictions/ucl/*_predictions.json  — written by ucl/run_predictions.py
  data/ucl.db                         — standings read live, DB is committed

Endpoints:
  GET /api/ucl/predictions/upcoming   — all upcoming fixtures
  GET /api/ucl/predictions/{slug}     — a specific round (slugified round_name)
  GET /api/ucl/standings              — league-phase table
  GET /api/ucl/bracket                — knockout bracket (B6 placeholder)
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

from fastapi import APIRouter, HTTPException

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UCL_DB = PROJECT_ROOT / "data" / "ucl.db"
UCL_PREDS = PROJECT_ROOT / "predictions" / "ucl"

router = APIRouter(prefix="/api/ucl", tags=["ucl"])


def _slug(round_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", round_name.lower()).strip("_")


def _load_predictions(filename: str) -> dict:
    p = UCL_PREDS / filename
    if not p.exists():
        raise HTTPException(404, f"No predictions file: {filename}")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise HTTPException(502, f"predictions file unreadable: {exc}") from exc


@router.get("/predictions/upcoming")
def predictions_upcoming() -> dict:
    """All upcoming UCL fixtures with scoreline predictions."""
    p = UCL_PREDS / "all_upcoming_predictions.json"
    if not p.exists():
        raise HTTPException(
            404,
            "UCL predictions not yet published. Run ucl/run_predictions.py first."
        )
    return _load_predictions("all_upcoming_predictions.json")


@router.get("/predictions/{round_slug}")
def predictions_round(round_slug: str) -> dict:
    """Predictions for a specific round (e.g. 'round_of_16', 'league_phase')."""
    filename = f"{round_slug}_predictions.json"
    return _load_predictions(filename)


@router.get("/standings")
def standings() -> dict:
    """UCL league-phase table from data/ucl.db."""
    if not UCL_DB.exists():
        raise HTTPException(
            404,
            "UCL database not initialised. Run ucl/init_db.py and ucl/ingest_fixtures.py."
        )
    conn = sqlite3.connect(UCL_DB)
    try:
        rows = conn.execute("""
            SELECT
                t.short_name, t.name,
                ls.played, ls.won, ls.drawn, ls.lost,
                ls.goals_for, ls.goals_against,
                ls.goals_for - ls.goals_against AS gd,
                ls.points, ls.fetched_at_utc
            FROM league_standings ls
            JOIN teams t ON t.team_id = ls.team_id
            ORDER BY ls.points DESC, gd DESC, ls.goals_for DESC
        """).fetchall()
    finally:
        conn.close()

    if not rows:
        raise HTTPException(
            404,
            "No standings data yet. Run ucl/ingest_fixtures.py to populate."
        )

    fetched_at = rows[0][10] if rows else None
    return {
        "fetched_at_utc": fetched_at,
        "standings": [
            {
                "rank": i + 1,
                "team": r[0],
                "team_name": r[1],
                "played": r[2],
                "won": r[3],
                "drawn": r[4],
                "lost": r[5],
                "goals_for": r[6],
                "goals_against": r[7],
                "goal_difference": r[8],
                "points": r[9],
            }
            for i, r in enumerate(rows)
        ],
    }


@router.get("/bracket")
def bracket() -> dict:
    """UCL knockout bracket. Placeholder — implemented in B6."""
    raise HTTPException(
        501,
        "UCL bracket endpoint not yet implemented (B6). "
        "Check back after the league phase concludes."
    )
