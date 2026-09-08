"""
ucl_api.py
Read-only UCL endpoints. Mounted into api.py:

    from src.ucl_api import router as ucl_router
    app.include_router(ucl_router)

Data sources (Rule 8 — Railway has no DB, and data/ is gitignored, so every
endpoint here reads a git-committed artifact and never the database):
  predictions/ucl/*_predictions.json  — written by ucl/run_predictions.py
  predictions/ucl/standings.json      — written by ucl/ingest_fd.py

Endpoints:
  GET /api/ucl/predictions/upcoming   — all upcoming fixtures
  GET /api/ucl/predictions/{slug}     — a specific round (slugified round_name)
  GET /api/ucl/standings              — league-phase table
  GET /api/ucl/bracket                — knockout bracket (B6 placeholder)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException

PROJECT_ROOT = Path(__file__).resolve().parent.parent
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
    """UCL league-phase table, as exported at ingest time.

    Read from the committed artifact rather than data/ucl.db: Railway has no
    database and data/ is gitignored (Rule 8), so a query here would 404 in
    production while passing locally. ucl/standings.py derives it from finished
    fixtures and writes the file whenever results are ingested.

    Before a ball is kicked this is every entrant on zero with
    season_started: false — a real standing of a season that has not started,
    not an error. The frontend renders that state.
    """
    p = UCL_PREDS / "standings.json"
    if not p.exists():
        raise HTTPException(
            404,
            "UCL standings not published yet. Run: python ucl/ingest_fd.py",
        )
    return _load_predictions("standings.json")


@router.get("/league-phase/sim")
def league_phase_sim() -> dict:
    """Monte-Carlo qualification probabilities for the UCL league phase."""
    p = UCL_PREDS / "league_phase_sim.json"
    if not p.exists():
        raise HTTPException(
            404,
            "League phase simulation not yet run. "
            "Run: python ucl/simulate_league_phase.py"
        )
    return _load_predictions("league_phase_sim.json")


@router.get("/bracket")
def bracket() -> dict:
    """UCL knockout bracket — available after the league phase concludes (B6)."""
    p = UCL_PREDS / "bracket.json"
    if not p.exists():
        raise HTTPException(
            404,
            "UCL bracket not yet published. "
            "It will be available after the league phase concludes."
        )
    return _load_predictions("bracket.json")
