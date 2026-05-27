"""
api.py — FastAPI server for the WC2026 score predictor.

Run:
    uvicorn api:app --reload

Requires a trained DC model and populated DB:
    python wc/scripts/ingest_sb.py
    python wc/scripts/ingest_recent_form.py
    python wc/scripts/train_dc.py
    python wc/scripts/ingest_fixtures.py
"""

import os
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path
import sys

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from wc.src.score_predictor import DCPredictor, _canonical

DB_PATH = PROJECT_ROOT / "wc" / "data" / "wc.db"

_predictor: DCPredictor | None = None
_load_error: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _predictor, _load_error
    try:
        p = DCPredictor()
        p.load()
        _predictor = p
        print("DC model loaded successfully.")
    except FileNotFoundError as e:
        _load_error = str(e)
        print(f"[warn] DC model not loaded: {e}")
    yield


app = FastAPI(
    title="WC2026 Score Predictor",
    description="Dixon-Coles Poisson model blending WC2018/2022 history with recent international form.",
    version="1.0.0",
    lifespan=lifespan,
)


def _get_predictor() -> DCPredictor:
    if _predictor is None:
        raise HTTPException(
            status_code=503,
            detail=_load_error or "Model not loaded. Run: python wc/scripts/train_dc.py",
        )
    return _predictor


def _resolve_team_id(name: str) -> int | None:
    """Canonical-name match against the teams table."""
    canonical = _canonical(name)
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT team_id, name FROM teams").fetchall()
    conn.close()
    for tid, tname in rows:
        if _canonical(tname) == canonical:
            return tid
    return None


# ── Request / Response models ─────────────────────────────────────────────────

class PredictRequest(BaseModel):
    home_team: str
    away_team: str
    home_advantage: bool = False


class ScoreLine(BaseModel):
    home_goals: int
    away_goals: int
    probability_pct: float


class PredictResponse(BaseModel):
    home_name: str
    away_name: str
    home_xg: float
    away_xg: float
    predicted_score: str
    win_pct: float
    draw_pct: float
    loss_pct: float
    most_likely: list[ScoreLine]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _predictor is not None}


@app.post("/api/wc/predict", response_model=PredictResponse)
def predict_match(req: PredictRequest) -> PredictResponse:
    """
    Predict the scoreline for a single match.

    Team names are matched case-insensitively against the teams database.
    Use the same names FIFA/StatsBomb uses (e.g. "Brazil", "Korea Republic",
    "IR Iran", "USA").
    """
    predictor = _get_predictor()

    home_id = _resolve_team_id(req.home_team)
    away_id = _resolve_team_id(req.away_team)

    if home_id is None:
        raise HTTPException(status_code=404, detail=f"Team not found: '{req.home_team}'")
    if away_id is None:
        raise HTTPException(status_code=404, detail=f"Team not found: '{req.away_team}'")

    try:
        result = predictor.predict(home_id, away_id, home_advantage=req.home_advantage)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    top = result["most_likely"][0]
    return PredictResponse(
        home_name=result["home_name"],
        away_name=result["away_name"],
        home_xg=result["home_xg"],
        away_xg=result["away_xg"],
        predicted_score=f"{top[0]}-{top[1]}",
        win_pct=result["win_pct"],
        draw_pct=result["draw_pct"],
        loss_pct=result["loss_pct"],
        most_likely=[
            ScoreLine(home_goals=h, away_goals=a, probability_pct=p)
            for h, a, p in result["most_likely"]
        ],
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port)
