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
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from wc.src.score_predictor import DCPredictor, _canonical
from wc.src.fantasy_optimizer import optimise as _optimise, captain_picks as _captain_picks

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

    # Seed fantasy players — always overwrite so deploys pick up price updates
    try:
        from wc.scripts.seed_fantasy_players import seed as _seed
        print("Seeding fantasy players…")
        _seed(overwrite=True)
    except Exception as _e:
        print(f"[warn] Fantasy seed skipped: {_e}")

    yield


app = FastAPI(
    title="WC2026 Score Predictor",
    description="Dixon-Coles Poisson model blending WC2018/2022 history with recent international form.",
    version="1.0.0",
    lifespan=lifespan,
)

ALLOWED_ORIGINS = os.environ.get(
    "ALLOWED_ORIGINS", "*"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
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


@app.get("/api/wc/teams")
def list_teams():
    """Return all team names the model knows, sorted alphabetically."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT name FROM teams ORDER BY name").fetchall()
    conn.close()
    return {"teams": [r[0] for r in rows]}


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


# ── Fantasy endpoints ─────────────────────────────────────────────────────────

class FantasyPlayerOut(BaseModel):
    id: int
    name: str
    team: str
    pos: str
    price: int
    price_m: float
    projected_pts: float
    pts_per_match: float
    is_captain: bool = False


class OptimiseRequest(BaseModel):
    budget: int = 1000  # $0.1m units; default $100m


class OptimiseResponse(BaseModel):
    squad: list[FantasyPlayerOut]
    total_pts: float
    total_cost: int
    total_cost_m: float
    captain: FantasyPlayerOut


class CaptainsResponse(BaseModel):
    picks: list[FantasyPlayerOut]


def _fmt_player(p: dict, is_captain: bool = False) -> FantasyPlayerOut:
    return FantasyPlayerOut(
        id=p["id"], name=p["name"], team=p["team"], pos=p["pos"],
        price=p["price"], price_m=round(p["price"] / 10, 1),
        projected_pts=p.get("projected_pts", 0.0),
        pts_per_match=p.get("pts_per_match", 0.0),
        is_captain=is_captain,
    )


@app.post("/api/wc/fantasy/optimise", response_model=OptimiseResponse)
def fantasy_optimise(req: OptimiseRequest):
    try:
        res = _optimise(budget=req.budget)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    captain_id = res["captain"]["id"]
    squad = [_fmt_player(p, is_captain=(p["id"] == captain_id)) for p in res["squad"]]
    return OptimiseResponse(
        squad=squad,
        total_pts=res["total_pts"],
        total_cost=res["total_cost"],
        total_cost_m=round(res["total_cost"] / 10, 1),
        captain=_fmt_player(res["captain"], is_captain=True),
    )


@app.get("/api/wc/fantasy/captains", response_model=CaptainsResponse)
def fantasy_captains(top_n: int = 10):
    picks = _captain_picks(top_n=top_n)
    return CaptainsResponse(picks=[_fmt_player(p) for p in picks])


# ── Tournament simulator ──────────────────────────────────────────────────────

class TournamentTeamOut(BaseModel):
    team: str
    group: str
    r32_pct: float
    qf_pct: float
    sf_pct: float
    final_pct: float
    win_pct: float


class TournamentResponse(BaseModel):
    teams: list[TournamentTeamOut]
    n_sim: int


@app.get("/api/wc/simulate", response_model=TournamentResponse)
def simulate_tournament(n_sim: int = 50_000):
    """
    Monte Carlo tournament simulation (default 50 000 runs).
    Returns win/final/SF/QF/R32 probabilities for all 48 teams.
    Results reflect the current DC model — retrain after each matchday for live updates.
    """
    predictor = _get_predictor()
    results = predictor.simulate_tournament(n_sim=n_sim)
    return TournamentResponse(
        teams=[TournamentTeamOut(**r) for r in results],
        n_sim=n_sim,
    )


# ── Live retrain ──────────────────────────────────────────────────────────────

import subprocess
import threading

_retrain_lock = threading.Lock()
_retrain_status: dict = {"running": False, "last": None, "error": None}


def _run_retrain() -> None:
    global _predictor
    try:
        subprocess.run(
            ["python", "wc/scripts/ingest_recent_form.py", "--months", "30"],
            check=True, capture_output=True, cwd=str(PROJECT_ROOT),
        )
        subprocess.run(
            ["python", "wc/scripts/train_dc.py"],
            check=True, capture_output=True, cwd=str(PROJECT_ROOT),
        )
        # Hot-reload the model in the running API
        p = DCPredictor()
        p.load()
        _predictor = p
        _retrain_status["last"] = "success"
        _retrain_status["error"] = None
        print("[retrain] DC model refreshed successfully.")
    except subprocess.CalledProcessError as e:
        _retrain_status["error"] = e.stderr.decode(errors="replace")[-500:]
        print(f"[retrain] failed: {_retrain_status['error']}")
    finally:
        _retrain_status["running"] = False


@app.post("/api/wc/retrain")
def trigger_retrain():
    """
    Re-ingest recent international results (including any WC2026 matches played)
    and retrain the DC model in the background.

    Call this after each matchday to keep predictions up to date.
    The martj42 dataset (GitHub) updates within hours of each match finishing.
    """
    if not _retrain_lock.acquire(blocking=False):
        return {"status": "already_running"}
    _retrain_status["running"] = True
    t = threading.Thread(target=_run_retrain, daemon=True)
    t.start()
    return {"status": "started", "message": "Retraining in background — check /health for completion."}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port)
