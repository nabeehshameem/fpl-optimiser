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

import logging
import os
import sqlite3
import subprocess
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
import sys

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Sentry (optional — only active when SENTRY_DSN env var is set) ────────────
_SENTRY_DSN = os.environ.get("SENTRY_DSN", "")
if _SENTRY_DSN:
    import sentry_sdk
    from sentry_sdk.integrations.fastapi import FastApiIntegration
    from sentry_sdk.integrations.starlette import StarletteIntegration
    sentry_sdk.init(
        dsn=_SENTRY_DSN,
        integrations=[StarletteIntegration(), FastApiIntegration()],
        traces_sample_rate=0.1,
        send_default_pii=False,
    )

from wc.src.score_predictor import DCPredictor, _canonical
from wc.src.fantasy_optimizer import optimise as _optimise, captain_picks as _captain_picks, live_captain_advice as _live_captain_advice

DB_PATH = PROJECT_ROOT / "wc" / "data" / "wc.db"

_predictor: DCPredictor | None = None
_load_error: str | None = None

# ── Simulate cache (keyed by n_sim, TTL = 30 min) ────────────────────────────
_sim_cache: dict[int, tuple[float, list]] = {}
_SIM_CACHE_TTL = 1800

# ── Rate limiter ──────────────────────────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])

# ── Retrain auth ──────────────────────────────────────────────────────────────

_RETRAIN_TOKEN = os.environ.get("RETRAIN_SECRET_TOKEN", "")
_retrain_header = APIKeyHeader(name="X-Retrain-Token", auto_error=False)


def _require_retrain_token(token: str | None = Depends(_retrain_header)):
    if not _RETRAIN_TOKEN:
        raise HTTPException(status_code=503, detail="Retrain endpoint not configured.")
    if token != _RETRAIN_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing retrain token.")


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

    # Ensure subscribers table exists
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS subscribers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.commit()
        conn.close()
    except Exception as _e:
        print(f"[warn] Subscribers table init failed: {_e}")

    yield


app = FastAPI(
    title="WC2026 Score Predictor",
    description="Dixon-Coles Poisson model blending WC2018/2022 history with recent international form.",
    version="1.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "An internal error occurred."})

ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "http://localhost:5173").split(",")

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
    home_team: str = Field(..., min_length=1, max_length=100)
    away_team: str = Field(..., min_length=1, max_length=100)
    home_advantage: bool = False
    knockout: bool = False


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
    ko_win_pct: float | None = None
    ko_loss_pct: float | None = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _predictor is not None}


@app.get("/api/wc/teams")
@limiter.limit("30/minute")
def list_teams(request: Request):
    """Return all team names the model knows, sorted alphabetically."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT name FROM teams ORDER BY name").fetchall()
    conn.close()
    return {"teams": [r[0] for r in rows]}


@app.post("/api/wc/predict", response_model=PredictResponse)
@limiter.limit("30/minute")
def predict_match(req: PredictRequest, request: Request) -> PredictResponse:
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
        result = predictor.predict(home_id, away_id, home_advantage=req.home_advantage, knockout=req.knockout)
    except RuntimeError as e:
        logger.exception("predict failed for %s vs %s", req.home_team, req.away_team)
        raise HTTPException(status_code=400, detail="Prediction failed. Check team names and try again.")

    return PredictResponse(
        home_name=result["home_name"],
        away_name=result["away_name"],
        home_xg=result["home_xg"],
        away_xg=result["away_xg"],
        predicted_score=f"{round(result['home_xg'])}-{round(result['away_xg'])}",
        win_pct=result["win_pct"],
        draw_pct=result["draw_pct"],
        loss_pct=result["loss_pct"],
        most_likely=[
            ScoreLine(home_goals=h, away_goals=a, probability_pct=p)
            for h, a, p in result["most_likely"]
        ],
        ko_win_pct=result.get("ko_win_pct"),
        ko_loss_pct=result.get("ko_loss_pct"),
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
    is_starter: bool = True
    is_captain: bool = False


class OptimiseRequest(BaseModel):
    budget: int = Field(default=1000, ge=500, le=1500)  # $50m–$150m in $0.1m units


class OptimiseResponse(BaseModel):
    squad: list[FantasyPlayerOut]
    starters: list[FantasyPlayerOut]
    bench: list[FantasyPlayerOut]
    total_pts: float
    total_cost: int
    total_cost_m: float
    captain: FantasyPlayerOut


class CaptainsResponse(BaseModel):
    picks: list[FantasyPlayerOut]


class LiveAdviceResponse(BaseModel):
    is_live: bool
    matchday: int
    played_team_count: int
    remaining_team_count: int
    remaining_picks: list[FantasyPlayerOut]


def _fmt_player(p: dict, is_captain: bool = False) -> FantasyPlayerOut:
    return FantasyPlayerOut(
        id=p["id"], name=p["name"], team=p["team"], pos=p["pos"],
        price=p["price"], price_m=round(p["price"] / 10, 1),
        projected_pts=p.get("projected_pts", 0.0),
        pts_per_match=p.get("pts_per_match", 0.0),
        is_starter=p.get("is_starter", True),
        is_captain=is_captain,
    )


@app.post("/api/wc/fantasy/optimise", response_model=OptimiseResponse)
@limiter.limit("10/minute")
def fantasy_optimise(req: OptimiseRequest, request: Request):
    try:
        res = _optimise(budget=req.budget, predictor=_predictor)
    except RuntimeError:
        logger.exception("fantasy optimise failed")
        raise HTTPException(status_code=503, detail="Optimisation unavailable. Try again shortly.")
    captain_id = res["captain"]["id"]
    squad    = [_fmt_player(p, is_captain=(p["id"] == captain_id)) for p in res["squad"]]
    starters = [_fmt_player(p, is_captain=(p["id"] == captain_id)) for p in res["starters"]]
    bench    = [_fmt_player(p) for p in res["bench"]]
    return OptimiseResponse(
        squad=squad,
        starters=starters,
        bench=bench,
        total_pts=res["total_pts"],
        total_cost=res["total_cost"],
        total_cost_m=round(res["total_cost"] / 10, 1),
        captain=_fmt_player(res["captain"], is_captain=True),
    )


@app.get("/api/wc/fantasy/captains", response_model=CaptainsResponse)
@limiter.limit("20/minute")
def fantasy_captains(request: Request, top_n: int = 10, matchday: int | None = None):
    top_n = min(max(top_n, 1), 20)
    if matchday is not None:
        matchday = max(1, min(3, matchday))
    picks = _captain_picks(top_n=top_n, predictor=_predictor, matchday=matchday)
    return CaptainsResponse(picks=[_fmt_player(p) for p in picks])


@app.get("/api/wc/fantasy/captains/live", response_model=LiveAdviceResponse)
@limiter.limit("10/minute")
def fantasy_captains_live(request: Request, matchday: int = 1):
    matchday = max(1, min(3, matchday))
    result = _live_captain_advice(matchday=matchday, predictor=_predictor)
    return LiveAdviceResponse(
        is_live=result["is_live"],
        matchday=result["matchday"],
        played_team_count=result["played_team_count"],
        remaining_team_count=result["remaining_team_count"],
        remaining_picks=[_fmt_player(p) for p in result["remaining_picks"]],
    )


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
@limiter.limit("2/minute")
def simulate_tournament(request: Request, n_sim: int = 50_000):
    """
    Monte Carlo tournament simulation (default 50 000 runs).
    Returns win/final/SF/QF/R32 probabilities for all 48 teams.
    Results reflect the current DC model — retrain after each matchday for live updates.
    Results are cached for 30 minutes per n_sim value.
    """
    n_sim = min(max(n_sim, 1_000), 100_000)
    predictor = _get_predictor()

    cached = _sim_cache.get(n_sim)
    if cached and (time.time() - cached[0]) < _SIM_CACHE_TTL:
        return TournamentResponse(teams=[TournamentTeamOut(**r) for r in cached[1]], n_sim=n_sim)

    results = predictor.simulate_tournament(n_sim=n_sim)
    _sim_cache[n_sim] = (time.time(), results)
    return TournamentResponse(
        teams=[TournamentTeamOut(**r) for r in results],
        n_sim=n_sim,
    )


# ── Email notifications ───────────────────────────────────────────────────────

class SubscribeRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=254)


@app.post("/api/notify/subscribe")
@limiter.limit("5/minute")
def subscribe_email(req: SubscribeRequest, request: Request):
    """Save an email address for match-day notifications."""
    import re
    _OK = {"status": "ok"}
    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", req.email):
        return _OK
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("INSERT INTO subscribers (email) VALUES (?)", (req.email,))
        conn.commit()
        conn.close()
    except sqlite3.IntegrityError:
        pass
    except Exception:
        logger.exception("Failed to save subscriber")
    return _OK


# ── Live retrain ──────────────────────────────────────────────────────────────

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
        p = DCPredictor()
        p.load()
        _predictor = p
        _sim_cache.clear()
        _retrain_status["last"] = "success"
        _retrain_status["error"] = None
        print("[retrain] DC model refreshed successfully.")
    except subprocess.CalledProcessError as e:
        _retrain_status["error"] = e.stderr.decode(errors="replace")[-500:]
        print(f"[retrain] failed: {_retrain_status['error']}")
    finally:
        _retrain_status["running"] = False


@app.post("/api/wc/retrain", dependencies=[Depends(_require_retrain_token)])
def trigger_retrain():
    """
    Re-ingest recent international results and retrain the DC model in the background.
    Requires X-Retrain-Token header matching RETRAIN_SECRET_TOKEN env var.
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
