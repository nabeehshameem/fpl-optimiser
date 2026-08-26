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

import datetime as _dt
import itertools
import logging
import math as _math
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
from typing import Literal, Optional
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

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
from wc.src.fantasy_optimizer import optimise as _optimise, captain_picks as _captain_picks, get_projected_players as _get_projected_players

DB_PATH = PROJECT_ROOT / "wc" / "data" / "wc.db"

# User-data paths — these must NOT be inside PROJECT_ROOT on Railway because
# the project directory is wiped on every deploy.  Set these env vars to point
# at a mounted Railway volume (/data/...) before first deploy.
SUBSCRIBERS_DB = Path(os.getenv("SUBSCRIBERS_DB", str(PROJECT_ROOT / "data" / "subscribers.db")))
# Mirror the env-var defaults used by the sub-routers so the guard can check
# them without importing those modules (imports haven't happened yet).
_RECEIPTS_DB_DEFAULT = Path(os.getenv("RECEIPTS_DB", str(PROJECT_ROOT / "data" / "receipts.db")))
_CARDS_DIR_DEFAULT = Path(os.getenv("CARDS_DIR", str(PROJECT_ROOT / "cards")))


def _assert_persistent_paths() -> None:
    """Refuse to start on Railway if any user-data path is inside the
    ephemeral project directory, which is wiped on every deploy."""
    if not os.environ.get("RAILWAY_ENVIRONMENT"):
        return
    root = PROJECT_ROOT.resolve()
    checks = [
        ("SUBSCRIBERS_DB", SUBSCRIBERS_DB),
        ("RECEIPTS_DB",    _RECEIPTS_DB_DEFAULT),
        ("CARDS_DIR",      _CARDS_DIR_DEFAULT),
    ]
    offenders = [
        f"  {name}={path}  →  set {name}=/data/{name.lower()} on a Railway volume"
        for name, path in checks
        if Path(path).resolve().is_relative_to(root)
    ]
    if offenders:
        raise RuntimeError(
            "Refusing to start: user-data paths are inside the ephemeral "
            "project directory and will be lost on every deploy:\n"
            + "\n".join(offenders)
            + "\n\nAttach a Railway volume at /data and set the env vars above."
        )


_assert_persistent_paths()

_predictor: DCPredictor | None = None
_load_error: str | None = None

# ── Static lookup caches (built at startup, never change mid-session) ─────────
_team_id_cache: dict[str, int] = {}       # canonical_name → team_id
_fixture_md_cache: dict[tuple, int] = {}  # (home_id, away_id) → matchday

# ── Simulate cache (keyed by n_sim, TTL = 30 min) ────────────────────────────
_sim_cache: dict[int, tuple[float, list]] = {}
_SIM_CACHE_TTL = 1800

# ── Captain picks cache (keyed by (top_n, matchday), TTL = 30 min) ───────────
_captains_cache: dict[tuple, tuple[float, list]] = {}
_CAPTAINS_CACHE_TTL = 1800

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

    # Pull live fantasy player prices from FIFA API on every deploy
    try:
        from wc.scripts.ingest_fantasy_players import ingest_from_fifa_api as _ingest_players
        print("Ingesting fantasy players from FIFA API…")
        _ingest_players()
    except Exception as _e:
        print(f"[warn] FIFA fantasy ingest failed, falling back to seed: {_e}")
        try:
            from wc.scripts.seed_fantasy_players import seed as _seed
            _seed(overwrite=False)
        except Exception as _e2:
            print(f"[warn] Fantasy seed also skipped: {_e2}")

    # Seed group-stage fixtures (matchday pairings + kickoff times)
    try:
        from wc.scripts.seed_fixtures import seed as _seed_fixtures
        print("Seeding fixtures…")
        _seed_fixtures(overwrite=True)
    except Exception as _e:
        print(f"[warn] Fixture seed skipped: {_e}")

    # Ensure subscribers table exists in its own DB (not in wc.db)
    try:
        SUBSCRIBERS_DB.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(SUBSCRIBERS_DB)
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

    # Build in-memory lookup caches (team IDs + fixture matchdays)
    _build_lookup_caches()

    # Warm the simulation cache in background so first user request is instant
    if _predictor is not None:
        _sim_running.add(10_000)
        threading.Thread(target=_run_simulation_bg, args=(_predictor, 10_000), daemon=True).start()
        print("[startup] Warming simulation cache in background…")

    # Daily auto-retrain at 06:00 UTC — keeps the model current with latest results
    threading.Thread(target=_daily_retrain_loop, daemon=True).start()
    print("[scheduler] Daily retrain scheduled at 06:00 UTC.")

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

# CORS must be outermost (added last so it runs first), ensuring preflight
# OPTIONS requests are handled before rate limiting or other middleware sees them.
app.add_middleware(SlowAPIMiddleware)  # applies default_limits to all routes

@app.middleware("http")
async def _security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

from src.fpl_api import router as fpl_router  # noqa: E402
from src.fpl_receipts import router as receipts_router  # noqa: E402
from src.fpl_cards import router as cards_router  # noqa: E402
from src.ucl_api import router as ucl_router  # noqa: E402
app.include_router(fpl_router)
app.include_router(receipts_router)
app.include_router(cards_router)
app.include_router(ucl_router)


def _build_lookup_caches() -> None:
    """Populate team-id and fixture-matchday caches from the DB. Call at startup and after retrain."""
    global _team_id_cache, _fixture_md_cache
    conn = sqlite3.connect(DB_PATH)
    _team_id_cache = {
        _canonical(name): tid
        for tid, name in conn.execute("SELECT team_id, name FROM teams").fetchall()
    }
    _fixture_md_cache = {
        (h, a): md
        for h, a, md in conn.execute("SELECT home_team_id, away_team_id, matchday FROM fixtures").fetchall()
    }
    conn.close()


def _get_predictor() -> DCPredictor:
    if _predictor is None:
        raise HTTPException(
            status_code=503,
            detail=_load_error or "Model not loaded. Run: python wc/scripts/train_dc.py",
        )
    return _predictor


def _resolve_team_id(name: str) -> int | None:
    """Canonical-name match against the in-memory team cache."""
    return _team_id_cache.get(_canonical(name))


def _fixture_matchday(home_id: int, away_id: int) -> int:
    """Return the matchday for a fixture, checking both home/away orderings."""
    return (
        _fixture_md_cache.get((home_id, away_id))
        or _fixture_md_cache.get((away_id, home_id))
        or 0
    )


# ── Request / Response models ─────────────────────────────────────────────────

class PredictRequest(BaseModel):
    home_team: str = Field(..., min_length=1, max_length=100)
    away_team: str = Field(..., min_length=1, max_length=100)
    home_advantage: bool = False


class ScoreLine(BaseModel):
    home_goals: int
    away_goals: int
    probability_pct: float


class ScorerOut(BaseModel):
    name: str
    pos: str
    price_m: float
    first_scorer_pct: float


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
    home_first_pct: float = 0.0
    away_first_pct: float = 0.0
    no_goal_pct: float = 0.0
    home_scorers: list[ScorerOut] = Field(default_factory=list)
    away_scorers: list[ScorerOut] = Field(default_factory=list)


# ── Endpoints ─────────────────────────────────────────────────────────────────

_GOAL_SHARE_OUTFIELD = {"DEF": 0.07, "MID": 0.30, "FWD": 0.62}


def _compute_first_scorers(team_id: int, team_xg: float, total_xg: float, top_n: int = 4) -> list[dict]:
    """Return top-N most likely first scorers for a team given match xG totals."""
    if total_xg <= 0 or team_xg <= 0 or team_id is None:
        return []
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT fp.name, fp.position, fp.price,
               COALESCE(AVG(ml.minutes_played), 0) AS avg_mins
        FROM fantasy_players fp
        LEFT JOIN match_lineups ml ON ml.player_name = fp.name
            AND ml.team_name = (SELECT name FROM teams WHERE team_id = ?)
        WHERE fp.team_id = ? AND fp.position IN ('FWD', 'MID', 'DEF') AND fp.price > 0
        GROUP BY fp.name, fp.position, fp.price
    """, [team_id, team_id]).fetchall()
    conn.close()
    if not rows:
        return []
    entries = []
    for name, pos, price, avg_mins in rows:
        pt = min(avg_mins / 90.0, 1.0)
        lam = _GOAL_SHARE_OUTFIELD.get(pos, 0.0) * pt
        entries.append((name, pos, price, lam))
    total_lam = sum(e[3] for e in entries)
    if total_lam <= 0:
        return []
    p_goal = (1.0 - _math.exp(-total_xg)) * 100.0
    scorers = []
    for name, pos, price, raw_lam in entries:
        player_xg = team_xg * (raw_lam / total_lam)
        first_pct = round((player_xg / total_xg) * p_goal, 1)
        scorers.append(ScorerOut(name=name, pos=pos, price_m=round(price / 10.0, 1), first_scorer_pct=first_pct))
    scorers.sort(key=lambda x: x.first_scorer_pct, reverse=True)
    return scorers[:top_n]


@app.get("/health")
@limiter.exempt
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


def _most_likely_outcome_score(result: dict) -> str:
    """Return the top predicted score that matches the most likely WDL outcome.

    The DC tau correction inflates 1-1 enough to make it the raw top score even
    when one team is a clear favourite. Instead we pick the top score within
    whichever WDL bucket (home-win / draw / away-win) the model considers most
    probable.
    """
    win, draw, loss = result["win_pct"], result["draw_pct"], result["loss_pct"]
    best = max(win, draw, loss)
    scores = result["most_likely"]
    if best == draw:
        top = next((s for s in scores if s[0] == s[1]), scores[0])
    elif best == win:
        top = next((s for s in scores if s[0] > s[1]), scores[0])
    else:
        top = next((s for s in scores if s[1] > s[0]), scores[0])
    return f"{top[0]}-{top[1]}"


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
        md = _fixture_matchday(home_id, away_id)
        draw_boost = 0.10 if md == 1 else 0.0
        result = predictor.predict(home_id, away_id, home_advantage=req.home_advantage, draw_boost=draw_boost)
    except RuntimeError as e:
        logger.exception("predict failed for %s vs %s", req.home_team, req.away_team)
        raise HTTPException(status_code=400, detail="Prediction failed. Check team names and try again.")

    home_xg = result["home_xg"]
    away_xg = result["away_xg"]
    total_xg = home_xg + away_xg
    p_no_goal = round(_math.exp(-total_xg) * 100.0, 1) if total_xg > 0 else 100.0
    p_goal = 100.0 - p_no_goal
    home_first_pct = round((home_xg / total_xg) * p_goal, 1) if total_xg > 0 else 0.0
    away_first_pct = round((away_xg / total_xg) * p_goal, 1) if total_xg > 0 else 0.0

    return PredictResponse(
        home_name=result["home_name"],
        away_name=result["away_name"],
        home_xg=home_xg,
        away_xg=away_xg,
        predicted_score=_most_likely_outcome_score(result),
        win_pct=result["win_pct"],
        draw_pct=result["draw_pct"],
        loss_pct=result["loss_pct"],
        most_likely=[
            ScoreLine(home_goals=h, away_goals=a, probability_pct=p)
            for h, a, p in result["most_likely"]
        ],
        home_first_pct=home_first_pct,
        away_first_pct=away_first_pct,
        no_goal_pct=p_no_goal,
        home_scorers=_compute_first_scorers(home_id, home_xg, total_xg),
        away_scorers=_compute_first_scorers(away_id, away_xg, total_xg),
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
    booster: Optional[Literal["12th_man", "max_captain", "qualification_booster", "clean_sheet_shield"]] = None
    locked_player_ids: list[int] = Field(default_factory=list)
    locked_starter_ids: list[int] = Field(default_factory=list)
    matchdays: Optional[list[int]] = Field(default=None, description="Matchdays to optimise for. None = knockout projection mode.")
    per_team_cap: int = Field(default=3, ge=1, le=15)


class PlayersResponse(BaseModel):
    players: list[FantasyPlayerOut]


class QualBonusEntry(BaseModel):
    name: str
    team: str
    pos: str
    qual_bonus: float


class CSShieldEntry(BaseModel):
    name: str
    team: str
    pos: str
    concede_saved: float


class OptimiseResponse(BaseModel):
    squad: list[FantasyPlayerOut]
    starters: list[FantasyPlayerOut]
    bench: list[FantasyPlayerOut]
    total_pts: float
    total_cost: int
    total_cost_m: float
    captain: FantasyPlayerOut
    booster: Optional[str] = None
    model_trained_at: Optional[str] = None
    twelfth_man: Optional[FantasyPlayerOut] = None
    max_cap_candidates: Optional[list[FantasyPlayerOut]] = None
    expected_max_cap_pts: Optional[float] = None
    qual_booster_breakdown: Optional[list[QualBonusEntry]] = None
    qual_booster_total: Optional[float] = None
    cs_shield_breakdown: Optional[list[CSShieldEntry]] = None
    cs_shield_total: Optional[float] = None


class CaptainsResponse(BaseModel):
    picks: list[FantasyPlayerOut]


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
        res = _optimise(budget=req.budget, predictor=_predictor, booster=req.booster,
                        locked_player_ids=req.locked_player_ids or [],
                        locked_starter_ids=req.locked_starter_ids or [],
                        matchdays=req.matchdays or None,
                        per_team_cap=req.per_team_cap)
    except RuntimeError:
        logger.exception("fantasy optimise failed")
        raise HTTPException(status_code=503, detail="Optimisation unavailable. Try again shortly.")
    captain_id = res["captain"]["id"]
    squad    = [_fmt_player(p, is_captain=(p["id"] == captain_id)) for p in res["squad"]]
    starters = [_fmt_player(p, is_captain=(p["id"] == captain_id)) for p in res["starters"]]
    bench    = [_fmt_player(p) for p in res["bench"]]

    twelfth_man = _fmt_player(res["twelfth_man"]) if res.get("twelfth_man") else None
    max_cap_candidates = [_fmt_player(p) for p in res.get("max_cap_candidates") or []]

    cs_raw = res.get("cs_shield_breakdown") or []
    cs_shield = [CSShieldEntry(name=e["name"], team=e["team"], pos=e["pos"], concede_saved=e["concede_saved"]) for e in cs_raw] or None

    return OptimiseResponse(
        squad=squad,
        starters=starters,
        bench=bench,
        total_pts=res["total_pts"],
        total_cost=res["total_cost"],
        total_cost_m=round(res["total_cost"] / 10, 1),
        captain=_fmt_player(res["captain"], is_captain=True),
        booster=req.booster,
        model_trained_at=res.get("model_trained_at"),
        twelfth_man=twelfth_man,
        max_cap_candidates=max_cap_candidates or None,
        expected_max_cap_pts=res.get("expected_max_cap_pts"),
        qual_booster_breakdown=res.get("qual_booster_breakdown"),
        qual_booster_total=res.get("qual_booster_total"),
        cs_shield_breakdown=cs_shield,
        cs_shield_total=res.get("cs_shield_total"),
    )


@app.get("/api/wc/fantasy/players", response_model=PlayersResponse)
@limiter.limit("20/minute")
def fantasy_players_list(request: Request):
    try:
        players = _get_projected_players(predictor=_predictor)
        return PlayersResponse(players=[_fmt_player(p) for p in players])
    except Exception:
        logger.exception("fantasy players list failed")
        raise HTTPException(status_code=503, detail="Player data unavailable.")


@app.get("/api/wc/fantasy/captains", response_model=CaptainsResponse)
@limiter.limit("20/minute")
def fantasy_captains(request: Request, top_n: int = 10, matchday: int | None = None):
    top_n = min(max(top_n, 1), 20)
    if matchday is not None:
        matchday = max(1, min(3, matchday))
    cache_key = (top_n, matchday)
    cached = _captains_cache.get(cache_key)
    if cached and time.time() - cached[0] < _CAPTAINS_CACHE_TTL:
        return CaptainsResponse(picks=cached[1])
    picks = _captain_picks(top_n=top_n, predictor=_predictor, matchday=matchday)
    formatted = [_fmt_player(p) for p in picks]
    _captains_cache[cache_key] = (time.time(), formatted)
    return CaptainsResponse(picks=formatted)



# ── Tournament simulator ──────────────────────────────────────────────────────

class TournamentTeamOut(BaseModel):
    team: str
    group: str
    group_1st_pct: float = 0.0
    r32_pct: float
    r16_pct: float = 0.0
    qf_pct: float
    sf_pct: float
    final_pct: float
    win_pct: float
    third_place_pct: float = 0.0


class TournamentResponse(BaseModel):
    teams: list[TournamentTeamOut]
    n_sim: int
    last_updated: str | None = None


_sim_lock = threading.Lock()
_sim_running: set[int] = set()


def _run_simulation_bg(predictor: DCPredictor, n_sim: int) -> None:
    """Run tournament simulation in background and populate cache."""
    try:
        results = predictor.simulate_tournament(n_sim=n_sim)
        _sim_cache[n_sim] = (time.time(), results)
    except Exception as e:
        logger.exception("Background simulation failed (n_sim=%d): %s", n_sim, e)
    finally:
        _sim_running.discard(n_sim)


def _inject_third_place_pcts(results: list[dict], predictor) -> None:
    """Use a direct DC prediction to set win probability for the two SF losers."""
    tp = [r for r in results if r.get("sf_pct", 0) >= 99 and r.get("final_pct", 0) <= 1]
    if len(tp) != 2:
        return
    id_a = _resolve_team_id(tp[0]["team"])
    id_b = _resolve_team_id(tp[1]["team"])
    if not id_a or not id_b:
        return
    try:
        pred = predictor.predict(id_a, id_b, knockout=True)
        tp[0]["third_place_pct"] = pred["ko_win_pct"]
        tp[1]["third_place_pct"] = pred["ko_loss_pct"]
    except Exception:
        pass


@app.get("/api/wc/simulate", response_model=TournamentResponse)
@limiter.limit("10/minute")
def simulate_tournament(request: Request, n_sim: int = 10_000):
    """
    Monte Carlo tournament simulation (default 10 000 runs).
    Returns win/final/SF/QF/R32 probabilities for all 48 teams.
    Results reflect the current DC model — retrain after each matchday for live updates.
    Results are cached for 30 minutes. First call after cache expiry returns the previous
    cached result immediately and kicks off a background refresh.
    """
    n_sim = min(max(n_sim, 1_000), 20_000)
    predictor = _get_predictor()

    cached = _sim_cache.get(n_sim)
    cache_fresh = cached and (time.time() - cached[0]) < _SIM_CACHE_TTL

    if not cache_fresh and n_sim not in _sim_running:
        _sim_running.add(n_sim)
        threading.Thread(target=_run_simulation_bg, args=(predictor, n_sim), daemon=True).start()

    if cached:
        _inject_third_place_pcts(cached[1], predictor)
        data = TournamentResponse(
            teams=[TournamentTeamOut(**r) for r in cached[1]],
            n_sim=n_sim,
            last_updated=_retrain_status.get("last_time"),
        )
        return JSONResponse(content=data.model_dump(), headers={"Cache-Control": "no-store"})

    # No cache at all yet — block once to get initial data
    if n_sim in _sim_running:
        _sim_running.discard(n_sim)
    results = predictor.simulate_tournament(n_sim=n_sim)
    _inject_third_place_pcts(results, predictor)
    _sim_cache[n_sim] = (time.time(), results)
    data = TournamentResponse(
        teams=[TournamentTeamOut(**r) for r in results],
        n_sim=n_sim,
        last_updated=_retrain_status.get("last_time"),
    )
    return JSONResponse(content=data.model_dump(), headers={"Cache-Control": "no-store"})


# ── Knockout bracket state ────────────────────────────────────────────────────

class BracketMatchOut(BaseModel):
    team_a: str
    team_b: str
    score: str | None = None
    winner: str | None = None
    status: Literal["played", "pending"]
    team_a_win_pct: float | None = None
    team_b_win_pct: float | None = None


@app.get("/api/wc/bracket")
@limiter.limit("30/minute")
def get_bracket(request: Request):
    """
    Derive current SF/Final bracket state from simulation probabilities + played results.
    SF teams = sf_pct >= 99. Finalists = final_pct >= 99. 3rd place = sf_pct >= 99 and final_pct <= 1.
    """
    predictor = _get_predictor()

    cached = _sim_cache.get(10_000)
    if cached:
        sim_teams = [TournamentTeamOut(**r) for r in cached[1]]
    else:
        results = predictor.simulate_tournament(n_sim=10_000)
        _sim_cache[10_000] = (time.time(), results)
        sim_teams = [TournamentTeamOut(**r) for r in results]

    team_probs = {_canonical(t.team): t for t in sim_teams}
    sf_teams = {name: t for name, t in team_probs.items() if t.sf_pct >= 99}

    conn = sqlite3.connect(str(DB_PATH))
    ko_rows = conn.execute(
        "SELECT home_team, away_team, home_score, away_score FROM match_stats WHERE match_date >= '2026-06-28'"
    ).fetchall()
    conn.close()

    sf_results: list[BracketMatchOut] = []
    played_canon: set[str] = set()
    for home, away, hs, as_ in ko_rows:
        h_c, a_c = _canonical(home), _canonical(away)
        if h_c in sf_teams and a_c in sf_teams:
            winner = home if int(hs) > int(as_) else away
            sf_results.append(BracketMatchOut(
                team_a=home, team_b=away,
                score=f"{hs}-{as_}", winner=winner, status="played",
            ))
            played_canon.update([h_c, a_c])

    pending_names = [name for name in sf_teams if name not in played_canon]
    sf_matches = sf_results[:]
    if len(pending_names) >= 2:
        ta, tb = sf_teams[pending_names[0]], sf_teams[pending_names[1]]
        total = ta.final_pct + tb.final_pct
        ta_win = round(ta.final_pct / total * 100, 1) if total > 0 else 50.0
        tb_win = round(tb.final_pct / total * 100, 1) if total > 0 else 50.0
        sf_matches.append(BracketMatchOut(
            team_a=ta.team, team_b=tb.team, status="pending",
            team_a_win_pct=ta_win, team_b_win_pct=tb_win,
        ))

    finalists = [
        {"team": t.team, "win_pct": round(t.win_pct, 1)}
        for t in sf_teams.values() if t.final_pct >= 99
    ]
    pending_finalists = [
        {"team": t.team, "win_pct": round(t.win_pct, 1), "reach_final_pct": round(t.final_pct, 1)}
        for t in sf_teams.values() if 1 < t.final_pct < 99
    ]
    third_place = [t.team for t in sf_teams.values() if t.final_pct <= 1]

    return JSONResponse(content={
        "sf_matches": [m.model_dump() for m in sf_matches],
        "finalists": finalists,
        "pending_finalists": pending_finalists,
        "third_place_teams": third_place,
    }, headers={"Cache-Control": "no-store"})


# ── Group standings + lock status ────────────────────────────────────────────

class TeamStandingOut(BaseModel):
    team: str
    pos: int
    pts: int
    gd: int
    pos_locked: bool
    qualified_locked: bool
    top2_locked_out: bool

class StandingsResponse(BaseModel):
    groups: dict[str, list[TeamStandingOut]]


def _calc_standings(keys: list, played_map: dict, extra: dict | None = None):
    """Return (ranked_keys, stats) where stats[k]=[pts,gd,gf,w,d,l]."""
    all_res = {**played_map, **(extra or {})}
    stats = {k: [0, 0, 0, 0, 0, 0] for k in keys}
    h2h   = {k: {o: [0, 0] for o in keys if o != k} for k in keys}
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            h, a = keys[i], keys[j]
            pair = all_res.get((h, a)) or all_res.get((a, h))
            if pair is None:
                continue
            hg, ag = pair if (h, a) in all_res else (pair[1], pair[0])
            diff = hg - ag
            stats[h][2] += hg; stats[h][1] += diff
            stats[a][2] += ag; stats[a][1] -= diff
            if diff > 0:
                stats[h][0] += 3; stats[h][3] += 1; stats[a][5] += 1
                h2h[h][a][0] = 3; h2h[a][h][0] = 0
            elif diff < 0:
                stats[a][0] += 3; stats[a][3] += 1; stats[h][5] += 1
                h2h[h][a][0] = 0; h2h[a][h][0] = 3
            else:
                stats[h][0] += 1; stats[h][4] += 1
                stats[a][0] += 1; stats[a][4] += 1
                h2h[h][a][0] = 1; h2h[a][h][0] = 1
            h2h[h][a][1] = diff; h2h[a][h][1] = -diff
    def sk(k):
        pts  = stats[k][0]
        tied = [o for o in keys if o != k and stats[o][0] == pts]
        return (pts, sum(h2h[k][o][0] for o in tied), sum(h2h[k][o][1] for o in tied), stats[k][1], stats[k][2])
    return sorted(keys, key=sk, reverse=True), stats


@app.get("/api/wc/standings", response_model=StandingsResponse)
@limiter.limit("30/minute")
def get_standings(request: Request):
    """
    Current group standings in live order with mathematical lock status.
    Enumerates all 9 possible MD3 outcomes per group (H/D/A for each fixture)
    to determine which positions are mathematically fixed.
    """
    WC2026_GROUPS = DCPredictor.WC2026_GROUPS
    conn = sqlite3.connect(str(DB_PATH))
    played = conn.execute(
        "SELECT home_team, away_team, home_score, away_score FROM match_stats"
    ).fetchall()
    conn.close()

    played_map: dict = {}
    for h, a, hs, as_ in played:
        played_map[(_canonical(h), _canonical(a))] = (int(hs), int(as_))

    _OUTCOMES = [(1, 0), (0, 0), (0, 1)]
    groups_out: dict[str, list[TeamStandingOut]] = {}

    for group_name, teams in WC2026_GROUPS.items():
        keys    = [_canonical(t) for t in teams]
        display = {_canonical(t): t for t in teams}
        ranked, stats = _calc_standings(keys, played_map)
        # Derive unplayed pairs directly from played_map — avoids stale fixtures table.
        fixtures = [
            (keys[i], keys[j])
            for i in range(len(keys))
            for j in range(i + 1, len(keys))
            if (keys[i], keys[j]) not in played_map and (keys[j], keys[i]) not in played_map
        ]
        combos     = list(itertools.product(_OUTCOMES, repeat=len(fixtures)))
        all_orders = [_calc_standings(keys, played_map, dict(zip(fixtures, c)))[0] for c in combos]

        group_list = []
        for pos, tk in enumerate(ranked, 1):
            s = stats[tk]
            all_pos  = [o.index(tk) + 1 for o in all_orders]
            min_p, max_p = min(all_pos), max(all_pos)
            group_list.append(TeamStandingOut(
                team=display[tk],
                pos=pos,
                pts=s[0],
                gd=s[1],
                pos_locked=min_p == max_p,
                qualified_locked=max_p <= 2,
                top2_locked_out=min_p > 2,
            ))
        groups_out[group_name] = group_list

    return StandingsResponse(groups=groups_out)


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
        conn = sqlite3.connect(SUBSCRIBERS_DB)
        conn.execute("INSERT INTO subscribers (email) VALUES (?)", (req.email,))
        conn.commit()
        conn.close()
    except sqlite3.IntegrityError:
        pass
    except Exception:
        logger.exception("Failed to save subscriber")
    return _OK


def _csv_safe(val: str) -> str:
    """Neutralise CSV formula injection by prefixing formula triggers with a tab."""
    return ("\t" + val) if val and val[0] in ("=", "+", "-", "@") else val


@app.get("/api/notify/export")
def export_subscribers(token: str | None = Depends(_retrain_header)):
    """Download all subscriber emails as CSV. Requires the retrain token."""
    _require_retrain_token(token)
    conn = sqlite3.connect(SUBSCRIBERS_DB)
    rows = conn.execute("SELECT email, created_at FROM subscribers ORDER BY created_at").fetchall()
    conn.close()
    lines = ["email,created_at"] + [f"{_csv_safe(r[0])},{r[1]}" for r in rows]
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse("\n".join(lines), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=subscribers.csv"})


# ── Live retrain ──────────────────────────────────────────────────────────────

_retrain_lock = threading.Lock()
_retrain_status: dict = {"running": False, "last": None, "error": None}


def _run_retrain() -> None:
    global _predictor
    try:
        subprocess.run(
            [sys.executable, "wc/scripts/ingest_recent_form.py", "--months", "30"],
            check=True, capture_output=True, cwd=str(PROJECT_ROOT),
        )
        subprocess.run(
            [sys.executable, "wc/scripts/train_dc.py"],
            check=True, capture_output=True, cwd=str(PROJECT_ROOT),
        )
        # Refresh player prices from FIFA API alongside model retrain
        try:
            from wc.scripts.ingest_fantasy_players import ingest_from_fifa_api as _ingest_players
            _ingest_players()
        except Exception as _pe:
            print(f"[retrain] player price refresh failed (non-fatal): {_pe}")
        p = DCPredictor()
        p.load()
        _predictor = p
        _sim_cache.clear()
        _captains_cache.clear()
        _build_lookup_caches()
        _retrain_status["last"] = "success"
        _retrain_status["last_time"] = _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        _retrain_status["error"] = None
        print("[retrain] DC model refreshed successfully.")
    except subprocess.CalledProcessError as e:
        _retrain_status["error"] = e.stderr.decode(errors="replace")[-500:]
        print(f"[retrain] failed: {_retrain_status['error']}")
    finally:
        _retrain_status["running"] = False
        _retrain_lock.release()


def _daily_retrain_loop() -> None:
    """Daemon thread: retrain once per day at 06:00 UTC."""
    while True:
        now = _dt.datetime.utcnow()
        next_run = now.replace(hour=6, minute=0, second=0, microsecond=0)
        if next_run <= now:
            next_run += _dt.timedelta(days=1)
        time.sleep((next_run - now).total_seconds())
        if _retrain_lock.acquire(blocking=False):
            _retrain_status["running"] = True
            threading.Thread(target=_run_retrain, daemon=True).start()
        # If lock not acquired, a manual retrain is already running — skip this cycle


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
    return {"status": "started", "message": "Retraining in background."}


@app.get("/api/wc/retrain/status")
def retrain_status():
    """Returns the result of the most recent retrain run."""
    model_trained_at = getattr(_predictor, "trained_at", None) if _predictor else None
    return {
        "running":          _retrain_status["running"],
        "last":             _retrain_status["last"],
        "last_triggered":   _retrain_status.get("last_time"),
        "model_trained_at": model_trained_at,
        "error":            _retrain_status["error"] is not None,  # bool — don't leak stderr
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port)
