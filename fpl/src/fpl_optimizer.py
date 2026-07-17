"""
fpl/src/fpl_optimizer.py
FPL squad optimizer using Dixon-Coles fixture difficulty + historical player rates.

Point projection:
  1. Compute each player's per-GW averages from player_gameweek_history
     (goals, assists, CS rate, clean_sheets, bonus, minutes)
  2. Get upcoming fixture for each team via DC model:
       xg_adj = DC_predicted_xg_for / team_season_avg_xg_for
       xga_adj = DC_predicted_xg_against / team_season_avg_xga
  3. Scale base rates by fixture difficulty adjustments
  4. Convert to fantasy points via FPL scoring rules

Squad optimization uses scipy.optimize.milp (binary MILP) — same structure
as WC optimizer.

FPL scoring (2025/26 rules):
  Minutes 1-59:   1 pt (appearance)
  Minutes 60+:    2 pts
  CS:             GK/DEF = 4 pts,  MID = 1 pt,  FWD = 0
  Goal:           GK = 10, DEF = 6, MID = 5, FWD = 4
  Assist:         3 pts
  Saves per 3:    1 pt (GK only)
  2 goals conceded: -1 pt (GK/DEF only)
  Yellow card:    -1 pt
  Bonus:          1-3 pts (historical avg included)
"""

import json
import sqlite3
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH      = PROJECT_ROOT / "data" / "fpl.db"
MODEL_PATH   = Path(__file__).resolve().parent.parent / "models" / "fpl_dc_params.json"

# ── FPL scoring constants ─────────────────────────────────────────────────────
PT_APP60    = 2.0   # pts for 60+ min (includes base 1 + bonus minute pt)
PT_APP_SUB  = 1.0   # pts for <60 min appearance
PT_CS    = {"GK": 4.0, "DEF": 4.0, "MID": 1.0, "FWD": 0.0}
PT_GOAL  = {"GK": 10.0, "DEF": 6.0, "MID": 5.0, "FWD": 4.0}
PT_ASSIST = 3.0
PT_YELLOW = -1.0
PT_SAVE3  = 1.0    # per 3 saves block
PT_CONCEDE = -1.0  # per 2 goals conceded by GK/DEF (i.e., -0.5 per goal beyond clean sheet)

SQUAD_RULES = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
BUDGET_DEFAULT = 1000   # £100.0m
PER_TEAM_CAP   = 3

# Position map: FPL API position codes → strings
POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

# Minimum minutes threshold to count a GW appearance in the historical sample
MIN_APP_MINUTES = 1


def _load_dc() -> dict:
    if not MODEL_PATH.exists():
        return {}
    return json.loads(MODEL_PATH.read_text())


def _load_players(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("""
        SELECT p.player_id, p.web_name, p.first_name || ' ' || p.second_name AS full_name,
               p.team_id, p.position, p.current_cost,
               t.name AS team_name, t.short_name AS team_short
        FROM   players p
        JOIN   teams   t ON p.team_id = t.team_id
        WHERE  p.position IS NOT NULL AND p.current_cost IS NOT NULL
          AND  p.current_cost > 0
    """).fetchall()

    out = []
    for pid, web_name, full_name, team_id, pos_code, cost, team_name, team_short in rows:
        pos = POSITION_MAP.get(pos_code, "MID")
        out.append({
            "id":         pid,
            "name":       web_name,
            "full_name":  full_name,
            "team_id":    team_id,
            "team":       team_name or "Unknown",
            "team_short": team_short or "???",
            "pos":        pos,
            "price":      cost,
        })
    return out


def _compute_player_rates(conn: sqlite3.Connection, min_gws: int = 3) -> dict[int, dict]:
    """
    Compute per-GW averages from player_gameweek_history.
    Returns {player_id: {avg_minutes, avg_goals, avg_assists, cs_rate,
                         avg_bonus, avg_saves_est, avg_yellow, gw_count}}
    """
    rows = conn.execute("""
        SELECT player_id,
               AVG(minutes)               AS avg_min,
               AVG(goals_scored)          AS avg_goals,
               AVG(assists)               AS avg_assists,
               AVG(CASE WHEN clean_sheets = 1 AND minutes >= 60 THEN 1.0 ELSE 0.0 END) AS cs_rate,
               AVG(bonus)                 AS avg_bonus,
               AVG(total_points)          AS avg_pts,
               COUNT(*)                   AS gw_count,
               AVG(expected_goals)        AS avg_xg,
               AVG(expected_assists)      AS avg_xa
        FROM   player_gameweek_history
        WHERE  minutes >= ?
        GROUP  BY player_id
        HAVING COUNT(*) >= ?
    """, (MIN_APP_MINUTES, min_gws)).fetchall()

    out: dict[int, dict] = {}
    for pid, avg_min, avg_goals, avg_assists, cs_rate, avg_bonus, avg_pts, gw_count, avg_xg, avg_xa in rows:
        # Estimate avg saves from team xGA — will be refined per-fixture
        out[pid] = {
            "avg_minutes":  float(avg_min or 0),
            "avg_goals":    float(avg_goals or 0),
            "avg_assists":  float(avg_assists or 0),
            "cs_rate":      float(cs_rate or 0),
            "avg_bonus":    float(avg_bonus or 0),
            "avg_pts":      float(avg_pts or 0),
            "gw_count":     int(gw_count),
            "avg_xg":       float(avg_xg or 0),
            "avg_xa":       float(avg_xa or 0),
        }
    return out


def _compute_team_season_avgs(conn: sqlite3.Connection) -> dict[int, dict]:
    """
    Per-team season averages: avg xG scored and conceded per GW.
    Used as baseline for DC fixture difficulty multipliers.
    """
    rows = conn.execute("""
        SELECT f.home_team_id, f.away_team_id,
               p_home.avg_xg_for, p_home.avg_xg_against,
               p_away.avg_xg_for, p_away.avg_xg_against
        FROM (
            SELECT home_team_id AS team_id,
                   AVG(home_score) AS avg_xg_for,
                   AVG(away_score) AS avg_xg_against
            FROM   fixtures WHERE finished = 1
            GROUP  BY home_team_id
        ) p_home
        JOIN (
            SELECT away_team_id AS team_id,
                   AVG(away_score) AS avg_xg_for,
                   AVG(home_score) AS avg_xg_against
            FROM   fixtures WHERE finished = 1
            GROUP  BY away_team_id
        ) p_away ON p_home.team_id = p_away.team_id
        JOIN fixtures f ON f.home_team_id = p_home.team_id LIMIT 1
    """).fetchall()

    # Simpler: just home/away combined
    rows = conn.execute("""
        SELECT team_id,
               AVG(xg_for)      AS avg_xg_for,
               AVG(xg_against)  AS avg_xg_against
        FROM (
            SELECT home_team_id AS team_id,
                   CAST(home_score AS REAL) AS xg_for,
                   CAST(away_score AS REAL) AS xg_against
            FROM   fixtures WHERE finished = 1
            UNION ALL
            SELECT away_team_id,
                   CAST(away_score AS REAL),
                   CAST(home_score AS REAL)
            FROM   fixtures WHERE finished = 1
        )
        GROUP  BY team_id
    """).fetchall()

    return {
        int(tid): {"avg_xg_for": float(xgf or 1.2), "avg_xg_against": float(xga or 1.2)}
        for tid, xgf, xga in rows
    }


def _get_fixture_dc_adjustments(
    home_team_id: int,
    away_team_id: int,
    dc: dict,
    team_avgs: dict[int, dict],
    is_home: bool,
) -> tuple[float, float]:
    """
    Returns (xg_for_adj, xg_against_adj) for a given team in a given fixture.
    xg_for_adj > 1.0 means the fixture is easier to score in vs season average.
    """
    team_params = dc.get("team_params", {})
    form_adj    = dc.get("form_adjustments", {})
    home_adv    = dc.get("home_adv", 1.20)

    if not team_params:
        return 1.0, 1.0

    mean_atk = float(np.mean([v["attack"]  for v in team_params.values()]))
    mean_def = float(np.mean([v["defense"] for v in team_params.values()]))

    def _p(tid: int) -> dict:
        return team_params.get(str(tid), {"attack": mean_atk, "defense": mean_def})

    def _form(tid: int) -> tuple[float, float]:
        f = form_adj.get(str(tid), [1.0, 1.0])
        return tuple(f) if len(f) == 2 else (1.0, 1.0)

    h_p = _p(home_team_id); a_p = _p(away_team_id)
    h_atk_m, h_def_m = _form(home_team_id)
    a_atk_m, a_def_m = _form(away_team_id)

    mu_h = h_p["attack"] * h_atk_m * a_p["defense"] * a_def_m * home_adv
    mu_a = a_p["attack"] * a_atk_m * h_p["defense"] * h_def_m

    team_id = home_team_id if is_home else away_team_id
    avgs = team_avgs.get(team_id, {"avg_xg_for": 1.2, "avg_xg_against": 1.2})

    if is_home:
        xg_for_adj     = mu_h / max(avgs["avg_xg_for"], 0.5)
        xg_against_adj = mu_a / max(avgs["avg_xg_against"], 0.5)
    else:
        xg_for_adj     = mu_a / max(avgs["avg_xg_for"], 0.5)
        xg_against_adj = mu_h / max(avgs["avg_xg_against"], 0.5)

    # Cap adjustments to avoid extreme values
    xg_for_adj     = max(0.3, min(2.5, xg_for_adj))
    xg_against_adj = max(0.3, min(2.5, xg_against_adj))
    return xg_for_adj, xg_against_adj


def _cs_probability(mu_against: float) -> float:
    """P(team concedes 0 goals) given DC-expected goals against = mu_against."""
    from scipy.stats import poisson
    return float(poisson.pmf(0, max(mu_against, 0.01)))


def _project_players(
    players: list[dict],
    dc: dict,
    team_avgs: dict[int, dict],
    fixtures: list[tuple[int, int]] | None = None,
    gameweek_id: int | None = None,
    conn: sqlite3.Connection | None = None,
) -> list[dict]:
    """
    Project fantasy points for each player.

    fixtures: list of (home_team_id, away_team_id) for the upcoming GW.
              If None, uses DC season averages (no fixture-specific adjustment).
    gameweek_id: if given and conn provided, loads fixtures from DB for that GW.
    """
    # Build team → fixture DC adjustments
    team_fixture_adj: dict[int, tuple[float, float]] = {}

    if gameweek_id is not None and conn is not None:
        rows = conn.execute("""
            SELECT home_team_id, away_team_id
            FROM   fixtures
            WHERE  gameweek_id = ?
        """, (gameweek_id,)).fetchall()
        fixtures = [(int(h), int(a)) for h, a in rows]

    if fixtures:
        for home_id, away_id in fixtures:
            team_fixture_adj[home_id] = _get_fixture_dc_adjustments(
                home_id, away_id, dc, team_avgs, is_home=True
            )
            team_fixture_adj[away_id] = _get_fixture_dc_adjustments(
                home_id, away_id, dc, team_avgs, is_home=False
            )

    team_params = dc.get("team_params", {})
    form_adj    = dc.get("form_adjustments", {})
    home_adv    = dc.get("home_adv", 1.20)

    mean_atk = float(np.mean([v["attack"]  for v in team_params.values()])) if team_params else 1.0
    mean_def = float(np.mean([v["defense"] for v in team_params.values()])) if team_params else 1.0

    for p in players:
        rates  = p.get("_rates", {})
        tid    = p["team_id"]
        pos    = p["pos"]

        avg_min     = rates.get("avg_minutes", 45.0)
        avg_goals   = rates.get("avg_goals", 0.0)
        avg_assists = rates.get("avg_assists", 0.0)
        cs_rate     = rates.get("cs_rate", 0.0)
        avg_bonus   = rates.get("avg_bonus", 0.5)
        avg_xg      = rates.get("avg_xg", avg_goals)
        avg_xa      = rates.get("avg_xa", avg_assists)

        # Starter probability from minutes history
        if avg_min >= 75:
            start_prob = 0.95
        elif avg_min >= 60:
            start_prob = 0.85
        elif avg_min >= 45:
            start_prob = 0.65
        elif avg_min >= 25:
            start_prob = 0.40
        else:
            start_prob = 0.15

        # Appearance points
        p60_prob    = start_prob * max(0, min(1, (avg_min - 30) / 40))
        app_pts     = p60_prob * PT_APP60 + (start_prob - p60_prob) * PT_APP_SUB

        # Fixture difficulty adjustments
        xg_adj, xga_adj = team_fixture_adj.get(tid, (1.0, 1.0))

        # Expected DC xGA for CS probability
        if team_params:
            tp = team_params.get(str(tid), {"attack": mean_atk, "defense": mean_def})
            fa = form_adj.get(str(tid), [1.0, 1.0])
            _adjusted_xga = tp["defense"] * fa[1] * mean_atk * xga_adj
        else:
            _adjusted_xga = 1.2 * xga_adj

        fixture_cs_prob = _cs_probability(_adjusted_xga)

        # CS points — use fixture-specific CS probability, scaled by start prob
        if pos in ("GK", "DEF"):
            cs_pts = fixture_cs_prob * PT_CS[pos] * p60_prob
        elif pos == "MID":
            cs_pts = fixture_cs_prob * PT_CS["MID"] * p60_prob
        else:
            cs_pts = 0.0

        # Concede penalty (GK/DEF): -1 per 2 goals conceded
        if pos in ("GK", "DEF"):
            exp_goals_conceded = _adjusted_xga * xga_adj
            concede_pts = max(0.0, (exp_goals_conceded - 1.0) * 0.5) * PT_CONCEDE * p60_prob
        else:
            concede_pts = 0.0

        # Goal pts: use xG when available, else historical goals
        eff_xg = avg_xg if avg_xg > 0 else avg_goals
        goal_pts = eff_xg * xg_adj * PT_GOAL[pos] * start_prob

        # Assist pts
        eff_xa = avg_xa if avg_xa > 0 else avg_assists
        assist_pts = eff_xa * xg_adj * PT_ASSIST * start_prob

        # Save pts (GK only)
        if pos == "GK":
            exp_shots_on_target = max(0, _adjusted_xga * 2.5)  # rough SoT estimate
            exp_saves = max(0, exp_shots_on_target - _adjusted_xga)
            save_pts = (exp_saves / 3.0) * PT_SAVE3 * p60_prob
        else:
            save_pts = 0.0

        # Bonus pts
        bonus_pts = avg_bonus * start_prob

        projected = app_pts + cs_pts + concede_pts + goal_pts + assist_pts + save_pts + bonus_pts

        p["projected_pts"] = round(max(0.0, projected), 2)
        p["start_prob"]    = round(start_prob, 2)
        p["fixture_cs_pct"] = round(fixture_cs_prob * 100, 1)
        p["xg_adj"]        = round(xg_adj, 2)

    return players


def optimise(
    budget: int = BUDGET_DEFAULT,
    gameweek_id: int | None = None,
    fixtures: list[tuple[int, int]] | None = None,
    locked_player_ids: list[int] | None = None,
    excluded_player_ids: list[int] | None = None,
    per_team_cap: int = PER_TEAM_CAP,
    existing_squad_ids: list[int] | None = None,
    free_transfers: int = 1,
    transfer_penalty: int = 4,
) -> dict:
    """
    Select the optimal 15-player FPL squad.

    gameweek_id: load upcoming fixtures for this GW from DB.
    fixtures: explicit list of (home_id, away_id) — overrides gameweek_id.
    locked_player_ids: players that must be in the squad.
    excluded_player_ids: players to exclude entirely.
    existing_squad_ids: current squad — used to compute transfer penalties.
    free_transfers: number of free transfers available (default 1).
    transfer_penalty: points deducted per extra transfer beyond free_transfers.
    """
    conn    = sqlite3.connect(DB_PATH)
    players = _load_players(conn)

    # Attach historical rates
    rates = _compute_player_rates(conn)
    for p in players:
        p["_rates"] = rates.get(p["id"], {})

    team_avgs = _compute_team_season_avgs(conn)

    dc = _load_dc()

    players = _project_players(
        players, dc, team_avgs,
        fixtures=fixtures, gameweek_id=gameweek_id, conn=conn,
    )
    conn.close()

    # Apply exclusions
    excluded_ids = set(excluded_player_ids or [])
    players = [p for p in players if p["id"] not in excluded_ids]

    n      = len(players)
    pts    = np.array([p["projected_pts"] for p in players], dtype=float)
    prices = np.array([p["price"]         for p in players], dtype=float)
    pos_l  = [p["pos"]  for p in players]
    teams  = [p["team"] for p in players]

    # Transfer penalty: adjust pts for players NOT in existing squad beyond free transfers
    if existing_squad_ids:
        existing_set = set(existing_squad_ids)
        for i, p in enumerate(players):
            if p["id"] not in existing_set:
                # Will incur a penalty if this is an extra transfer
                # Penalise uniformly; solver will weigh this against quality gain
                if free_transfers <= 0:
                    pts[i] -= transfer_penalty

    # Variables: [x_0..x_{n-1}, s_0..s_{n-1}, c_0..c_{n-1}]
    BENCH_WEIGHT = 0.60
    obj = np.concatenate([-pts * BENCH_WEIGHT, -pts * (1 - BENCH_WEIGHT), -pts])

    rows: list[np.ndarray] = []
    lbs:  list[float]      = []
    ubs:  list[float]      = []

    def _xrow(v): return np.concatenate([v,           np.zeros(n), np.zeros(n)])
    def _srow(v): return np.concatenate([np.zeros(n), v,           np.zeros(n)])
    def _crow(v): return np.concatenate([np.zeros(n), np.zeros(n), v          ])

    # ── Cardinality ───────────────────────────────────────────────────────────
    rows.append(_xrow(np.ones(n))); lbs.append(15.0); ubs.append(15.0)
    rows.append(_srow(np.ones(n))); lbs.append(11.0); ubs.append(11.0)
    rows.append(_crow(np.ones(n))); lbs.append(1.0);  ubs.append(1.0)

    # ── Squad position counts ─────────────────────────────────────────────────
    for pos, count in SQUAD_RULES.items():
        v = np.array([1.0 if p == pos else 0.0 for p in pos_l])
        rows.append(_xrow(v)); lbs.append(float(count)); ubs.append(float(count))

    # ── Starting XI formation bounds ──────────────────────────────────────────
    for pos, lo, hi in [("GK", 1, 1), ("DEF", 3, 5), ("MID", 3, 5), ("FWD", 1, 3)]:
        v = np.array([1.0 if p == pos else 0.0 for p in pos_l])
        rows.append(_srow(v)); lbs.append(float(lo)); ubs.append(float(hi))

    # ── Budget ────────────────────────────────────────────────────────────────
    rows.append(_xrow(prices)); lbs.append(0.0); ubs.append(float(budget))

    # ── Per-team cap ──────────────────────────────────────────────────────────
    for team in set(teams):
        v = np.array([1.0 if t == team else 0.0 for t in teams])
        rows.append(_xrow(v)); lbs.append(0.0); ubs.append(float(per_team_cap))

    # ── Hierarchy: c_i <= s_i <= x_i ─────────────────────────────────────────
    I_n = np.eye(n)
    Z_n = np.zeros((n, n))
    for row in np.hstack([-I_n, I_n, Z_n]):
        rows.append(row); lbs.append(-np.inf); ubs.append(0.0)
    for row in np.hstack([Z_n, -I_n, I_n]):
        rows.append(row); lbs.append(-np.inf); ubs.append(0.0)

    A           = np.vstack(rows)
    constraints = LinearConstraint(A, lb=np.array(lbs), ub=np.array(ubs))

    lb_arr = np.zeros(3 * n)
    ub_arr = np.ones(3 * n)
    locked_ids = set(locked_player_ids or [])
    for i, player in enumerate(players):
        if player["id"] in locked_ids:
            lb_arr[i] = 1.0

    result = milp(obj, constraints=constraints,
                  integrality=np.ones(3 * n), bounds=Bounds(lb_arr, ub_arr))

    if result.status != 0:
        raise RuntimeError(f"FPL optimizer failed: {result.message}")

    x_sol = result.x[:n]
    s_sol = result.x[n:2 * n]
    c_sol = result.x[2 * n:]

    pos_order = list(SQUAD_RULES.keys())
    squad = []
    for i, player in enumerate(players):
        if x_sol[i] > 0.5:
            player["is_starter"] = bool(s_sol[i] > 0.5)
            player["is_captain"] = bool(c_sol[i] > 0.5)
            squad.append(player)

    squad.sort(key=lambda p: (
        pos_order.index(p["pos"]),
        not p["is_starter"],
        -p["projected_pts"],
    ))

    captain  = next(p for p in squad if p["is_captain"])
    starters = [p for p in squad if p["is_starter"]]
    bench    = [p for p in squad if not p["is_starter"]]
    total_cost = int(sum(p["price"] for p in squad))

    # Transfer summary if existing squad provided
    transfers_out = transfers_in = []
    if existing_squad_ids:
        squad_ids    = {p["id"] for p in squad}
        existing_set = set(existing_squad_ids)
        transfers_in  = [p for p in squad if p["id"] not in existing_set]
        transfers_out = existing_squad_ids and [pid for pid in existing_squad_ids if pid not in squad_ids]
        n_transfers   = len(transfers_in)
        hit = max(0, n_transfers - free_transfers) * transfer_penalty
    else:
        n_transfers = hit = 0

    dc_meta = json.loads(MODEL_PATH.read_text()) if MODEL_PATH.exists() else {}

    return {
        "squad":           squad,
        "starters":        starters,
        "bench":           bench,
        "captain":         captain,
        "total_pts":       round(sum(p["projected_pts"] for p in starters) + captain["projected_pts"], 1),
        "total_cost":      total_cost,
        "budget_remaining": budget - total_cost,
        "n_transfers":     n_transfers,
        "transfer_hit":    hit,
        "net_pts":         round(sum(p["projected_pts"] for p in starters) + captain["projected_pts"] - hit, 1),
        "model_trained_at": dc_meta.get("trained_at"),
    }
