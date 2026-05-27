"""
fantasy_optimizer.py
Projects fantasy points per player and optimises squad selection.

Point projection uses Monte Carlo simulation (N=50,000 by default):
  - Draws team goals from Poisson distributions using actual WC2026 group
    fixtures and opponent-specific DC attack/defence ratings.
  - Averages binary events (clean sheets, conceding 2+) across all simulations
    so projected points reflect real schedule difficulty, not just mean opponent.

Squad optimization uses scipy.optimize.milp (binary ILP).

WC Fantasy rules:
  Squad:  2 GK + 5 DEF + 5 MID + 3 FWD = 15 players
  Budget: $100m (1000 in $0.1m units)
  Cap:    max 3 players per team
  Projection base: 3 group-stage matches
"""

import json
import sqlite3
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH      = PROJECT_ROOT / "data" / "wc.db"
MODEL_PATH   = PROJECT_ROOT / "models" / "dc_params.json"

SQUAD_RULES    = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
BUDGET_DEFAULT = 1000  # $100.0m

PT_PLAYED  = 1.0
PT_CS      = {"GK": 4.0, "DEF": 4.0, "MID": 1.0, "FWD": 0.0}
PT_GOAL    = {"GK": 6.0, "DEF": 6.0, "MID": 5.0, "FWD": 4.0}
PT_ASSIST  = 3.0
PT_SAVE3   = 1.0   # per 3 saves (GK only)
PT_CONCEDE = -1.0  # per match with 2+ goals conceded (GK/DEF)

GOAL_SHARE   = {"GK": 0.01, "DEF": 0.07, "MID": 0.30, "FWD": 0.62}
ASSIST_SHARE = {"GK": 0.01, "DEF": 0.12, "MID": 0.50, "FWD": 0.37}
ASSIST_RATIO = 0.85

# Confirmed WC2026 group draw (official, April 2026)
WC2026_GROUPS: dict[str, list[str]] = {
    "A": ["Mexico",        "South Korea",  "South Africa",          "Czech Republic"],
    "B": ["Canada",        "Switzerland",  "Qatar",                 "Bosnia and Herzegovina"],
    "C": ["Brazil",        "Morocco",      "Scotland",              "Haiti"],
    "D": ["United States", "Australia",    "Paraguay",              "Turkey"],
    "E": ["Germany",       "Curaçao",      "Ivory Coast",           "Ecuador"],
    "F": ["Netherlands",   "Japan",        "Tunisia",               "Sweden"],
    "G": ["Belgium",       "Iran",         "Egypt",                 "New Zealand"],
    "H": ["Spain",         "Uruguay",      "Saudi Arabia",          "Cape Verde"],
    "I": ["France",        "Senegal",      "Norway",                "Iraq"],
    "J": ["Argentina",     "Austria",      "Algeria",               "Jordan"],
    "K": ["Portugal",      "Colombia",     "Uzbekistan",            "DR Congo"],
    "L": ["England",       "Croatia",      "Panama",                "Ghana"],
}


def _canonical(name: str) -> str:
    aliases = {
        "united states":                    "usa",
        "south korea":                      "korea republic",
        "iran":                             "ir iran",
        "china":                            "china pr",
        # Cape Verde is stored as "cape verde islands" in StatsBomb / DC model
        "cape verde":                       "cape verde islands",
        # Flatten legacy variants to the DC-model canonical form
        "democratic republic of the congo": "dr congo",
        "bosnia & herzegovina":             "bosnia and herzegovina",
    }
    n = name.lower().strip()
    return aliases.get(n, n)


def _load_players(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("""
        SELECT fp.fantasy_id, fp.name, fp.position, fp.price,
               t.name AS team_name
        FROM   fantasy_players fp
        LEFT JOIN teams t ON fp.team_id = t.team_id
        WHERE  fp.position IS NOT NULL AND fp.price IS NOT NULL AND fp.price > 0
    """).fetchall()
    return [
        {"id": r[0], "name": r[1], "pos": r[2], "price": r[3], "team": r[4] or "Unknown"}
        for r in rows
    ]


def _load_dc() -> dict:
    if not MODEL_PATH.exists():
        return {}
    return json.loads(MODEL_PATH.read_text())


def _build_fixture_lambdas(dc: dict) -> dict[str, list[tuple[float, float]]]:
    """
    For each WC team compute (lambda_for, lambda_against) for their 3 group matches.
    Returns canonical_name -> [(lf, la), (lf, la), (lf, la)].
    """
    team_params = dc.get("team_params", {})
    if not team_params:
        return {}

    mean_atk = float(np.mean([v["attack"]  for v in team_params.values()]))
    mean_def = float(np.mean([v["defense"] for v in team_params.values()]))

    def _p(name: str) -> dict:
        key = _canonical(name)
        return team_params.get(key, {"attack": mean_atk, "defense": mean_def})

    result: dict[str, list[tuple[float, float]]] = {}
    for group_teams in WC2026_GROUPS.values():
        for team in group_teams:
            tk = _canonical(team)
            tp = _p(team)
            result[tk] = [
                (tp["attack"] * _p(opp)["defense"],
                 _p(opp)["attack"] * tp["defense"])
                for opp in group_teams if opp != team
            ]
    return result


def _project_mc(players: list[dict], dc: dict, n_sim: int = 50_000) -> list[dict]:
    """
    Monte Carlo projection using actual WC2026 group fixtures.

    For each team, simulates n_sim independent draws of (goals_for, goals_against)
    per group match using opponent-specific Poisson lambdas from the DC model.
    Fantasy points are accumulated per simulation then averaged.
    """
    rng = np.random.default_rng(42)

    team_params = dc.get("team_params", {})
    if team_params:
        mean_atk = float(np.mean([v["attack"]  for v in team_params.values()]))
        mean_def = float(np.mean([v["defense"] for v in team_params.values()]))
    else:
        mean_atk = mean_def = 1.0

    fixture_lambdas = _build_fixture_lambdas(dc)
    fallback = [(mean_atk * mean_def, mean_atk * mean_def)] * 3

    # Pre-simulate all 48 teams' match goals once — shape (n_sim, 3) per team
    team_sims: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for group_teams in WC2026_GROUPS.values():
        for team in group_teams:
            tk = _canonical(team)
            if tk in team_sims:
                continue
            matches = fixture_lambdas.get(tk, fallback)
            gf = np.column_stack([rng.poisson(lf, n_sim) for lf, _  in matches])
            ga = np.column_stack([rng.poisson(la, n_sim) for _,  la in matches])
            team_sims[tk] = (gf, ga)

    for p in players:
        tk = _canonical(p["team"])
        pos = p["pos"]
        # Price-based star factor: $4.5m → 0.7, $10.5m → 1.4
        sf = 0.7 + (max(45, min(105, p["price"])) - 45) / 60.0 * 0.7

        if tk in team_sims:
            gf_arr, ga_arr = team_sims[tk]   # (n_sim, 3)
        else:
            lam = mean_atk * mean_def
            gf_arr = rng.poisson(lam, (n_sim, 3))
            ga_arr = rng.poisson(lam, (n_sim, 3))

        # --- Per-match events, summed across 3 matches ---

        # Clean-sheet points: 4/4/1/0 per match with zero goals against
        cs_pts = (ga_arr == 0).astype(np.float32).sum(axis=1) * PT_CS.get(pos, 0.0)

        # Concede-2+ penalty: −1 per match where ≥2 goals against (GK/DEF only)
        if pos in ("GK", "DEF"):
            concede_pts = (ga_arr >= 2).astype(np.float32).sum(axis=1) * PT_CONCEDE
        else:
            concede_pts = 0.0

        # Goal contribution: team goals × position share × star factor × pts/goal
        gf_total = gf_arr.sum(axis=1).astype(np.float32)
        pt_g = gf_total * GOAL_SHARE.get(pos, 0.1) * sf * PT_GOAL.get(pos, 4.0)

        # Assist contribution
        pt_a = gf_total * ASSIST_RATIO * ASSIST_SHARE.get(pos, 0.1) * sf * PT_ASSIST

        # Saves (GK only): estimate ~2 saves per goal conceded, 1pt per 3 saves
        if pos == "GK":
            save_pts = (ga_arr.sum(axis=1) * 2.0 / 3.0).astype(np.float32) * PT_SAVE3
        else:
            save_pts = 0.0

        # Appearance: 1pt × 3 matches
        total = 3.0 + pt_g + pt_a + cs_pts + concede_pts + save_pts

        p["projected_pts"] = round(float(total.mean()), 2)
        p["pts_per_match"] = round(float(total.mean()) / 3, 2)

    return players


def optimise(budget: int = BUDGET_DEFAULT) -> dict:
    conn    = sqlite3.connect(DB_PATH)
    players = _load_players(conn)
    conn.close()

    if not players:
        raise RuntimeError("No fantasy players in DB. Run: python wc/scripts/seed_fantasy_players.py")

    dc      = _load_dc()
    players = _project_mc(players, dc)
    n       = len(players)

    pts    = np.array([p["projected_pts"] for p in players], dtype=float)
    prices = np.array([p["price"]         for p in players], dtype=float)
    pos_l  = [p["pos"]  for p in players]
    teams  = [p["team"] for p in players]

    rows, lbs, ubs = [], [], []

    rows.append(np.ones(n)); lbs.append(15.0); ubs.append(15.0)

    for pos, count in SQUAD_RULES.items():
        rows.append(np.array([1.0 if x == pos else 0.0 for x in pos_l]))
        lbs.append(float(count)); ubs.append(float(count))

    rows.append(prices); lbs.append(0.0); ubs.append(float(budget))

    for team in set(teams):
        rows.append(np.array([1.0 if t == team else 0.0 for t in teams]))
        lbs.append(0.0); ubs.append(3.0)

    A           = np.vstack(rows)
    constraints = LinearConstraint(A, lb=np.array(lbs), ub=np.array(ubs))
    result      = milp(-pts, constraints=constraints,
                       integrality=np.ones(n), bounds=Bounds(0.0, 1.0))

    if result.status != 0:
        raise RuntimeError(f"Optimizer failed: {result.message}")

    pos_order = list(SQUAD_RULES.keys())
    squad = sorted(
        [players[i] for i, x in enumerate(result.x) if x > 0.5],
        key=lambda p: (pos_order.index(p["pos"]), -p["projected_pts"]),
    )
    captain = max(squad, key=lambda p: p["projected_pts"])

    return {
        "squad":      squad,
        "total_pts":  round(sum(p["projected_pts"] for p in squad), 1),
        "total_cost": int(sum(p["price"] for p in squad)),
        "captain":    captain,
    }


def captain_picks(top_n: int = 10) -> list[dict]:
    conn    = sqlite3.connect(DB_PATH)
    players = _load_players(conn)
    conn.close()
    if not players:
        return []
    dc      = _load_dc()
    players = _project_mc(players, dc)
    outfield = [p for p in players if p["pos"] != "GK"]
    outfield.sort(key=lambda p: p["projected_pts"], reverse=True)
    return outfield[:top_n]
