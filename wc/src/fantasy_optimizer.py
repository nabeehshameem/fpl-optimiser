"""
fantasy_optimizer.py
Projects fantasy points per player and optimises squad selection.

Point projection uses DC team attack/defense ratings from the trained model.
Squad optimization uses scipy.optimize.milp (binary ILP).

Assumed WC Fantasy rules:
  Squad:  2 GK + 5 DEF + 5 MID + 3 FWD = 15 players
  Budget: £100m (1000 in £0.1m units)
  Cap:    max 3 players per team
  Projection base: 3 group-stage matches
"""

import json
import sqlite3
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH      = PROJECT_ROOT / "data" / "wc.db"
MODEL_PATH   = PROJECT_ROOT / "models" / "dc_params.json"

SQUAD_RULES    = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
BUDGET_DEFAULT = 1000  # £100.0m

# Points per action
PT_PLAYED  = 1.0
PT_CS      = {"GK": 4.0, "DEF": 4.0, "MID": 1.0, "FWD": 0.0}
PT_GOAL    = {"GK": 6.0, "DEF": 6.0, "MID": 5.0, "FWD": 4.0}
PT_ASSIST  = 3.0
PT_SAVE3   = 1.0   # per 3 saves (GK only)
PT_CONCEDE = -1.0  # if 2+ goals conceded (GK/DEF)

# Position shares of team goals / assists
GOAL_SHARE   = {"GK": 0.01, "DEF": 0.07, "MID": 0.30, "FWD": 0.62}
ASSIST_SHARE = {"GK": 0.01, "DEF": 0.12, "MID": 0.50, "FWD": 0.37}
ASSIST_RATIO = 0.85  # ~85 % of goals accompanied by an assist


def _canonical(name: str) -> str:
    aliases = {
        "united states": "usa", "south korea": "korea republic",
        "iran": "ir iran", "china": "china pr",
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


def _project(players: list[dict], dc: dict, matches: int = 3) -> list[dict]:
    team_params = dc.get("team_params", {})
    if team_params:
        mean_atk = float(np.mean([v["attack"]  for v in team_params.values()]))
        mean_def = float(np.mean([v["defense"] for v in team_params.values()]))
    else:
        mean_atk = mean_def = 1.0

    # Team-level metrics keyed by canonical name
    metrics: dict[str, dict] = {}
    for p in players:
        key = _canonical(p["team"])
        if key in metrics:
            continue
        tp   = team_params.get(key, {"attack": mean_atk, "defense": mean_def})
        xg_f = tp["attack"]  * mean_def
        xg_a = mean_atk      * tp["defense"]
        metrics[key] = {
            "xg_for":     xg_f,
            "xg_against": xg_a,
            "cs_prob":    float(np.exp(-xg_a)),
        }

    for p in players:
        m   = metrics.get(_canonical(p["team"]), {"xg_for": 1.0, "xg_against": 1.0, "cs_prob": 0.35})
        pos = p["pos"]
        # Price-based star factor: £4.5m → 0.7, £12.0m → 1.4
        sf  = 0.7 + (max(45, min(120, p["price"])) - 45) / 75.0 * 0.7

        exp_goals   = m["xg_for"] * GOAL_SHARE.get(pos, 0.1)   * sf
        exp_assists = m["xg_for"] * ASSIST_RATIO * ASSIST_SHARE.get(pos, 0.1) * sf

        pt_cs = m["cs_prob"] * PT_CS.get(pos, 0.0)
        pt_g  = exp_goals   * PT_GOAL.get(pos, 4.0)
        pt_a  = exp_assists * PT_ASSIST

        pt_concede = 0.0
        if pos in ("GK", "DEF"):
            xga = m["xg_against"]
            p2plus = 1.0 - np.exp(-xga) - xga * np.exp(-xga)
            pt_concede = float(p2plus) * PT_CONCEDE

        pt_saves = 0.0
        if pos == "GK":
            pt_saves = (m["xg_against"] * 2.0 / 3.0) * PT_SAVE3

        per_match = PT_PLAYED + pt_g + pt_a + pt_cs + pt_concede + pt_saves
        p["projected_pts"]  = round(per_match * matches, 2)
        p["pts_per_match"]  = round(per_match, 2)

    return players


def optimise(budget: int = BUDGET_DEFAULT) -> dict:
    conn    = sqlite3.connect(DB_PATH)
    players = _load_players(conn)
    conn.close()

    if not players:
        raise RuntimeError("No fantasy players in DB. Run: python wc/scripts/seed_fantasy_players.py")

    dc      = _load_dc()
    players = _project(players, dc)
    n       = len(players)

    pts    = np.array([p["projected_pts"] for p in players], dtype=float)
    prices = np.array([p["price"]         for p in players], dtype=float)
    pos_l  = [p["pos"]  for p in players]
    teams  = [p["team"] for p in players]

    rows, lbs, ubs = [], [], []

    # Total squad = 15
    rows.append(np.ones(n)); lbs.append(15.0); ubs.append(15.0)

    # Position counts
    for pos, count in SQUAD_RULES.items():
        rows.append(np.array([1.0 if x == pos else 0.0 for x in pos_l]))
        lbs.append(float(count)); ubs.append(float(count))

    # Budget
    rows.append(prices); lbs.append(0.0); ubs.append(float(budget))

    # Max 3 per team
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
        "squad":         squad,
        "total_pts":     round(sum(p["projected_pts"] for p in squad), 1),
        "total_cost":    int(sum(p["price"] for p in squad)),
        "captain":       captain,
    }


def captain_picks(top_n: int = 10) -> list[dict]:
    conn    = sqlite3.connect(DB_PATH)
    players = _load_players(conn)
    conn.close()
    if not players:
        return []
    dc      = _load_dc()
    players = _project(players, dc)
    outfield = [p for p in players if p["pos"] != "GK"]
    outfield.sort(key=lambda p: p["projected_pts"], reverse=True)
    return outfield[:top_n]
