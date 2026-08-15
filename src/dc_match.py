"""
dc_match.py

Pure Dixon-Coles scoreline matrix computation. Takes short_name strings —
never integer IDs. Rule 3: season/tournament boundaries join on names, not IDs.

Shared across:
  - scripts/build_gw_tools.py  (PL GW predictions)
  - ucl/                       (UCL matchday predictions, Phase B1)
"""
from __future__ import annotations

import numpy as np
from scipy.stats import poisson

MAX_GOALS = 8


def _tau(x: int, y: int, mu_h: float, mu_a: float, rho: float) -> float:
    if x == 0 and y == 0:
        return 1.0 - mu_h * mu_a * rho
    if x == 1 and y == 0:
        return 1.0 + mu_a * rho
    if x == 0 and y == 1:
        return 1.0 + mu_h * rho
    if x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


def predict_match(
    home_sn: str,
    away_sn: str,
    dc: dict,
    max_goals: int = MAX_GOALS,
) -> dict:
    """
    Scoreline distribution for home_sn vs away_sn from DC params.

    dc: the dict from models/fpl_dc_params.json (short_name-keyed team_params).

    Returns a fully serialisable dict — no numpy arrays.
    The caller must convert live_team_id -> short_name before calling here.
    """
    tp = dc.get("team_params", {})
    form_adj = dc.get("form_adjustments", {})
    home_adv = float(dc.get("home_adv", 1.20))
    rho = float(dc.get("rho", -0.10))

    if tp:
        mean_atk = sum(v["attack"] for v in tp.values()) / len(tp)
        mean_def = sum(v["defense"] for v in tp.values()) / len(tp)
    else:
        mean_atk = mean_def = 1.0

    h = tp.get(home_sn, {"attack": mean_atk, "defense": mean_def})
    a = tp.get(away_sn, {"attack": mean_atk, "defense": mean_def})
    h_form = form_adj.get(home_sn, [1.0, 1.0])
    a_form = form_adj.get(away_sn, [1.0, 1.0])

    mu_h = h["attack"] * h_form[0] * a["defense"] * a_form[1] * home_adv
    mu_a = a["attack"] * a_form[0] * h["defense"] * h_form[1]
    mu_h = min(mu_h, 6.0)
    mu_a = min(mu_a, 6.0)

    goals = np.arange(max_goals + 1)
    mat = np.outer(poisson.pmf(goals, mu_h), poisson.pmf(goals, mu_a))
    for x in range(2):
        for y in range(2):
            mat[x, y] *= max(_tau(x, y, mu_h, mu_a, rho), 0.0)
    mat /= mat.sum()

    p_home = float(np.tril(mat, -1).sum()) * 100
    p_draw = float(np.diag(mat).sum()) * 100
    p_away = float(np.triu(mat, 1).sum()) * 100

    flat = sorted(
        [(i, j, float(mat[i, j]) * 100)
         for i in range(max_goals + 1)
         for j in range(max_goals + 1)],
        key=lambda t: t[2], reverse=True,
    )
    top_h, top_a, top_pct = flat[0]

    return {
        "home_sn": home_sn,
        "away_sn": away_sn,
        "xg_home": round(mu_h, 2),
        "xg_away": round(mu_a, 2),
        "p_home": round(p_home, 1),
        "p_draw": round(p_draw, 1),
        "p_away": round(p_away, 1),
        "home_cs_pct": round(float(mat[:, 0].sum()) * 100, 1),
        "away_cs_pct": round(float(mat[0, :].sum()) * 100, 1),
        "top_scoreline": f"{top_h}-{top_a}",
        "top_scoreline_pct": round(top_pct, 1),
    }
