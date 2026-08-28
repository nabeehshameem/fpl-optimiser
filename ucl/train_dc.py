"""
ucl/train_dc.py
Fit a Dixon-Coles Poisson model on UCL historical results from data/ucl.db
and save parameters to models/ucl_dc_params.json.

Differences from FPL model (scripts/train_dc.py):
  - Much smaller sample (~125 league-phase + knockout fixtures); regularisation
    is tightened (REG_LAMBDA = 10 vs 5) to prevent overfitting.
  - xG is used as the response variable where API-Football provides it;
    falls back to actual goals where xG is missing.
  - Home advantage is meaningful only for two-legged ties and early league-
    phase fixtures; apply cautiously — the optimizer is free to shrink it.
  - No FDR or form adjustments by default for the first season (too few matches
    per team to compute a reliable recent-form window).

    python ucl/train_dc.py                # current season
    python ucl/train_dc.py --season 2026  # explicit
    python ucl/train_dc.py --use-actual-goals
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DB_PATH = PROJECT_ROOT / "data" / "ucl.db"
MODEL_PATH = PROJECT_ROOT / "models" / "ucl_dc_params.json"

MAX_GOALS = 8
REG_LAMBDA = 10.0  # stronger regularisation than FPL (smaller dataset)


def _current_ucl_season() -> int:
    now = datetime.now(timezone.utc)
    return now.year if now.month >= 7 else now.year - 1


def _recency_weight(kickoff: str | None, decay: float = 0.8) -> float:
    if not kickoff:
        return 0.5
    try:
        d = date.fromisoformat(str(kickoff)[:10])
    except ValueError:
        return 0.5
    days_ago = (date.today() - d).days
    return float(np.exp(-decay * days_ago / 365.0))


def _dc_tau(x: int, y: int, mu_h: float, mu_a: float, rho: float) -> float:
    if x == 0 and y == 0:
        return 1.0 - mu_h * mu_a * rho
    if x == 1 and y == 0:
        return 1.0 + mu_a * rho
    if x == 0 and y == 1:
        return 1.0 + mu_h * rho
    if x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


def load_matches(season: int, use_xg: bool = True) -> list[dict]:
    """Load finished UCL matches with goal/xG data from ucl.db."""
    if not DB_PATH.exists():
        raise RuntimeError(
            f"UCL database not found: {DB_PATH}\n"
            "Run: python ucl/init_db.py && python ucl/ingest_fixtures.py"
        )
    conn = sqlite3.connect(DB_PATH)
    try:
        rows = conn.execute("""
            SELECT
                f.fixture_id,
                f.home_team_id, f.away_team_id,
                f.home_score, f.away_score,
                f.kickoff_utc,
                s.xg_home, s.xg_away,
                th.short_name AS home_sn,
                ta.short_name AS away_sn
            FROM fixtures f
            JOIN teams th ON f.home_team_id = th.team_id
            JOIN teams ta ON f.away_team_id = ta.team_id
            LEFT JOIN match_stats s ON s.fixture_id = f.fixture_id
            WHERE f.status = 'FT'
              AND f.home_score IS NOT NULL AND f.away_score IS NOT NULL
            ORDER BY f.kickoff_utc
        """).fetchall()
    finally:
        conn.close()

    matches = []
    for fid, hid, aid, hs, as_, ko, xg_h, xg_a, h_sn, a_sn in rows:
        if use_xg and xg_h is not None and xg_a is not None:
            hg = float(xg_h)
            ag = float(xg_a)
        else:
            hg = float(hs)
            ag = float(as_)
        matches.append({
            "home_sn": h_sn,
            "away_sn": a_sn,
            "home_goals": hg,
            "away_goals": ag,
            "kickoff": str(ko or ""),
            "weight": _recency_weight(ko),
        })
    return matches


def fit(matches: list[dict]) -> dict:
    """Fit DC model and return params dict ready for JSON serialisation."""
    if not matches:
        raise RuntimeError(
            "No finished UCL fixtures in DB. "
            "Run ucl/ingest_results.py to populate match_stats."
        )

    all_teams = sorted({m["home_sn"] for m in matches} | {m["away_sn"] for m in matches})
    idx = {sn: i for i, sn in enumerate(all_teams)}
    n = len(all_teams)

    home_idx = np.array([idx[m["home_sn"]] for m in matches], dtype=np.int32)
    away_idx = np.array([idx[m["away_sn"]] for m in matches], dtype=np.int32)
    hg = np.array([m["home_goals"] for m in matches], dtype=np.float64)
    ag = np.array([m["away_goals"] for m in matches], dtype=np.float64)
    weights = np.array([m["weight"] for m in matches], dtype=np.float64)

    from scipy.special import gammaln
    log_fact_h = gammaln(hg + 1)
    log_fact_a = gammaln(ag + 1)

    m00 = (hg < 0.5) & (ag < 0.5)
    m10 = (hg < 1.5) & (hg >= 0.5) & (ag < 0.5)
    m01 = (hg < 0.5) & (ag < 1.5) & (ag >= 0.5)
    m11 = (hg < 1.5) & (hg >= 0.5) & (ag < 1.5) & (ag >= 0.5)

    def neg_ll(params: np.ndarray) -> float:
        atk = np.exp(params[:n])
        dfn = np.exp(params[n:2 * n])
        ha = np.exp(params[2 * n])
        rho = params[2 * n + 1]

        mu_h = atk[home_idx] * dfn[away_idx] * ha
        mu_a = atk[away_idx] * dfn[home_idx]

        log_p = (hg * np.log(np.maximum(mu_h, 1e-12)) - mu_h - log_fact_h
               + ag * np.log(np.maximum(mu_a, 1e-12)) - mu_a - log_fact_a)

        tau = np.ones(len(hg))
        tau[m00] = np.maximum(1.0 - mu_h[m00] * mu_a[m00] * rho, 1e-9)
        tau[m10] = np.maximum(1.0 + mu_a[m10] * rho, 1e-9)
        tau[m01] = np.maximum(1.0 + mu_h[m01] * rho, 1e-9)
        tau[m11] = np.maximum(1.0 - rho, 1e-9)

        ll = float(np.dot(weights, np.log(tau) + log_p))
        reg = REG_LAMBDA * float(np.sum(params[1:n] ** 2) + np.sum(params[n:2 * n] ** 2))
        return -ll + reg

    x0 = np.zeros(2 * n + 2)
    x0[2 * n] = np.log(1.10)   # UCL home adv lower than PL (~10%)
    x0[2 * n + 1] = -0.10

    bounds = (
        [(0.0, 0.0)]
        + [(-3.0, 3.0)] * (n - 1)
        + [(-3.0, 3.0)] * n
        + [(np.log(0.9), np.log(1.4))]
        + [(-0.5, 0.2)]
    )

    result = minimize(
        neg_ll, x0, method="L-BFGS-B", bounds=bounds,
        options={"maxiter": 1000, "maxfun": 200000, "ftol": 1e-8, "gtol": 1e-4},
    )

    opt = result.x
    log_atk = opt[:n]
    log_def = opt[n:2 * n]
    home_adv = float(np.exp(opt[2 * n]))
    rho = float(opt[2 * n + 1])

    team_params = {
        sn: {
            "attack": float(np.exp(log_atk[i])),
            "defense": float(np.exp(log_def[i])),
        }
        for i, sn in enumerate(all_teams)
    }

    return {
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_teams": n,
        "n_fixtures": len(matches),
        "home_adv": round(home_adv, 4),
        "rho": round(rho, 4),
        "converged": bool(result.success),
        "message": result.message,
        "team_params": team_params,
        # form_adjustments reserved for B3 — computed from recent match stats
        "form_adjustments": {},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--use-actual-goals", action="store_true",
                    help="Use raw goals instead of xG even when xG is available.")
    args = ap.parse_args()

    season = args.season or _current_ucl_season()
    use_xg = not args.use_actual_goals
    print(f"=== UCL DC training — season {season}/{season % 100 + 1} "
          f"({'xG' if use_xg else 'actual goals'}) ===")

    matches = load_matches(season, use_xg=use_xg)
    if not matches:
        print("No finished matches found. Run ucl/ingest_results.py first.")
        sys.exit(1)

    print(f"Fitting on {len(matches)} finished fixtures…")
    params = fit(matches)

    print(f"\nFit complete:")
    print(f"  Teams:     {params['n_teams']}")
    print(f"  Fixtures:  {params['n_fixtures']}")
    print(f"  Home adv:  {params['home_adv']:.4f}  ({(params['home_adv']-1)*100:.1f}%)")
    print(f"  Rho:       {params['rho']:.4f}")
    print(f"  Converged: {params['converged']} — {params['message']}")

    print("\nTop 10 teams by attack strength:")
    ranked = sorted(params["team_params"].items(),
                    key=lambda kv: kv[1]["attack"], reverse=True)
    for sn, p in ranked[:10]:
        print(f"  {sn:<6} atk={p['attack']:.3f}  def={p['defense']:.3f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.write_text(json.dumps(params, indent=2))
    print(f"\nUCL DC params saved -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
