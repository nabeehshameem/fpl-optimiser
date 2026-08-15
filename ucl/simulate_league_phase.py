"""
ucl/simulate_league_phase.py  (B5)
Monte-Carlo simulation of the UCL league phase using DC scoreline probabilities.
Runs N simulations of all remaining (not-yet-played) league-phase fixtures and
outputs a qualification probability table — probability each team finishes in
the top 8 (auto-qualify R16), 9-24 (play-off round), 25-36 (eliminated).

Writes predictions/ucl/league_phase_sim.json — served by /api/ucl/bracket once B6 is done.

    python ucl/simulate_league_phase.py
    python ucl/simulate_league_phase.py --n 10000
    python ucl/simulate_league_phase.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import poisson

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dc_match import predict_match  # noqa: E402

DB_PATH = PROJECT_ROOT / "data" / "ucl.db"
MODEL_PATH = PROJECT_ROOT / "models" / "ucl_dc_params.json"
EXPORT_DIR = PROJECT_ROOT / "predictions" / "ucl"

MAX_GOALS = 8
LEAGUE_PHASE_ROUND = "League Phase"
TOP8_SPOTS = 8
PLAYOFF_SPOTS = 16  # positions 9-24 go to play-off round
DEFAULT_SIMS = 5000


def _load_dc() -> dict:
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"UCL DC params not found: {MODEL_PATH}\n"
            "Run: python ucl/train_dc.py"
        )
    return json.loads(MODEL_PATH.read_text(encoding="utf-8"))


def _load_fixtures_and_results() -> tuple[list[dict], list[dict]]:
    """Return (played, remaining) league-phase fixtures."""
    if not DB_PATH.exists():
        raise RuntimeError(f"UCL database not found: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    try:
        rows = conn.execute("""
            SELECT
                f.fixture_id,
                f.status,
                f.home_score, f.away_score,
                th.short_name AS home_sn,
                ta.short_name AS away_sn
            FROM fixtures f
            JOIN teams th ON f.home_team_id = th.team_id
            JOIN teams ta ON f.away_team_id = ta.team_id
            WHERE f.round_name = ?
            ORDER BY f.kickoff_utc
        """, (LEAGUE_PHASE_ROUND,)).fetchall()
    finally:
        conn.close()

    played, remaining = [], []
    for fid, status, hs, as_, h_sn, a_sn in rows:
        fix = {"home": h_sn, "away": a_sn, "home_score": hs, "away_score": as_}
        if status in ("FT", "AET", "PEN") and hs is not None and as_ is not None:
            played.append(fix)
        else:
            remaining.append(fix)
    return played, remaining


def _teams_from_fixtures(played: list[dict], remaining: list[dict]) -> set[str]:
    teams: set[str] = set()
    for f in played + remaining:
        teams.add(f["home"])
        teams.add(f["away"])
    return teams


def _initial_table(played: list[dict]) -> dict[str, dict]:
    """Build a points table from already-played fixtures."""
    table: dict[str, dict] = {}

    def _entry(sn: str) -> dict:
        if sn not in table:
            table[sn] = {"played": 0, "won": 0, "drawn": 0, "lost": 0,
                         "gf": 0, "ga": 0, "gd": 0, "pts": 0}
        return table[sn]

    for f in played:
        h, a = f["home"], f["away"]
        hg, ag = int(f["home_score"]), int(f["away_score"])
        _entry(h)["played"] += 1
        _entry(a)["played"] += 1
        table[h]["gf"] += hg; table[h]["ga"] += ag
        table[a]["gf"] += ag; table[a]["ga"] += hg
        table[h]["gd"] += hg - ag; table[a]["gd"] += ag - hg
        if hg > ag:
            table[h]["won"] += 1; table[h]["pts"] += 3
            table[a]["lost"] += 1
        elif ag > hg:
            table[a]["won"] += 1; table[a]["pts"] += 3
            table[h]["lost"] += 1
        else:
            table[h]["drawn"] += 1; table[h]["pts"] += 1
            table[a]["drawn"] += 1; table[a]["pts"] += 1

    return table


def _simulate_scoreline(h_sn: str, a_sn: str, dc: dict) -> tuple[int, int]:
    """Sample one scoreline from the DC distribution."""
    tp = dc.get("team_params", {})
    home_adv = float(dc.get("home_adv", 1.10))
    rho = float(dc.get("rho", -0.10))

    if tp:
        mean_atk = sum(v["attack"] for v in tp.values()) / len(tp)
        mean_def = sum(v["defense"] for v in tp.values()) / len(tp)
    else:
        mean_atk = mean_def = 1.0

    h = tp.get(h_sn, {"attack": mean_atk, "defense": mean_def})
    a = tp.get(a_sn, {"attack": mean_atk, "defense": mean_def})
    form_adj = dc.get("form_adjustments", {})
    hf = form_adj.get(h_sn, [1.0, 1.0])
    af = form_adj.get(a_sn, [1.0, 1.0])

    mu_h = min(h["attack"] * hf[0] * a["defense"] * af[1] * home_adv, 6.0)
    mu_a = min(a["attack"] * af[0] * h["defense"] * hf[1], 6.0)

    goals = np.arange(MAX_GOALS + 1)
    mat = np.outer(poisson.pmf(goals, mu_h), poisson.pmf(goals, mu_a))
    # DC correction for low-score cells
    for x in range(2):
        for y in range(2):
            tau = 1.0
            if x == 0 and y == 0:
                tau = max(1.0 - mu_h * mu_a * rho, 0.0)
            elif x == 1 and y == 0:
                tau = max(1.0 + mu_a * rho, 0.0)
            elif x == 0 and y == 1:
                tau = max(1.0 + mu_h * rho, 0.0)
            elif x == 1 and y == 1:
                tau = max(1.0 - rho, 0.0)
            mat[x, y] *= tau
    mat /= mat.sum()

    flat_probs = mat.ravel()
    idx = np.searchsorted(np.cumsum(flat_probs), random.random())
    idx = min(idx, len(flat_probs) - 1)
    h_goals = int(idx // (MAX_GOALS + 1))
    a_goals = int(idx % (MAX_GOALS + 1))
    return h_goals, a_goals


def _final_rank(table: dict[str, dict]) -> list[str]:
    """Sort 36 teams by UCL tiebreak: pts → gd → gf."""
    return sorted(
        table.keys(),
        key=lambda sn: (table[sn]["pts"], table[sn]["gd"], table[sn]["gf"]),
        reverse=True,
    )


def run_simulation(n_sims: int, dc: dict) -> dict:
    played, remaining = _load_fixtures_and_results()
    all_teams = sorted(_teams_from_fixtures(played, remaining))
    n_teams = len(all_teams)

    if not all_teams:
        raise RuntimeError(
            "No league-phase fixtures found in data/ucl.db.\n"
            "Run: python ucl/ingest_fixtures.py"
        )

    # Counts: how many times each team finishes in each rank bucket
    top8_count: dict[str, int] = defaultdict(int)
    playoff_count: dict[str, int] = defaultdict(int)
    elim_count: dict[str, int] = defaultdict(int)
    pts_sum: dict[str, float] = defaultdict(float)

    base_table = _initial_table(played)
    # Seed any team not yet in base_table
    for sn in all_teams:
        if sn not in base_table:
            base_table[sn] = {"played": 0, "won": 0, "drawn": 0, "lost": 0,
                               "gf": 0, "ga": 0, "gd": 0, "pts": 0}

    for _ in range(n_sims):
        sim_table = {sn: dict(row) for sn, row in base_table.items()}

        for f in remaining:
            h, a = f["home"], f["away"]
            hg, ag = _simulate_scoreline(h, a, dc)
            sim_table[h]["gf"] += hg; sim_table[h]["ga"] += ag
            sim_table[a]["gf"] += ag; sim_table[a]["ga"] += hg
            sim_table[h]["gd"] += hg - ag; sim_table[a]["gd"] += ag - hg
            if hg > ag:
                sim_table[h]["won"] += 1; sim_table[h]["pts"] += 3
                sim_table[a]["lost"] += 1
            elif ag > hg:
                sim_table[a]["won"] += 1; sim_table[a]["pts"] += 3
                sim_table[h]["lost"] += 1
            else:
                sim_table[h]["drawn"] += 1; sim_table[h]["pts"] += 1
                sim_table[a]["drawn"] += 1; sim_table[a]["pts"] += 1

        ranked = _final_rank(sim_table)
        for rank, sn in enumerate(ranked, 1):
            pts_sum[sn] += sim_table[sn]["pts"]
            if rank <= TOP8_SPOTS:
                top8_count[sn] += 1
            elif rank <= TOP8_SPOTS + PLAYOFF_SPOTS:
                playoff_count[sn] += 1
            else:
                elim_count[sn] += 1

    # Current table for display
    current_rank = _final_rank(base_table)
    rows = []
    for rank, sn in enumerate(current_rank, 1):
        row = base_table[sn]
        rows.append({
            "rank": rank,
            "team": sn,
            "played": row["played"],
            "points": row["pts"],
            "gd": row["gd"],
            "goals_for": row["gf"],
            "p_top8_pct": round(top8_count[sn] / n_sims * 100, 1),
            "p_playoff_pct": round(playoff_count[sn] / n_sims * 100, 1),
            "p_eliminated_pct": round(elim_count[sn] / n_sims * 100, 1),
            "projected_pts": round(pts_sum[sn] / n_sims, 1),
        })

    return {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_simulations": n_sims,
        "played_fixtures": len(played),
        "remaining_fixtures": len(remaining),
        "table": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=DEFAULT_SIMS,
                    help=f"Number of simulations (default: {DEFAULT_SIMS})")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    dc = _load_dc()
    print(f"Running {args.n:,} simulations of the UCL league phase…")
    result = run_simulation(args.n, dc)

    print(f"\nLeague phase simulation ({result['played_fixtures']} played, "
          f"{result['remaining_fixtures']} remaining):")
    print(f"{'Rk':<3} {'Team':<6} {'Pts':<5} {'GD':<5} "
          f"{'Top8%':<8} {'Playoff%':<10} {'Elim%':<8}")
    for row in result["table"]:
        print(f"{row['rank']:<3} {row['team']:<6} {row['points']:<5} "
              f"{row['gd']:<5} {row['p_top8_pct']:<8} "
              f"{row['p_playoff_pct']:<10} {row['p_eliminated_pct']:<8}")

    if args.dry_run:
        return

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = EXPORT_DIR / "league_phase_sim.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"\nSimulation results written → {out}")


if __name__ == "__main__":
    main()
