"""
ucl/simulate_bracket.py  (B6)
Monte-Carlo simulation of the UCL knockout bracket (play-off round through Final).
Requires the league phase to have concluded so we have the actual seedings.

Bracket structure (2024/25+ format):
  - Teams 1-8:  directly to R16
  - Teams 9-24: play-off round (2-leg); winners go to R16
  - Play-off draw: team 9-16 hosts team 17-24 (unseeded vs seeded legs)
  - R16 through SF: two-legged; Final: single-match neutral venue

Input: either the committed league-phase standings (from data/ucl.db) or a
manually supplied seeding file (for testing with hypothetical standings).

    python ucl/simulate_bracket.py                     # use DB standings
    python ucl/simulate_bracket.py --n 10000
    python ucl/simulate_bracket.py --standings-json path/to/standings.json
    python ucl/simulate_bracket.py --dry-run

Writes predictions/ucl/bracket.json; /api/ucl/bracket serves this file.
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
DEFAULT_SIMS = 5000
NEUTRAL_VENUE_HOME_ADV = 1.0  # no home advantage in the Final


def _load_dc() -> dict:
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"UCL DC params not found: {MODEL_PATH}\n"
            "Run: python ucl/train_dc.py"
        )
    return json.loads(MODEL_PATH.read_text(encoding="utf-8"))


def _load_league_standings() -> list[str]:
    """Return short_names ordered 1-36 by final league phase standings."""
    if not DB_PATH.exists():
        raise RuntimeError(f"UCL database not found: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    try:
        rows = conn.execute("""
            SELECT t.short_name, ls.points,
                   ls.goals_for - ls.goals_against AS gd, ls.goals_for
            FROM league_standings ls
            JOIN teams t ON t.team_id = ls.team_id
            ORDER BY ls.points DESC, gd DESC, ls.goals_for DESC
        """).fetchall()
    finally:
        conn.close()
    return [r[0] for r in rows]


def _simulate_leg(h_sn: str, a_sn: str, dc: dict,
                  home_adv: float | None = None) -> tuple[int, int]:
    """Sample one match from DC distribution. home_adv=1.0 for neutral venues."""
    tp = dc.get("team_params", {})
    if home_adv is None:
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
    return int(idx // (MAX_GOALS + 1)), int(idx % (MAX_GOALS + 1))


def _two_leg_winner(home: str, away: str, dc: dict) -> str:
    """Determine aggregate winner of a two-legged tie (home team plays first leg at home)."""
    # Leg 1: home hosts away
    h1, a1 = _simulate_leg(home, away, dc)
    # Leg 2: away hosts home
    h2, a2 = _simulate_leg(away, home, dc)
    # Aggregate: from home team's perspective
    home_agg = h1 + a2
    away_agg = a1 + h2
    if home_agg > away_agg:
        return home
    if away_agg > home_agg:
        return away
    # Away goals rule: if still tied, random (in practice away-goals abolished, but coin-toss is correct)
    return random.choice([home, away])


def _single_match_winner(sn1: str, sn2: str, dc: dict,
                         home_adv: float = NEUTRAL_VENUE_HOME_ADV) -> str:
    g1, g2 = _simulate_leg(sn1, sn2, dc, home_adv=home_adv)
    if g1 > g2:
        return sn1
    if g2 > g1:
        return sn2
    return random.choice([sn1, sn2])


def _simulate_bracket(seedings: list[str], dc: dict) -> str:
    """Run one bracket from play-off through Final. Returns winner short_name.

    seedings: list of 36 teams ordered 1-36 (index 0 = rank 1).
    """
    # Teams 1-8 (index 0-7): direct R16 seeds
    r16_seeds = seedings[:8]
    # Teams 9-24 (index 8-23): play-off round
    # UCL draw: seed (9-16) vs non-seed (17-24) — pair by rank proximity
    playoff_seeded = seedings[8:16]   # ranks 9-16
    playoff_unseeded = seedings[16:24]  # ranks 17-24

    # Play-off: 8 ties, seeded team hosts second leg
    playoff_winners = []
    for i in range(8):
        # unseeded plays first leg at home (seeded get 2nd-leg home advantage)
        winner = _two_leg_winner(
            home=playoff_unseeded[i],
            away=playoff_seeded[i],
            dc=dc,
        )
        playoff_winners.append(winner)

    # R16: 16 teams (8 seeds + 8 playoff winners)
    # Real draw is complex; we randomise pairings for simulation purposes
    r16_field = r16_seeds + playoff_winners
    random.shuffle(r16_field)
    r16_pairs = [(r16_field[i * 2], r16_field[i * 2 + 1]) for i in range(8)]
    qf_field = [_two_leg_winner(h, a, dc) for h, a in r16_pairs]

    # QF: 4 ties, 2-leg
    random.shuffle(qf_field)
    qf_pairs = [(qf_field[i * 2], qf_field[i * 2 + 1]) for i in range(4)]
    sf_field = [_two_leg_winner(h, a, dc) for h, a in qf_pairs]

    # SF: 2 ties, 2-leg
    random.shuffle(sf_field)
    sf_pairs = [(sf_field[0], sf_field[1]), (sf_field[2], sf_field[3])]
    finalists = [_two_leg_winner(h, a, dc) for h, a in sf_pairs]

    # Final: single match, neutral venue
    return _single_match_winner(finalists[0], finalists[1], dc,
                                home_adv=NEUTRAL_VENUE_HOME_ADV)


def run_simulation(n_sims: int, seedings: list[str], dc: dict) -> dict:
    if len(seedings) < 24:
        raise ValueError(
            f"Need at least 24 seedings for a bracket simulation; got {len(seedings)}."
        )

    win_count: dict[str, int] = defaultdict(int)
    final_count: dict[str, int] = defaultdict(int)
    sf_count: dict[str, int] = defaultdict(int)

    for _ in range(n_sims):
        # Randomise ties within each seeding tier for each simulation
        # (real UCL draw has restrictions we simplify away)
        seedings_copy = list(seedings)
        winner = _simulate_bracket(seedings_copy, dc)
        win_count[winner] += 1

    # Full bracket win probabilities
    results = []
    for rank, sn in enumerate(seedings, 1):
        results.append({
            "rank": rank,
            "team": sn,
            "p_win_pct": round(win_count.get(sn, 0) / n_sims * 100, 1),
        })

    return {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_simulations": n_sims,
        "seedings": seedings[:24],
        "teams": results,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=DEFAULT_SIMS)
    ap.add_argument("--standings-json", type=Path, default=None,
                    help="Path to a JSON file with {'standings': [{'team': 'MCI', ...}, ...]}")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    dc = _load_dc()

    if args.standings_json:
        data = json.loads(args.standings_json.read_text())
        seedings = [r["team"] for r in data["standings"]]
    else:
        seedings = _load_league_standings()

    if not seedings:
        print("No standings data. Run ucl/ingest_fixtures.py to populate standings.")
        sys.exit(1)

    print(f"Running {args.n:,} bracket simulations…")
    print(f"Seedings 1-8 (auto-qualify R16): {', '.join(seedings[:8])}")
    if len(seedings) >= 24:
        print(f"Seedings 9-24 (play-off round): {', '.join(seedings[8:24])}")

    result = run_simulation(args.n, seedings, dc)

    print(f"\n{'Rk':<3} {'Team':<6} {'Win%':<8}")
    for row in sorted(result["teams"], key=lambda r: r["p_win_pct"], reverse=True)[:16]:
        print(f"{row['rank']:<3} {row['team']:<6} {row['p_win_pct']:<8}")

    if args.dry_run:
        return

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = EXPORT_DIR / "bracket.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"\nBracket simulation written → {out}")


if __name__ == "__main__":
    main()
