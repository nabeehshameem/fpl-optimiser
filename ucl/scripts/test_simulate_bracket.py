"""
Synthetic tests for ucl/simulate_bracket.py.
No UCL database required — seedings are passed directly.
Run: python ucl/test_simulate_bracket.py

B1  Output structure contract (required keys present)
B2  Win percentages sum to 100% across all teams
B3  Highest-attack team wins more often than lowest-attack team
B4  ValueError raised when fewer than 24 seedings provided
B5  run_simulation produces exactly one winner per simulation internally
    (verified by: win_count total == n_sims)
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ucl.simulate_bracket import run_simulation  # noqa: E402


def check(label, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {label}" + (f"  ({detail})" if detail else ""))
    return cond


N_TEAMS = 36
N_SIMS = 500


def _make_seedings(strong_at_rank1: bool = True) -> list[str]:
    return [f"T{i:02d}" for i in range(1, N_TEAMS + 1)]


def _make_dc(strong: str = "T01", weak: str = "T36") -> dict:
    tp = {}
    for i in range(1, N_TEAMS + 1):
        sn = f"T{i:02d}"
        # Linear gradient: rank 1 is strongest, rank 36 weakest
        strength = 1.0 + (N_TEAMS - i) / (N_TEAMS - 1) * 2.0
        tp[sn] = {"attack": strength, "defense": 1.0 / strength}
    return {
        "home_adv": 1.10,
        "rho": -0.10,
        "team_params": tp,
        "form_adjustments": {},
    }


def main():
    import random
    random.seed(42)
    ok = True

    seedings = _make_seedings()
    dc = _make_dc()

    result = run_simulation(n_sims=N_SIMS, seedings=seedings, dc=dc)

    # B1: required top-level keys
    required = {"generated_at_utc", "n_simulations", "seedings", "teams"}
    missing = required - set(result)
    ok &= check("B1 output structure contract", not missing, str(missing))

    teams = result.get("teams", [])

    # B2: win % sum to 100%
    total_pct = sum(t["p_win_pct"] for t in teams)
    ok &= check("B2 win percentages sum to 100%",
                abs(total_pct - 100.0) <= 1.5, f"sum={total_pct:.1f}")

    # B3: T01 (strongest) wins more often than T36 (weakest)
    by_team = {t["team"]: t["p_win_pct"] for t in teams}
    t01_pct = by_team.get("T01", 0)
    t36_pct = by_team.get("T36", 0)
    ok &= check("B3 strongest team wins more often than weakest",
                t01_pct > t36_pct, f"T01={t01_pct:.1f}% T36={t36_pct:.1f}%")

    # B4: ValueError on < 24 seedings
    try:
        run_simulation(n_sims=10, seedings=seedings[:20], dc=dc)
        ok &= check("B4 ValueError on < 24 seedings", False)
    except ValueError:
        ok &= check("B4 ValueError on < 24 seedings", True)

    # B5: exactly one winner per simulation (total wins == N_SIMS)
    total_wins = sum(
        round(t["p_win_pct"] * N_SIMS / 100) for t in teams
    )
    # Allow ±2% rounding tolerance
    ok &= check("B5 total wins == n_sims (one winner per sim)",
                abs(total_wins - N_SIMS) <= max(5, N_SIMS * 0.02),
                f"total_wins={total_wins} n_sims={N_SIMS}")

    print("\n" + ("ALL PASS" if ok else "FAILURES PRESENT"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
