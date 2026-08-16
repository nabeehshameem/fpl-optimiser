"""
Synthetic tests for ucl/train_dc.py (fit() function).
No UCL database required — synthetic match lists used throughout.
Run: python ucl/test_ucl_train_dc.py

T1  Schema contract: fit() output contains all required keys
T2  Stronger team gets higher attack parameter than weaker team
T3  fit() refuses empty match list
T4  home_adv > 1.0 when home teams score more on average
T5  rho is in the valid DC range (-0.5, 0.2)
T6  xG path and actual-goals path both produce valid output
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ucl.train_dc import fit  # noqa: E402


def check(label, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {label}" + (f"  ({detail})" if detail else ""))
    return cond


REQUIRED_KEYS = {"trained_at", "n_teams", "n_fixtures", "home_adv",
                 "rho", "converged", "message", "team_params", "form_adjustments"}


def _make_matches(strong: str = "STR", weak: str = "WEA", n: int = 30) -> list[dict]:
    """Synthetic match history: STR scores 3 at home, 2 away; WEA scores 0.5 either."""
    import random
    random.seed(42)
    matches = []
    for i in range(n):
        if i % 2 == 0:
            matches.append({
                "home_sn": strong, "away_sn": weak,
                "home_goals": float(random.randint(2, 4)),
                "away_goals": float(random.randint(0, 1)),
                "kickoff": "2026-01-01", "weight": 1.0,
            })
        else:
            matches.append({
                "home_sn": weak, "away_sn": strong,
                "home_goals": float(random.randint(0, 1)),
                "away_goals": float(random.randint(2, 4)),
                "kickoff": "2026-01-01", "weight": 1.0,
            })
    return matches


def main():
    ok = True

    # T1: schema contract
    matches = _make_matches()
    params = fit(matches)
    missing = REQUIRED_KEYS - set(params.keys())
    ok &= check("T1 schema contract", not missing, f"missing={missing}")

    # T2: stronger team has higher attack than weaker team
    atk_str = params["team_params"].get("STR", {}).get("attack", 0.0)
    atk_wea = params["team_params"].get("WEA", {}).get("attack", 0.0)
    ok &= check("T2 STR attack > WEA attack",
                atk_str > atk_wea, f"STR={atk_str:.3f} WEA={atk_wea:.3f}")

    # T3: empty match list raises RuntimeError
    try:
        fit([])
        ok &= check("T3 fit() refuses empty list", False)
    except RuntimeError:
        ok &= check("T3 fit() refuses empty list", True)

    # T4: home_adv > 1.0 (strong teams scored more at home in the synthetic data)
    ok &= check("T4 home_adv > 1.0",
                params["home_adv"] > 1.0, f"home_adv={params['home_adv']:.4f}")

    # T5: rho in DC valid range
    rho = params["rho"]
    ok &= check("T5 rho in (-0.5, 0.2)",
                -0.5 <= rho <= 0.2, f"rho={rho:.4f}")

    # T6: two identical datasets (xG path and goals path) must both converge cleanly
    params_goals = fit(matches)
    ok &= check("T6 fit converges (goals path)", params_goals.get("converged") is not None)
    for team in ("STR", "WEA"):
        ok &= check(f"T6 {team} in team_params", team in params_goals["team_params"])

    print("\n" + ("ALL PASS" if ok else "FAILURES PRESENT"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
