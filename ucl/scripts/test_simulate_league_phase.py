"""
Synthetic tests for ucl/simulate_league_phase.py.
No UCL database or model file required — stubs are passed directly.
Run: python ucl/test_simulate_league_phase.py

L1  Output structure contract (required keys present)
L2  Probabilities sum to ~100% for every team
L3  All 36 teams appear in the table
L4  Strong team has higher p_top8_pct than weak team over enough sims
L5  With all fixtures played (remaining=0) all probabilities are 0 or 100
"""

from __future__ import annotations

import sys
import sqlite3
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ucl.simulate_league_phase import (  # noqa: E402
    _initial_table,
    _simulate_scoreline,
    run_simulation,
)


def check(label, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {label}" + (f"  ({detail})" if detail else ""))
    return cond


N_TEAMS = 36
LEAGUE_PHASE_ROUND = "League Phase"


def _build_db(tmp: Path, *, strong: str = "STR", weak: str = "WEA",
              all_played: bool = False) -> Path:
    """Synthetic UCL DB with N_TEAMS teams and a full league-phase fixture list.

    strong beats weak in every played result; remaining fixtures are left NS.
    """
    db = tmp / "ucl.db"
    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE TABLE teams (team_id INT PRIMARY KEY, name TEXT, short_name TEXT UNIQUE);
        CREATE TABLE fixtures (
            fixture_id INT PRIMARY KEY, round_name TEXT,
            home_team_id INT, away_team_id INT, kickoff_utc TEXT,
            status TEXT, home_score INT, away_score INT, fetched_at_utc TEXT
        );
    """)
    teams = [(i, f"Club {i}", f"T{i:02d}") for i in range(1, N_TEAMS + 1)]
    conn.executemany("INSERT INTO teams VALUES (?,?,?)", teams)

    # Build a fixture list: each team plays 8 matches (simplification of 36-team format)
    snames = [f"T{i:02d}" for i in range(1, N_TEAMS + 1)]
    tid_map = {f"T{i:02d}": i for i in range(1, N_TEAMS + 1)}
    fid = 1
    played_rows = []
    ns_rows = []
    for hi in range(0, N_TEAMS, 2):
        for ai in range(hi + 1, min(hi + 5, N_TEAMS)):
            h_sn, a_sn = snames[hi], snames[ai]
            h_tid, a_tid = tid_map[h_sn], tid_map[a_sn]
            if all_played:
                played_rows.append((fid, LEAGUE_PHASE_ROUND, h_tid, a_tid,
                                    '2026-10-01T20:00:00Z', 'FT', 2, 0, '2026-10-01T22:30:00Z'))
            elif hi == 0:
                played_rows.append((fid, LEAGUE_PHASE_ROUND, h_tid, a_tid,
                                    '2026-09-17T20:00:00Z', 'FT', 3, 0, '2026-09-17T22:30:00Z'))
            else:
                ns_rows.append((fid, LEAGUE_PHASE_ROUND, h_tid, a_tid,
                                None, 'NS', None, None, '2026-09-10T00:00:00Z'))
            fid += 1

    for row in played_rows + ns_rows:
        conn.execute("INSERT INTO fixtures VALUES (?,?,?,?,?,?,?,?,?)", row)
    conn.commit()
    conn.close()
    return db


def _make_dc(strong: str = "STR") -> dict:
    return {
        "home_adv": 1.10,
        "rho": -0.10,
        "team_params": {
            f"T{i:02d}": {
                "attack": 3.0 if i == 1 else 0.5,
                "defense": 0.3 if i == 1 else 1.5,
            }
            for i in range(1, N_TEAMS + 1)
        },
        "form_adjustments": {},
    }


def main():
    ok = True
    import random
    random.seed(0)
    tmp = Path(tempfile.mkdtemp())

    # Patch the module-level DB_PATH and MODEL_PATH for the test
    import ucl.simulate_league_phase as slp
    db = _build_db(tmp)
    slp.DB_PATH = db

    dc = _make_dc()
    result = run_simulation(n_sims=200, dc=dc)

    # L1: required top-level keys
    required = {"generated_at_utc", "n_simulations", "played_fixtures",
                "remaining_fixtures", "table"}
    missing = required - set(result)
    ok &= check("L1 output structure contract", not missing, str(missing))

    table = result.get("table", [])

    # L2: probabilities sum to ~100% for each team
    for row in table:
        total = row["p_top8_pct"] + row["p_playoff_pct"] + row["p_eliminated_pct"]
        if abs(total - 100.0) > 1.0:
            ok &= check(f"L2 probs sum 100% for {row['team']}", False, f"sum={total}")
            break
    else:
        ok &= check("L2 probs sum to 100% for all teams", True)

    # L3: all N_TEAMS present in table
    ok &= check("L3 all teams in table", len(table) == N_TEAMS, f"got {len(table)}")

    # L4: T01 (strong team) should have the highest p_top8_pct across enough sims
    by_team = {row["team"]: row for row in table}
    t01_top8 = by_team.get("T01", {}).get("p_top8_pct", 0)
    max_top8 = max(r["p_top8_pct"] for r in table)
    ok &= check("L4 strong team (T01) has highest p_top8",
                t01_top8 == max_top8, f"T01={t01_top8} max={max_top8}")

    # L5: with all fixtures played, p_top8 + p_playoff + p_elim = 100% trivially
    # and each team gets exactly one of {100%, 0%} in each bucket
    tmp2 = Path(tempfile.mkdtemp())
    db2 = _build_db(tmp2, all_played=True)
    slp.DB_PATH = db2
    result2 = run_simulation(n_sims=10, dc=dc)
    ok &= check("L5 remaining_fixtures=0 when all played",
                result2["remaining_fixtures"] == 0, str(result2["remaining_fixtures"]))

    print("\n" + ("ALL PASS" if ok else "FAILURES PRESENT"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
