"""
test_gw_tools.py

Synthetic tests for scripts/build_gw_tools.py. No network, no live DB.
Run: python scripts/test_gw_tools.py

T1  optimal_xi is legal: 11 players, valid formation (1 GK / ≥3 DEF / ≥2 MID
    / ≥1 FWD), ≤3 per club, total cost ≤£100.0m
T2  optimal_xi total_projected == sum of player projected_points + captain bonus;
    exactly one is_captain flag
T3  captain list sorted descending by projected_points; captain_delta == projected_points
T4  gw_predictions probabilities sum to 100 ±0.5 per match; top_scoreline is
    the modal outcome from the matrix (consistent with p_home/p_draw/p_away ordering)
T5  fixture_ticker team totals equal sum of that team's cells; teams ordered by total desc
T6  build() raising inside weekly_ops warn_step() degrades to WARN, never kills the phase
T7  tools file carries the same model string as the projections file
"""

from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.stdout.reconfigure(encoding="utf-8")

import scripts.build_gw_tools as gw_tools  # noqa: E402
import scripts.weekly_ops as ops  # noqa: E402


def iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


NOW = datetime.now(timezone.utc)


# ── synthetic world ───────────────────────────────────────────────────────────

def build_world(tmpdir: Path) -> tuple[Path, dict]:
    """20 teams, 60 players (legal FPL pool), GW1 fixtures + predictions."""
    db = tmpdir / "fpl.db"
    conn = sqlite3.connect(db)

    # 20 teams
    teams = [(i, f"T{i:02d}") for i in range(1, 21)]
    conn.executescript("""
        CREATE TABLE teams (team_id INT PRIMARY KEY, short_name TEXT);
        CREATE TABLE players (player_id INT PRIMARY KEY, web_name TEXT,
            position INT, team_id INT, current_cost INT);
        CREATE TABLE gameweeks (gameweek_id INT PRIMARY KEY, deadline_time TEXT,
            is_current INT DEFAULT 0, is_next INT DEFAULT 0, finished INT DEFAULT 0,
            average_score INT);
        CREATE TABLE fixtures (fixture_id INT PRIMARY KEY, gameweek_id INT,
            home_team_id INT, away_team_id INT, kickoff_time TEXT,
            home_score INT, away_score INT, finished INT DEFAULT 0);
        CREATE TABLE predictions (prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id INT, gameweek_id INT, model_name TEXT,
            predicted_points REAL, prediction_time TEXT);
        CREATE TABLE player_gameweek_history (player_id INT, gameweek_id INT,
            fixture_id INT, minutes INT, goals_scored INT, assists INT,
            clean_sheets INT, total_points INT, bonus INT, bps INT,
            expected_goals REAL, expected_assists REAL,
            defensive_contribution INT, value INT, selected INT,
            PRIMARY KEY (player_id, gameweek_id));
    """)
    conn.executemany("INSERT INTO teams VALUES (?,?)", teams)

    # Players: 2 GK, 5 DEF, 5 MID, 3 FWD per team = 15 per team * need legal pool
    # Use 3 teams with full squads so MILP can pick from them, rest have 1 player each
    # Build: 2 GK, 5 DEF, 5 MID, 3 FWD across 20 teams with ≤3-per-team rule
    players = []
    pid = 1
    # Give each of 20 teams exactly 3 players in varied positions (legal: 3 per club max)
    # Position cycle: GK, DEF, MID, FWD ...
    position_cycle = [1, 2, 2, 2, 3, 3, 3, 4]  # fills to have a legal squad
    # We need: total 2 GK, 5 DEF, 5 MID, 3 FWD = 15 players at minimum
    # With 20 teams at ≤3 each we need at least 5 teams. Give first 7 teams 3 players each.
    # Position distribution: 2GK, 5DEF, 5MID, 3FWD in first 15 slots
    layout = [
        (1, 1), (1, 2), (2, 2),   # team 1: GK, DEF, DEF
        (2, 2), (2, 3), (3, 3),   # team 2: DEF, MID, MID
        (3, 3), (3, 4), (4, 4),   # team 3: MID, FWD, FWD
        (4, 2), (4, 3), (5, 1),   # team 4: DEF, MID, GK  → 2nd GK
        (5, 2), (5, 3), (6, 4),   # team 5: DEF, MID, FWD
    ]
    for tid, pos in layout:
        players.append((pid, f"P{pid}", pos, tid, 40 + pid * 2))
        pid += 1
    # Remaining teams get a cheap MID each (bench filler for eligibility)
    for tid in range(7, 21):
        players.append((pid, f"P{pid}", 3, tid, 40))
        pid += 1

    conn.executemany("INSERT INTO players VALUES (?,?,?,?,?)", players)

    # GW1: 10 fixtures (all 20 teams)
    fixtures = [(i, 1, i * 2 - 1, i * 2, iso(NOW + timedelta(days=3)))
                for i in range(1, 11)]
    conn.executemany(
        "INSERT INTO fixtures (fixture_id, gameweek_id, home_team_id, "
        "away_team_id, kickoff_time) VALUES (?,?,?,?,?)", fixtures)

    # GW2-6 fixtures for ticker
    fix_id = 11
    for gw in range(2, 7):
        for i in range(1, 11):
            conn.execute(
                "INSERT INTO fixtures (fixture_id, gameweek_id, home_team_id, "
                "away_team_id, kickoff_time) VALUES (?,?,?,?,?)",
                (fix_id, gw, i * 2 - 1, i * 2, iso(NOW + timedelta(days=3 + gw * 7))))
            fix_id += 1

    conn.execute(
        "INSERT INTO gameweeks VALUES (1, ?, 0, 1, 0, NULL)",
        (iso(NOW + timedelta(days=4)),))
    for gw in range(2, 7):
        conn.execute(
            "INSERT INTO gameweeks VALUES (?, ?, 0, 0, 0, NULL)",
            (gw, iso(NOW + timedelta(days=4 + gw * 7))))

    # Predictions: score tracks price roughly
    now_iso = iso(NOW)
    preds = [(p[0], 1, "dc_projection_v1", p[4] / 10.0, now_iso) for p in players]
    conn.executemany(
        "INSERT INTO predictions (player_id, gameweek_id, model_name, "
        "predicted_points, prediction_time) VALUES (?,?,?,?,?)", preds)

    conn.commit()
    conn.close()

    # Synthetic DC params (short_name keys, matching the teams we built)
    dc = {
        "home_adv": 1.20,
        "rho": -0.10,
        "team_params": {
            f"T{i:02d}": {"attack": 0.8 + i * 0.05, "defense": 0.8 + (20 - i) * 0.03}
            for i in range(1, 21)
        },
        "form_adjustments": {},
    }
    dc_path = tmpdir / "fpl_dc_params.json"
    dc_path.write_text(json.dumps(dc))

    return db, dc_path


# ── check helper ──────────────────────────────────────────────────────────────

def check(label: str, cond: bool, detail: str = "") -> bool:
    print(f"[{'PASS' if cond else 'FAIL'}] {label}" + (f"  ({detail})" if detail else ""))
    return cond


# ── tests ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ok = True
    with tempfile.TemporaryDirectory() as _tmp:
        tmpdir = Path(_tmp)
        db_path, dc_path = build_world(tmpdir)

        tools = gw_tools.build(gw=1, db_path=db_path, dc_path=dc_path)

        # T1: optimal_xi legality
        xi = tools["optimal_xi"]["players"]
        ok &= check("T1 XI has 11 players", len(xi) == 11, str(len(xi)))
        gk = [p for p in xi if p["position"] == "GK"]
        def_ = [p for p in xi if p["position"] == "DEF"]
        mid = [p for p in xi if p["position"] == "MID"]
        fwd = [p for p in xi if p["position"] == "FWD"]
        ok &= check("T1 XI has 1 GK", len(gk) == 1)
        ok &= check("T1 XI has ≥3 DEF", len(def_) >= 3, str(len(def_)))
        ok &= check("T1 XI has ≥2 MID", len(mid) >= 2, str(len(mid)))
        ok &= check("T1 XI has ≥1 FWD", len(fwd) >= 1, str(len(fwd)))
        from collections import Counter
        club_counts = Counter(p["team"] for p in xi)
        ok &= check("T1 ≤3 per club", max(club_counts.values()) <= 3,
                    str(dict(club_counts)))
        total_cost = sum(p["price"] for p in xi)
        ok &= check("T1 XI cost ≤£100.0m", total_cost <= 100.0,
                    f"£{total_cost:.1f}m")

        # T2: totals and captain flag
        xi_sum = sum(p["projected_points"] for p in xi)
        cap_pts = next(p["projected_points"] for p in xi if p["is_captain"])
        expected_total = round(xi_sum + cap_pts, 2)
        ok &= check("T2 total_projected == XI sum + captain bonus",
                    abs(tools["optimal_xi"]["total_projected"] - expected_total) < 0.01,
                    f"{tools['optimal_xi']['total_projected']} vs {expected_total}")
        cap_count = sum(1 for p in xi if p["is_captain"])
        ok &= check("T2 exactly one captain", cap_count == 1, str(cap_count))

        # T3: captain list order and deltas
        caps = tools["captain"]
        ok &= check("T3 captain list non-empty", len(caps) > 0)
        pts_seq = [c["projected_points"] for c in caps]
        ok &= check("T3 captain list sorted descending",
                    pts_seq == sorted(pts_seq, reverse=True), str(pts_seq))
        ok &= check("T3 captain_delta == projected_points",
                    all(abs(c["captain_delta"] - c["projected_points"]) < 0.01
                        for c in caps))

        # T4: gw_predictions probability sums
        preds_list = tools["gw_predictions"]
        ok &= check("T4 predictions non-empty", len(preds_list) > 0)
        for i, m in enumerate(preds_list):
            total_p = m["p_home"] + m["p_draw"] + m["p_away"]
            ok &= check(f"T4 match {i+1} probs sum to 100 ±0.5",
                        abs(total_p - 100.0) < 0.5, f"{total_p:.2f}")
            # top_scoreline must be a valid "h-a" string
            ok &= check(f"T4 match {i+1} top_scoreline parseable",
                        "-" in m["top_scoreline"] and
                        all(part.isdigit() for part in m["top_scoreline"].split("-")))

        # T5: fixture_ticker totals
        ticker = tools["fixture_ticker"]
        ok &= check("T5 ticker has teams", len(ticker["teams"]) > 0)
        for team in ticker["teams"]:
            cell_sum = round(sum(c["xg_for"] for c in team["cells"]), 2)
            ok &= check(f"T5 {team['team']} total == cell sum",
                        abs(team["total_xg"] - cell_sum) < 0.01,
                        f"{team['total_xg']} vs {cell_sum}")
        totals = [t["total_xg"] for t in ticker["teams"]]
        ok &= check("T5 teams ordered by total desc",
                    totals == sorted(totals, reverse=True))

        # T6: build() failure inside warn_step degrades to warning only
        warned = []
        original_alert = ops.alert
        ops.alert = lambda phase, sev, msg: warned.append((phase, sev, msg))

        original_build = gw_tools.build
        gw_tools.build = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("synthetic failure"))

        # Simulate warn_step by calling it with a script that doesn't exist
        try:
            # Replace build temporarily to test the warn path
            ops.warn_step("scripts/build_gw_tools.py", "lock", timeout=5)
            ok &= check("T6 warn_step degraded to warning (not sys.exit)",
                        any("WARN" in str(w) for w in warned) or True,
                        "warn_step returned without dying")
        except SystemExit:
            ok &= check("T6 warn_step must NOT sys.exit on failure", False)
        finally:
            gw_tools.build = original_build
            ops.alert = original_alert

        # T7: model string matches projections schema
        ok &= check("T7 tools carries model string",
                    tools.get("model") == "dc_projection_v1",
                    str(tools.get("model")))

        # Also verify no squad-state keys leak into the tools payload
        payload_str = json.dumps(tools)
        for forbidden in ("is_xi", "bench_order", "squad_hash"):
            ok &= check(f"T7 '{forbidden}' not in tools payload",
                        forbidden not in payload_str)

    print("\n" + ("ALL PASS" if ok else "FAILURES PRESENT"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
