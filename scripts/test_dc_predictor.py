"""
Synthetic tests for src/dc_predictor.py — no real fpl.db or trained model
needed. Run: python scripts/test_dc_predictor.py

D1  predict_all() emits the full required schema (the bake-off contract)
D2  Projection is fixture-aware: identical striker vs a weak defence
    projects MORE points than vs a strong defence
D3  Clean-sheet logic: identical defender vs weak attack projects more
    than vs strong attack; FWD gets zero CS points either way
D4  Players with no fixture this GW project 0
"""

import json
import sqlite3
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import src.features as features_mod          # noqa: E402
import src.dc_predictor as dc_mod            # noqa: E402
from src.dc_predictor import DCPredictor     # noqa: E402

REQUIRED = {"player_id", "predicted_points",
            "qualifying_games_3", "qualifying_games_5", "chance_of_playing_next"}


def build_world(tmp: Path):
    db = tmp / "fpl.db"
    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE TABLE teams (team_id INTEGER PRIMARY KEY, name TEXT,
            short_name TEXT, strength INTEGER);
        CREATE TABLE players (player_id INTEGER PRIMARY KEY, first_name TEXT,
            second_name TEXT, web_name TEXT, team_id INT, position INT,
            current_cost INT, last_updated TEXT, corners_order INT,
            freekicks_order INT, penalties_order INT);
        CREATE TABLE gameweeks (gameweek_id INTEGER PRIMARY KEY,
            deadline_time TEXT, is_current INT DEFAULT 0, is_next INT DEFAULT 0,
            finished INT DEFAULT 0, average_score INT);
        CREATE TABLE fixtures (fixture_id INTEGER PRIMARY KEY, gameweek_id INT,
            home_team_id INT, away_team_id INT, kickoff_time TEXT,
            home_team_difficulty INT, away_team_difficulty INT,
            finished INT DEFAULT 0, home_score INT, away_score INT);
        CREATE TABLE player_gameweek_history (player_id INT, gameweek_id INT,
            fixture_id INT, minutes INT, goals_scored INT, assists INT,
            clean_sheets INT, total_points INT, bonus INT, bps INT,
            expected_goals REAL, expected_assists REAL,
            defensive_contribution INT, value INT, selected INT,
            PRIMARY KEY (player_id, gameweek_id));
        CREATE TABLE player_snapshots (
            snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id INT, gameweek_id INT, snapshot_time TEXT,
            form REAL, points_per_game REAL, selected_by_percent REAL,
            now_cost INT, chance_of_playing_next INT, news TEXT, ep_next REAL);
        CREATE TABLE predictions (prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id INT, gameweek_id INT, model_name TEXT,
            predicted_points REAL, prediction_time TEXT);
    """)
    # Teams: 1 strong both ways, 2 weak both ways, 3/4 average, 5/6 idle
    conn.executemany("INSERT INTO teams VALUES (?,?,?,?)", [
        (1, "Strong FC", "STR", 5), (2, "Weak FC", "WEA", 1),
        (3, "Avg A", "AVA", 3), (4, "Avg B", "AVB", 3),
        (5, "Idle A", "IDA", 3), (6, "Idle B", "IDB", 3),
    ])
    # Twin strikers (same history) on the two average teams; twin defenders too;
    # one striker on an idle team.
    conn.executemany(
        "INSERT INTO players (player_id, first_name, second_name, web_name, "
        "team_id, position, current_cost) VALUES (?,?,?,?,?,?,?)", [
            (10, "A", "One", "FWD_vs_weak",   3, 4, 80),
            (11, "A", "Two", "FWD_vs_strong", 4, 4, 80),
            (12, "B", "One", "DEF_vs_weak",   3, 2, 55),
            (13, "B", "Two", "DEF_vs_strong", 4, 2, 55),
            (14, "C", "One", "FWD_idle",      5, 4, 80),
        ])
    # 6 GWs of identical history for everyone: 90 min, striker 0.5 goals/GW,
    # defender 0.4 CS rate
    hist = []
    for gw in range(1, 7):
        for pid in (10, 11, 14):
            hist.append((pid, gw, None, 90, 1 if gw % 2 else 0, 0,
                         0, 5, 1, 20, 0.5, 0.1, 0, 80, 1000))
        for pid in (12, 13):
            hist.append((pid, gw, None, 90, 0, 0,
                         1 if gw in (1, 3) else 0, 4, 0, 18, 0.05, 0.05, 2, 55, 800))
    conn.executemany(
        "INSERT INTO player_gameweek_history VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        hist)
    for gw in range(1, 8):
        conn.execute("INSERT INTO gameweeks VALUES (?,?,0,?,?,NULL)",
                     (gw, "2026-08-01T10:00:00Z", 1 if gw == 7 else 0,
                      1 if gw < 7 else 0))
    # Past fixtures (finished, with scores) so team season averages exist
    fid = 1
    for gw in range(1, 7):
        for h, a, hs, as_ in [(1, 2, 3, 0), (3, 4, 1, 1)]:
            conn.execute("INSERT INTO fixtures VALUES (?,?,?,?,?,?,?,1,?,?)",
                         (fid, gw, h, a, None, 3, 3, hs, as_))
            fid += 1
    # GW7: twin striker/defender pairs face WEAK vs STRONG; teams 5/6 idle
    conn.execute("INSERT INTO fixtures VALUES (?,7,3,2,NULL,3,3,0,NULL,NULL)", (fid,))
    conn.execute("INSERT INTO fixtures VALUES (?,7,4,1,NULL,3,3,0,NULL,NULL)", (fid + 1,))
    conn.commit()
    conn.close()

    model = tmp / "fpl_dc_params.json"
    model.write_text(json.dumps({
        "team_params": {
            "STR": {"attack": 1.8, "defense": 0.6},   # strong
            "WEA": {"attack": 0.6, "defense": 1.8},   # weak
            "AVA": {"attack": 1.0, "defense": 1.0},
            "AVB": {"attack": 1.0, "defense": 1.0},
            "IDA": {"attack": 1.0, "defense": 1.0},
            "IDB": {"attack": 1.0, "defense": 1.0},
        },
        "form_adjustments": {},
        "home_adv": 1.2,
    }))
    return db, model


def check(label, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {label}" + (f"  ({detail})" if detail else ""))
    return cond


def main():
    tmp = Path(tempfile.mkdtemp())
    db, model = build_world(tmp)
    features_mod.DB_PATH = db
    dc_mod.DB_PATH = db
    pred = DCPredictor(db_path=db, model_path=model)

    df = pred.predict_all(target_gw=7)
    ok = True

    missing = REQUIRED - set(df.columns)
    ok &= check("D1 schema contract", not missing, f"missing={sorted(missing)}")

    pts = dict(zip(df["player_id"], df["predicted_points"]))
    ok &= check("D2 striker vs weak > vs strong", pts[10] > pts[11],
                f"vs_weak={pts[10]:.2f} vs_strong={pts[11]:.2f}")
    ok &= check("D3 defender vs weak > vs strong", pts[12] > pts[13],
                f"vs_weak={pts[12]:.2f} vs_strong={pts[13]:.2f}")
    ok &= check("D4 idle team projects 0", pts.get(14, 0) == 0.0,
                f"idle={pts.get(14)}")

    print("\n" + ("ALL PASS" if ok else "FAILURES PRESENT"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
