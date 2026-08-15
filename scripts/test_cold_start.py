"""
test_cold_start.py

The post-rollover season opening: live DB has players, teams and fixtures but
ZERO player history; the archived previous season has everything. Kept
separate from test_dc_predictor.py because the two-database fixture is a
different world from the mid-season one.

Run: python scripts/test_cold_start.py

K1  Post-rollover GW1 produces real, differentiated projections (the bug this
    was written for: without archive rates every player projects 0.0 and the
    optimiser returns an arbitrary squad)
K2  Schema contract holds on the cold path — the bake-off/optimiser columns
    are all present
K3  Promoted-side players with NO archive history still project > 0, and a
    dear striker outprojects a cheap midfielder (priors are price/position
    aware, not a flat constant)
K4  Self-heal: at 3 live gameweeks the archive is dropped and live rates take
    over, with no flag or date involved
K5  Missing archive is a loud refusal, not a silent zero-projection squad
K6  build_prediction_features on empty history raises the diagnosis, not a
    pandas MergeError
"""

from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import src.cold_start as cs  # noqa: E402
import src.dc_predictor as dcp  # noqa: E402
import src.features as feat  # noqa: E402
from src.dc_predictor import DCPredictor  # noqa: E402

REQUIRED = {"player_id", "predicted_points", "qualifying_games_3",
            "qualifying_games_5", "chance_of_playing_next"}

SCHEMA = """
CREATE TABLE teams (team_id INTEGER PRIMARY KEY, name TEXT, short_name TEXT,
    strength INT);
CREATE TABLE players (player_id INTEGER PRIMARY KEY, first_name TEXT,
    second_name TEXT, web_name TEXT, team_id INT, position INT,
    current_cost INT, last_updated TEXT, corners_order INT,
    freekicks_order INT, penalties_order INT);
CREATE TABLE gameweeks (gameweek_id INTEGER PRIMARY KEY, deadline_time TEXT,
    is_current INT DEFAULT 0, is_next INT DEFAULT 0, finished INT DEFAULT 0,
    average_score INT);
CREATE TABLE fixtures (fixture_id INTEGER PRIMARY KEY, gameweek_id INT,
    home_team_id INT, away_team_id INT, kickoff_time TEXT,
    home_team_difficulty INT, away_team_difficulty INT, finished INT DEFAULT 0,
    home_score INT, away_score INT);
CREATE TABLE player_gameweek_history (player_id INT, gameweek_id INT,
    fixture_id INT, minutes INT, goals_scored INT, assists INT,
    clean_sheets INT, total_points INT, bonus INT, bps INT,
    expected_goals REAL, expected_assists REAL, defensive_contribution INT,
    value INT, selected INT, PRIMARY KEY (player_id, gameweek_id));
CREATE TABLE player_snapshots (snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id INT, gameweek_id INT, snapshot_time TEXT, form REAL,
    points_per_game REAL, selected_by_percent REAL, now_cost INT,
    chance_of_playing_next INT, news TEXT, ep_next REAL);
"""

# Returning players (in archive AND live) + promoted-side players (live only)
RETURNING = [
    (10, "Star FWD", 3, 4, 110), (11, "Mid FWD", 4, 4, 75),
    (12, "Star MID", 3, 3, 95), (13, "Cheap MID", 4, 3, 45),
    (14, "Good DEF", 3, 2, 60), (15, "Cheap DEF", 4, 2, 40),
    (16, "First GK", 3, 1, 50), (17, "Backup GK", 4, 1, 40),
]
PROMOTED = [  # team 5 = promoted; no archive history whatsoever
    (30, "Hull FWD", 5, 4, 80), (31, "Hull MID", 5, 3, 45),
    (32, "Hull DEF", 5, 2, 40), (33, "Hull GK", 5, 1, 40),
    (34, "Hull MID dear", 5, 3, 90),   # same position as 31, dearer bucket
]

# Bucket priors need >= MIN_BUCKET_SAMPLES players in a (position, bucket) to
# be used at all. Fill two MID buckets in the archive so the £9.0m vs £4.5m
# comparison tests the PRIOR, not the difference in positional scoring rules.
FILLER = ([(200 + i, f"Fill dear MID {i}", 3, 3, 90) for i in range(6)]
          + [(300 + i, f"Fill cheap MID {i}", 4, 3, 45) for i in range(6)])


def _base(conn):
    conn.executescript(SCHEMA)
    conn.executemany("INSERT INTO teams VALUES (?,?,?,?)",
                     [(i, f"T{i}", f"T{i}", 3) for i in range(1, 7)])


def build_archive(tmp: Path) -> Path:
    """Last season: 6 GWs of history for the returning players only."""
    db = tmp / "fpl_2526.db"
    conn = sqlite3.connect(db)
    _base(conn)
    conn.executemany(
        "INSERT INTO players (player_id, web_name, team_id, position, "
        "current_cost) VALUES (?,?,?,?,?)",
        [(pid, nm, tm, pos, cost) for pid, nm, tm, pos, cost
         in RETURNING + FILLER])
    rows = []
    for gw in range(1, 7):
        for pid, _nm, _tm, pos, cost in RETURNING + FILLER:
            # dearer players score more — this is what the priors must learn
            goals = 1 if (cost >= 90 and gw % 2) else 0
            bonus = 2 if cost >= 90 else 0
            cs = 1 if (pos in (1, 2) and gw % 3 == 0) else 0
            rows.append((pid, gw, None, 90, goals, 0, cs, 5, bonus, 20,
                         0.6 if cost >= 90 else 0.1, 0.2, 0, cost, 1000))
    conn.executemany("INSERT INTO player_gameweek_history VALUES "
                     "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    for gw in range(1, 7):
        conn.execute("INSERT INTO gameweeks VALUES (?,?,0,0,1,50)",
                     (gw, "2025-08-01T10:00:00Z"))
        conn.execute("INSERT INTO fixtures VALUES (?,?,1,2,NULL,3,3,1,1,1)",
                     (gw, gw))
    conn.commit(); conn.close()
    return db


def build_live(tmp: Path, live_gws: int = 0, id_map: dict | None = None) -> Path:
    """New season: everyone present, `live_gws` gameweeks of history."""
    db = tmp / "fpl.db"
    conn = sqlite3.connect(db)
    _base(conn)
    m = id_map or {}
    conn.executemany(
        "INSERT INTO players (player_id, web_name, team_id, position, "
        "current_cost) VALUES (?,?,?,?,?)",
        [(m.get(pid, pid), nm, tm, pos, cost) for pid, nm, tm, pos, cost
         in RETURNING + PROMOTED])
    for gw in range(1, 9):
        conn.execute("INSERT INTO gameweeks VALUES (?,?,0,?,?,NULL)",
                     (gw, "2026-08-21T18:30:00Z", 1 if gw == 1 else 0,
                      1 if gw <= live_gws else 0))
    # past fixtures give team season averages; GW8 is the target
    fid = 100
    for gw in range(1, 8):
        conn.execute("INSERT INTO fixtures VALUES (?,?,1,2,NULL,3,3,1,1,1)",
                     (fid, gw)); fid += 1
    conn.execute("INSERT INTO fixtures VALUES (?,8,3,4,NULL,3,3,0,NULL,NULL)",
                 (fid,))
    conn.execute("INSERT INTO fixtures VALUES (?,8,5,6,NULL,3,3,0,NULL,NULL)",
                 (fid + 1,))
    if live_gws:
        rows = []
        for gw in range(1, live_gws + 1):
            for pid, _nm, _tm, _pos, cost in RETURNING + PROMOTED:
                pid = m.get(pid, pid)
                # live season: flat, low output — deliberately unlike archive
                rows.append((pid, gw, None, 90, 0, 0, 0, 2, 0, 10,
                             0.05, 0.05, 0, cost, 100))
        conn.executemany("INSERT INTO player_gameweek_history VALUES "
                         "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    conn.commit(); conn.close()
    return db


def dc_params(tmp: Path) -> Path:
    p = tmp / "params.json"
    p.write_text(json.dumps({
        "team_params": {f"T{i}": {"attack": 1.0, "defense": 1.0}
                        for i in range(1, 7)},
        "form_adjustments": {}, "home_adv": 1.2,
    }))
    return p


def check(label, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {label}" + (f"  ({detail})" if detail else ""))
    return cond


def main():
    ok = True
    tmp = Path(tempfile.mkdtemp())
    archive = build_archive(tmp)
    model = dc_params(tmp)

    # ── K1/K2/K3: GW1, zero live history ────────────────────────────────
    live = build_live(tmp, live_gws=0)
    feat.DB_PATH = live
    dcp.DB_PATH = live
    p = DCPredictor(db_path=live, model_path=model, archive_db_path=archive)
    df = p.predict_all(target_gw=8)
    pts = dict(zip(df["player_id"], df["predicted_points"]))

    nonzero = sum(1 for v in pts.values() if v > 0)
    ok &= check("K1 cold start produces non-zero projections",
                nonzero == len(pts) and nonzero > 0,
                f"{nonzero}/{len(pts)} players")
    ok &= check("K1 projections are differentiated, not a constant",
                len(set(pts.values())) > 3,
                f"{len(set(pts.values()))} distinct values")
    ok &= check("K1 archive rates carry through (£11.0m FWD tops £4.0m DEF)",
                pts[10] > pts[15], f"{pts[10]:.2f} vs {pts[15]:.2f}")

    missing = REQUIRED - set(df.columns)
    ok &= check("K2 schema contract on the cold path", not missing,
                f"missing={sorted(missing)}")

    promoted = {pid: pts[pid] for pid, *_ in PROMOTED}
    ok &= check("K3 promoted-side players are visible to the optimiser",
                all(v > 0 for v in promoted.values()), str(promoted))
    # Same position, different price bucket — isolates the prior from the
    # positional scoring rules (a MID outscores a FWD on identical output,
    # so a cross-position comparison would prove nothing about priors).
    ok &= check("K3 priors are price-aware within a position "
                "(£9.0m promoted MID > £4.5m promoted MID)",
                promoted[34] > promoted[31],
                f"{promoted[34]:.2f} vs {promoted[31]:.2f}")

    # ── K4: self-heal at 3 live gameweeks, no flag ──────────────────────
    warm_dir = tmp / "warm"
    warm_dir.mkdir(exist_ok=True)
    live3 = build_live(warm_dir, live_gws=3)
    feat.DB_PATH = live3
    dcp.DB_PATH = live3
    conn = sqlite3.connect(live3)
    still_cold = cs.is_cold_start(conn)
    conn.close()
    p3 = DCPredictor(db_path=live3, model_path=model, archive_db_path=archive)
    df3 = p3.predict_all(target_gw=8)
    pts3 = dict(zip(df3["player_id"], df3["predicted_points"]))
    ok &= check("K4 three live gameweeks ends cold start", not still_cold)
    ok &= check("K4 live rates now drive projections (archive ignored)",
                pts3[10] < pts[10],
                f"warm={pts3[10]:.2f} cold={pts[10]:.2f}")

    # ── K5: missing archive refuses loudly ──────────────────────────────
    feat.DB_PATH = live
    dcp.DB_PATH = live
    bad = DCPredictor(db_path=live, model_path=model,
                      archive_db_path=tmp / "does_not_exist.db")
    try:
        bad.predict_all(target_gw=8)
        ok &= check("K5 missing archive refuses", False, "returned!")
    except RuntimeError as e:
        ok &= check("K5 missing archive refuses", "archived season" in str(e)
                    or "archive" in str(e).lower(), str(e)[:60])

    # ── K6: the diagnostic guard ────────────────────────────────────────
    feat.DB_PATH = live  # zero history
    try:
        feat.build_prediction_features(target_gw=8)
        ok &= check("K6 empty history raises the diagnosis", False, "returned!")
    except RuntimeError as e:
        ok &= check("K6 empty history raises the diagnosis",
                    "cold-start path not taken" in str(e), str(e)[:50])
    except Exception as e:
        ok &= check("K6 empty history raises the diagnosis", False,
                    f"wrong type: {type(e).__name__}")

    # ── K7/K8: FPL reassigns player_ids between seasons ─────────────────
    # Archive: id 10 = "Star FWD" (FWD, high xG), id 16 = "First GK" (GK).
    # Live:    "First GK" now holds id 10; "Star FWD" moved to 916.
    # Matching by id would hand the GOALKEEPER a striker's expected goals —
    # multiplied by PT_GOAL["GK"], the largest coefficient in the table.
    shifted = tmp / "shifted"
    shifted.mkdir(exist_ok=True)
    live_s = build_live(shifted, live_gws=0, id_map={16: 10, 10: 916})
    feat.DB_PATH = live_s
    dcp.DB_PATH = live_s
    ps = DCPredictor(db_path=live_s, model_path=model, archive_db_path=archive)
    dfs = ps.predict_all(target_gw=8)
    ps._project(8, explain={10, 916})
    gk = ps.explained[10]
    fwd = ps.explained[916]

    ok &= check("K7 rates follow the PLAYER across reassigned ids",
                not gk["from_prior"] and not fwd["from_prior"],
                f"gk_prior={gk['from_prior']} fwd_prior={fwd['from_prior']}")
    ok &= check("K8 GK holding a striker's old id gets NO striker xG",
                gk["eff_xg"] < 0.2 and gk["PTS_goals"] < 2.0,
                f"eff_xg={gk['eff_xg']} PTS_goals={gk['PTS_goals']}")
    ok &= check("K8 the striker keeps his own xG at his new id",
                fwd["eff_xg"] > 0.4, f"eff_xg={fwd['eff_xg']}")

    print("\n" + ("ALL PASS" if ok else "FAILURES PRESENT"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
