"""
Synthetic tests for scripts/export_projections.py. No real fpl.db needed.
Run: python scripts/test_export_projections.py

P1  Export exists, carries no squad/hash, top projection matches predictions table,
    all positions present, disk file round-trips cleanly.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import scripts.export_projections as ep

MODEL_NAME = "dc_projection_v1"


def build_db(tmp: Path) -> Path:
    db = tmp / "fpl.db"
    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE TABLE teams (
            team_id INT PRIMARY KEY, short_name TEXT
        );
        CREATE TABLE players (
            player_id INT PRIMARY KEY, position INT,
            web_name TEXT, team_id INT, current_cost INT
        );
        CREATE TABLE gameweeks (
            gameweek_id INT PRIMARY KEY, deadline_time TEXT,
            is_current INT, is_next INT, finished INT, average_score INT
        );
        CREATE TABLE player_gameweek_history (
            player_id INT, gameweek_id INT, minutes INT, total_points INT,
            PRIMARY KEY (player_id, gameweek_id)
        );
        CREATE TABLE predictions (
            player_id INT, gameweek_id INT, model_name TEXT,
            predicted_points REAL, prediction_time TEXT
        );
    """)
    conn.execute("INSERT INTO teams VALUES (1, 'TST')")
    conn.executemany("INSERT INTO players VALUES (?,?,?,?,?)", [
        (1, 1, 'GK1',  1, 45),
        (2, 2, 'DEF1', 1, 50),
        (3, 3, 'MID1', 1, 65),
        (4, 3, 'MID2', 1, 75),
        (5, 4, 'FWD1', 1, 90),   # highest predicted — should top captain list
        (6, 4, 'FWD2', 1, 100),
    ])
    conn.execute(
        "INSERT INTO gameweeks VALUES (1,'2026-08-21T17:30:00Z',0,1,0,NULL)"
    )
    # player_gameweek_history left empty: GW1 cold-start, qualifying gates open
    conn.executemany("INSERT INTO predictions VALUES (?,?,?,?,?)", [
        (1, 1, MODEL_NAME, 3.5, '2026-08-17T10:00:00Z'),
        (2, 1, MODEL_NAME, 4.2, '2026-08-17T10:00:00Z'),
        (3, 1, MODEL_NAME, 7.8, '2026-08-17T10:00:00Z'),
        (4, 1, MODEL_NAME, 6.1, '2026-08-17T10:00:00Z'),
        (5, 1, MODEL_NAME, 9.3, '2026-08-17T10:00:00Z'),
        (6, 1, MODEL_NAME, 5.0, '2026-08-17T10:00:00Z'),
    ])
    conn.commit()
    conn.close()
    return db


def check(label, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {label}" + (f"  ({detail})" if detail else ""))
    return cond


def main():
    tmp = Path(tempfile.mkdtemp())
    db = build_db(tmp)
    export_dir = tmp / "predictions"

    proj = ep.export(gw=1, db_path=db, export_dir=export_dir)
    out = export_dir / "gw01_projections.json"

    ok = True

    # P1a: file was written to disk
    ok &= check("P1a export file written", out.exists(), str(out))

    # P1b: no squad or squad_hash — this is not a lock artifact
    ok &= check("P1b no squad key in output", "squad" not in proj)
    ok &= check("P1b no squad_hash key in output", "squad_hash" not in proj)

    # P1c: top captain candidate matches the top-predicted player (FWD1, pid=5)
    caps = proj.get("captain_candidates", [])
    ok &= check("P1c captain_candidates present", len(caps) > 0, str(len(caps)))
    if caps:
        top_pid = caps[0]["player_id"]
        ok &= check(
            "P1c top captain matches top prediction (FWD1 pid=5)",
            top_pid == 5, f"got pid={top_pid}"
        )

    # P1d: all four positions covered in by_position
    by_pos = proj.get("by_position", {})
    ok &= check(
        "P1d all four positions in by_position",
        set(by_pos) == {"GK", "DEF", "MID", "FWD"},
        str(set(by_pos))
    )

    # P1e: disk file round-trips (content equals the returned dict)
    on_disk = json.loads(out.read_text())
    ok &= check("P1e disk file round-trips cleanly", on_disk == proj)

    print("\n" + ("ALL PASS" if ok else "FAILURES PRESENT"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
