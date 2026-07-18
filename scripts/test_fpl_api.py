"""
test_fpl_api.py

TestClient tests for src/fpl_api.py against a synthetic DB.
Run: python scripts/test_fpl_api.py

A1  Unlocked GW -> 404
A2  Locked, BEFORE deadline -> commitment only: hash present, NO squad,
    NO transfers — the endpoint must not leak what commit-reveal withholds
A3  Locked, AFTER deadline, ungraded -> squad revealed with names, result null
A4  Graded -> result present: net/gross/hits, named effective captain, autosubs
A5  Season -> per-GW rows, cumulative, vs-average record
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

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import src.fpl_api as fpl_api  # noqa: E402
from src.squad_commit import CANONICAL, compute_squad_hash  # noqa: E402

NOW = datetime(2026, 8, 10, 12, 0, tzinfo=timezone.utc)
PAST = (NOW - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
FUTURE = (NOW + timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")

POS = {1: 1, 2: 2, 3: 2, 4: 2, 5: 3, 6: 3, 7: 3, 8: 3, 9: 3,
       10: 4, 11: 4, 12: 1, 13: 3, 14: 2, 15: 4}
XI = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
BENCH_ORDER = {12: 1, 13: 2, 14: 3, 15: 4}


def squad_rows():
    return [{"player_id": pid, "purchase_price": 50,
             "is_xi": int(pid in XI),
             "is_captain": int(pid == 10),
             "is_vice": int(pid == 5),
             "bench_order": BENCH_ORDER.get(pid, 0)} for pid in POS]


def build_db(tmp: Path) -> Path:
    db = tmp / "fpl.db"
    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE TABLE teams (team_id INTEGER PRIMARY KEY, name TEXT,
            short_name TEXT, strength INT);
        CREATE TABLE players (player_id INTEGER PRIMARY KEY, web_name TEXT,
            position INT, team_id INT, current_cost INT);
        CREATE TABLE gameweeks (gameweek_id INTEGER PRIMARY KEY,
            deadline_time TEXT, is_current INT, is_next INT, finished INT,
            average_score INT);
        CREATE TABLE model_squad_log (gameweek_id INT PRIMARY KEY,
            locked_at_utc TEXT, deadline_utc TEXT, squad_json TEXT,
            transfers_json TEXT, free_transfers INT, bank INT,
            expected_points REAL, squad_hash TEXT);
        CREATE TABLE model_gw_results (gameweek_id INT PRIMARY KEY,
            graded_at_utc TEXT, gross_points INT, hit_points INT,
            net_points INT, effective_captain INT, autosubs_json TEXT,
            detail_json TEXT);
    """)
    conn.execute("INSERT INTO teams VALUES (1,'Testers FC','TST',3)")
    conn.executemany("INSERT INTO players VALUES (?,?,?,?,?)",
                     [(pid, f"P{pid}", pos, 1, 50) for pid, pos in POS.items()])
    conn.executemany("INSERT INTO gameweeks VALUES (?,?,0,0,?,?)",
                     [(1, PAST, 1, 52), (2, PAST, 1, 60), (3, FUTURE, 0, None)])

    sq = squad_rows()
    h = compute_squad_hash(sq)
    # GW1: locked (deadline past) + graded
    conn.execute("INSERT INTO model_squad_log VALUES (1,?,?,?,?,1,5,62.4,?)",
                 (PAST, PAST, json.dumps(sq, **CANONICAL),
                  json.dumps({"in": [10], "out": [15], "hits": 4}), h))
    conn.execute("INSERT INTO model_gw_results VALUES (1,?,66,4,62,10,?,?)",
                 (PAST, json.dumps([{"out": 2, "in": 14}]), json.dumps([])))
    # GW2: locked (deadline past), NOT graded
    conn.execute("INSERT INTO model_squad_log VALUES (2,?,?,?,?,2,0,58.1,?)",
                 (PAST, PAST, json.dumps(sq, **CANONICAL),
                  json.dumps({"in": [], "out": [], "hits": 0}), h))
    # GW3: locked, deadline in the FUTURE (commitment phase)
    conn.execute("INSERT INTO model_squad_log VALUES (3,?,?,?,?,1,0,59.9,?)",
                 (PAST, FUTURE, json.dumps(sq, **CANONICAL),
                  json.dumps({"in": [], "out": [], "hits": 0}), h))
    conn.commit()
    conn.close()
    return db


def check(label, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {label}" + (f"  ({detail})" if detail else ""))
    return cond


def main():
    tmp = Path(tempfile.mkdtemp())
    fpl_api.FPL_DB_PATH = build_db(tmp)
    fpl_api._now = lambda: NOW

    app = FastAPI()
    app.include_router(fpl_api.router)
    c = TestClient(app)
    ok = True

    # A1
    r = c.get("/api/fpl/model/gw/9")
    ok &= check("A1 unlocked GW -> 404", r.status_code == 404, str(r.status_code))

    # A2: GW3 deadline is in the future
    r = c.get("/api/fpl/model/gw/3")
    j = r.json()
    ok &= check("A2 commitment phase: revealed=false", j.get("revealed") is False)
    ok &= check("A2 hash present, 64 chars", len(j.get("squad_hash", "")) == 64)
    leak = [k for k in ("squad", "transfers", "expected_points", "bank") if k in j]
    ok &= check("A2 no squad leakage pre-deadline", not leak, f"leaked={leak}")

    # A3: GW2 past deadline, ungraded
    r = c.get("/api/fpl/model/gw/2")
    j = r.json()
    ok &= check("A3 revealed with named squad", j["revealed"] is True
                and len(j["squad"]) == 15 and j["squad"][0]["name"] == "P1")
    ok &= check("A3 ungraded -> result null", j["result"] is None)

    # A4: GW1 graded
    r = c.get("/api/fpl/model/gw/1")
    j = r.json()
    res = j["result"]
    ok &= check("A4 result present", res is not None and res["net_points"] == 62)
    ok &= check("A4 effective captain named",
                res["effective_captain"]["name"] == "P10")
    ok &= check("A4 autosub named", res["autosubs"][0]["in"]["name"] == "P14")
    ok &= check("A4 transfers named", j["transfers"]["in"][0]["name"] == "P10"
                and j["transfers"]["hits"] == 4)

    # A5: season — only GW1 is graded
    r = c.get("/api/fpl/model/season")
    j = r.json()
    ok &= check("A5 one graded GW in season", len(j["gameweeks"]) == 1)
    ok &= check("A5 totals", j["model_total"] == 62 and j["average_total"] == 52)
    ok &= check("A5 vs average record", j["vs_average"] == {
        "above": 1, "below": 0, "equal": 0})
    ok &= check("A5 cumulative tracks", j["gameweeks"][0]["cumulative"] == 62)

    print("\n" + ("ALL PASS" if ok else "FAILURES PRESENT"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
