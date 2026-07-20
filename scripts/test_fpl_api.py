"""
test_fpl_api.py

TestClient tests for src/fpl_api.py against a synthetic JSON export directory.
Run: python scripts/test_fpl_api.py

A1  Unlocked GW -> 404
A2  Locked, BEFORE deadline -> commitment only: hash present, revealed=False,
    NO squad — file physically contains only the hash, no leakage possible
A3  Locked, AFTER deadline, no result file -> revealed=True, no squad yet
A4  Graded (result file present) -> squad revealed with names, result present
A5  Season -> per-GW rows, cumulative, vs-average record
"""

from __future__ import annotations

import json
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
_POS_NAMES = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


def squad_rows():
    return [{"player_id": pid, "purchase_price": 50,
             "is_xi": int(pid in XI),
             "is_captain": int(pid == 10),
             "is_vice": int(pid == 5),
             "bench_order": BENCH_ORDER.get(pid, 0)} for pid in POS]


def build_export_dir(tmp: Path) -> Path:
    export_dir = tmp / "fpl"
    export_dir.mkdir()

    sq = squad_rows()
    h = compute_squad_hash(sq)
    disp = [{"player_id": pid, "name": f"P{pid}",
              "position": _POS_NAMES.get(POS[pid], "?"), "team": "TST"}
            for pid in sorted(POS)]

    # GW1: commitment + result (graded)
    (export_dir / "gw01.json").write_text(json.dumps({
        "gameweek": 1, "locked_at_utc": PAST, "deadline_utc": PAST,
        "squad_hash": h,
    }))
    (export_dir / "gw01_result.json").write_text(json.dumps({
        "gameweek": 1,
        "squad": sq,
        "squad_display": disp,
        "squad_hash": h,
        "graded_at_utc": PAST,
        "gross_points": 66, "hit_points": 4, "net_points": 62,
        "effective_captain": 10,
        "effective_captain_display": {
            "player_id": 10, "name": "P10", "position": "FWD", "team": "TST"},
        "autosubs": [{"out": 2, "in": 14}],
        "autosubs_display": [{
            "out": {"player_id": 2, "name": "P2", "position": "DEF", "team": "TST"},
            "in": {"player_id": 14, "name": "P14", "position": "DEF", "team": "TST"},
        }],
        "transfers": {"in": [10], "out": [15], "hits": 4},
        "transfers_display": {
            "in": [{"player_id": 10, "name": "P10", "position": "FWD", "team": "TST"}],
            "out": [{"player_id": 15, "name": "P15", "position": "FWD", "team": "TST"}],
        },
        "average_score": 52,
        "expected_points": 62.4,
        "detail": [],
    }))

    # GW2: commitment only, deadline past -> revealed=True, no squad yet
    (export_dir / "gw02.json").write_text(json.dumps({
        "gameweek": 2, "locked_at_utc": PAST, "deadline_utc": PAST,
        "squad_hash": h,
    }))

    # GW3: commitment only, deadline future -> revealed=False
    (export_dir / "gw03.json").write_text(json.dumps({
        "gameweek": 3, "locked_at_utc": PAST, "deadline_utc": FUTURE,
        "squad_hash": h,
    }))

    return export_dir


def check(label, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {label}" + (f"  ({detail})" if detail else ""))
    return cond


def main():
    tmp = Path(tempfile.mkdtemp())
    fpl_api.EXPORT_DIR = build_export_dir(tmp)
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
    ok &= check("A2 commitment phase: revealed=False", j.get("revealed") is False)
    ok &= check("A2 hash present, 64 chars", len(j.get("squad_hash", "")) == 64)
    leak = [k for k in ("squad", "transfers", "expected_points", "bank") if k in j]
    ok &= check("A2 no squad leakage pre-deadline", not leak, f"leaked={leak}")

    # A3: GW2 past deadline, no result file
    r = c.get("/api/fpl/model/gw/2")
    j = r.json()
    ok &= check("A3 post-deadline pre-grade: revealed=True", j.get("revealed") is True)
    ok &= check("A3 no squad until result file written", "squad" not in j)

    # A4: GW1 graded
    r = c.get("/api/fpl/model/gw/1")
    j = r.json()
    ok &= check("A4 revealed with named squad",
                j["revealed"] is True and len(j["squad"]) == 15
                and j["squad"][0]["name"] == "P1")
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
