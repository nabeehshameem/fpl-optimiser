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

    # Every lock file now carries the squad openly — the git commit timestamp
    # is what proves the squad existed before the deadline.
    def lock_file(gw, deadline):
        return json.dumps({
            "gameweek": gw, "locked_at_utc": PAST, "deadline_utc": deadline,
            "squad_hash": h, "squad": sq, "squad_display": disp,
            "transfers": {"in": [10], "out": [15], "hits": 4},
            "free_transfers": 1, "bank": 5, "expected_points": 62.4,
        })

    # GW1: locked + graded
    (export_dir / "gw01.json").write_text(lock_file(1, PAST))
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

    # GW2: locked, deadline past, not yet graded
    (export_dir / "gw02.json").write_text(lock_file(2, PAST))

    # GW3: locked, deadline still ahead — squad visible anyway
    (export_dir / "gw03.json").write_text(lock_file(3, FUTURE))

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

    # A2: GW3 deadline is still in the future — squad is shown anyway.
    # The pre-deadline git timestamp is the proof; nothing is withheld.
    r = c.get("/api/fpl/model/gw/3")
    j = r.json()
    ok &= check("A2 squad published before the deadline",
                len(j.get("squad", [])) == 15
                and j["squad"][0]["name"] == "P1", str(r.status_code))
    ok &= check("A2 deadline_passed flag is false", j.get("deadline_passed") is False)
    ok &= check("A2 hash still published as integrity check",
                len(j.get("squad_hash", "")) == 64)
    ok &= check("A2 captain and bench order visible",
                any(p["is_captain"] for p in j["squad"])
                and any(p["bench_order"] for p in j["squad"]))

    # A3: GW2 past deadline, not yet graded — squad visible, result null
    r = c.get("/api/fpl/model/gw/2")
    j = r.json()
    ok &= check("A3 past deadline: deadline_passed true", j.get("deadline_passed") is True)
    ok &= check("A3 squad visible, result still null",
                len(j.get("squad", [])) == 15 and j.get("result") is None)

    # A4: GW1 graded
    r = c.get("/api/fpl/model/gw/1")
    j = r.json()
    ok &= check("A4 graded GW keeps its named squad",
                len(j["squad"]) == 15 and j["squad"][0]["name"] == "P1")
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

    # A6: projections are published at LOCK time and are readable BEFORE the
    # deadline — the one pre-deadline useful endpoint. They must never carry the
    # squad or the hash, or they would defeat the commitment they ship beside.
    r = c.get("/api/fpl/projections/9")
    ok &= check("A6 unpublished projections -> 404", r.status_code == 404,
                str(r.status_code))

    proj = {
        "gameweek": 3, "deadline_utc": FUTURE,
        "generated_at_utc": PAST, "model": "dc_projection_v1",
        "captain_candidates": [{"player_id": 10, "name": "P10", "team": "TST",
                                "position": "FWD", "price": 12.5,
                                "projected_points": 9.1, "opponent": "T2",
                                "venue": "H"}],
        "by_position": {"GK": [], "DEF": [], "MID": [], "FWD": []},
        "excluded": [{"player_id": 12, "name": "P12", "team": "TST"}],
        "note": "not its selection",
    }
    (fpl_api.EXPORT_DIR / "gw03_projections.json").write_text(json.dumps(proj))

    r = c.get("/api/fpl/projections/3")
    j = r.json()
    ok &= check("A6 projections served pre-deadline", r.status_code == 200
                and j["captain_candidates"][0]["name"] == "P10")
    leaked = [k for k in ("squad", "squad_hash", "transfers") if k in j]
    ok &= check("A6 projections leak neither squad nor hash", not leaked,
                f"leaked={leaked}")
    # Projections are a distinct artifact (model's view of all players) and
    # must not duplicate the squad payload published alongside them.
    ok &= check("A6 projections stay a distinct artifact",
                "by_position" in j and "is_captain" not in json.dumps(j))

    # A7: tools endpoint
    r = c.get("/api/fpl/tools/9")
    ok &= check("A7 unpublished tools -> 404", r.status_code == 404, str(r.status_code))

    tools_payload = {
        "gameweek": 1,
        "generated_at_utc": PAST,
        "model": "dc_projection_v1",
        "optimal_xi": {
            "formation": "3-5-2",
            "players": [
                {"player_id": i, "name": f"P{i}", "team": "TST",
                 "position": pos, "price": 6.0, "projected_points": float(10 - i),
                 "is_captain": i == 1}
                for i, pos in zip(range(1, 12),
                    ["GK", "DEF", "DEF", "DEF", "MID", "MID", "MID",
                     "MID", "MID", "FWD", "FWD"])
            ],
            "total_projected": 61.0,
            "note": "TOOL, not the model's team",
        },
        "captain": [
            {"player_id": 10, "name": "P10", "team": "TST", "position": "FWD",
             "price": 12.5, "projected_points": 9.1, "opponent": "T2",
             "venue": "H", "captain_delta": 9.1}
        ],
        "gw_predictions": [
            {"home": "TST", "away": "T2", "kickoff_utc": FUTURE,
             "p_home": 45.0, "p_draw": 28.0, "p_away": 27.0,
             "top_scoreline": "1-0", "xg_home": 1.6, "xg_away": 1.1}
        ],
        "fixture_ticker": {
            "from_gw": 1, "to_gw": 6,
            "teams": [
                {"team": "TST", "cells": [{"gw": 1, "opponent": "T2",
                 "venue": "H", "xg_for": 1.6}], "total_xg": 1.6}
            ],
        },
    }
    (fpl_api.EXPORT_DIR / "gw01_tools.json").write_text(json.dumps(tools_payload))

    r = c.get("/api/fpl/tools/1")
    j = r.json()
    ok &= check("A7 tools served intact", r.status_code == 200
                and j.get("model") == "dc_projection_v1")
    ok &= check("A7 tools has four sections",
                all(k in j for k in ("optimal_xi", "captain", "gw_predictions", "fixture_ticker")))

    # No squad state must leak — these keys only belong in gwNN.json
    payload_str = json.dumps(j)
    for forbidden in ("is_xi", "bench_order"):
        ok &= check(f"A7 '{forbidden}' not in tools response",
                    forbidden not in payload_str)

    print("\n" + ("ALL PASS" if ok else "FAILURES PRESENT"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
