"""
fpl_api.py

Read-only FPL "Beat the Model" endpoints. Mounted into the main api.py app:

    from src.fpl_api import router as fpl_router
    app.include_router(fpl_router)

Disclosure rule (mirrors the commit-reveal scheme exactly):
  * No lock row                -> 404
  * Locked, deadline not passed -> commitment only: {gameweek, locked_at_utc,
                                   deadline_utc, squad_hash}. NEVER squad
                                   details — the pre-deadline API must not
                                   leak what the pre-deadline JSON withholds.
  * Deadline passed            -> full squad (named), transfers, expected
                                   points; plus graded result when present.

The API is read-only by design: the ledger is written only by the cron-side
scripts (lock_model_squad.py, grade_model_gw.py). No write endpoint exists,
so the public surface cannot corrupt the record.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FPL_DB_PATH = PROJECT_ROOT / "data" / "fpl.db"

router = APIRouter(prefix="/api/fpl", tags=["fpl"])

POS_NAMES = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


def _now() -> datetime:
    """UTC now — module-level so tests can monkeypatch time."""
    return datetime.now(timezone.utc)


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(FPL_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _parse_deadline(s: str) -> datetime:
    return datetime.fromisoformat(str(s).replace("Z", "+00:00"))


def _player_names(conn, ids: list[int]) -> dict[int, dict]:
    if not ids:
        return {}
    ph = ",".join("?" * len(ids))
    rows = conn.execute(
        f"SELECT p.player_id, p.web_name, p.position, t.short_name AS team "
        f"FROM players p LEFT JOIN teams t ON t.team_id = p.team_id "
        f"WHERE p.player_id IN ({ph})", ids).fetchall()
    return {int(r["player_id"]): {
        "name": r["web_name"],
        "position": POS_NAMES.get(r["position"], "?"),
        "team": r["team"],
    } for r in rows}


@router.get("/model/gw/{gw}")
def model_gw(gw: int) -> dict:
    conn = _connect()
    try:
        lock = conn.execute(
            "SELECT * FROM model_squad_log WHERE gameweek_id = ?", (gw,)
        ).fetchone()
        if lock is None:
            raise HTTPException(404, f"GW{gw} has not been locked.")

        deadline = _parse_deadline(lock["deadline_utc"])
        base = {
            "gameweek": gw,
            "locked_at_utc": lock["locked_at_utc"],
            "deadline_utc": lock["deadline_utc"],
            "squad_hash": lock["squad_hash"],
        }

        if _now() < deadline:
            # Commitment phase: hash only. Squad details would let anyone
            # copy the model's team before the deadline.
            return {**base, "revealed": False}

        squad_rows = json.loads(lock["squad_json"])
        transfers = json.loads(lock["transfers_json"])
        names = _player_names(
            conn,
            [r["player_id"] for r in squad_rows]
            + transfers.get("in", []) + transfers.get("out", []),
        )

        def named(pid: int) -> dict:
            return {"player_id": pid, **names.get(pid, {"name": f"#{pid}"})}

        squad = [{
            **named(r["player_id"]),
            "is_xi": bool(r["is_xi"]),
            "is_captain": bool(r["is_captain"]),
            "is_vice": bool(r["is_vice"]),
            "bench_order": r.get("bench_order", 0),
        } for r in squad_rows]

        out = {
            **base,
            "revealed": True,
            "squad": squad,
            "transfers": {
                "in": [named(p) for p in transfers.get("in", [])],
                "out": [named(p) for p in transfers.get("out", [])],
                "hits": transfers.get("hits", 0),
            },
            "free_transfers": lock["free_transfers"],
            "bank": lock["bank"],
            "expected_points": lock["expected_points"],
            "result": None,
        }

        graded = conn.execute(
            "SELECT * FROM model_gw_results WHERE gameweek_id = ?", (gw,)
        ).fetchone()
        if graded is not None:
            autosubs = json.loads(graded["autosubs_json"])
            sub_names = _player_names(
                conn, [s["out"] for s in autosubs] + [s["in"] for s in autosubs])
            eff = graded["effective_captain"]
            out["result"] = {
                "gross_points": graded["gross_points"],
                "hit_points": graded["hit_points"],
                "net_points": graded["net_points"],
                "effective_captain": (
                    {"player_id": eff, **names.get(eff, sub_names.get(eff,
                        {"name": f"#{eff}"}))} if eff is not None else None),
                "autosubs": [{
                    "out": {"player_id": s["out"],
                            **sub_names.get(s["out"], {"name": f"#{s['out']}"})},
                    "in": {"player_id": s["in"],
                           **sub_names.get(s["in"], {"name": f"#{s['in']}"})},
                } for s in autosubs],
                "graded_at_utc": graded["graded_at_utc"],
            }
        return out
    finally:
        conn.close()


@router.get("/model/season")
def model_season() -> dict:
    conn = _connect()
    try:
        rows = conn.execute("""
            SELECT r.gameweek_id AS gw, r.net_points, r.hit_points,
                   g.average_score
            FROM model_gw_results r
            JOIN gameweeks g ON g.gameweek_id = r.gameweek_id
            ORDER BY r.gameweek_id
        """).fetchall()

        gws, model_total, avg_total = [], 0, 0
        above = below = equal = 0
        for r in rows:
            avg = r["average_score"]
            model_total += r["net_points"]
            if avg is not None:
                avg_total += avg
                if r["net_points"] > avg:
                    above += 1
                elif r["net_points"] < avg:
                    below += 1
                else:
                    equal += 1
            gws.append({
                "gameweek": r["gw"],
                "net_points": r["net_points"],
                "hit_points": r["hit_points"],
                "fpl_average": avg,
                "cumulative": model_total,
            })

        return {
            "gameweeks": gws,
            "model_total": model_total,
            "average_total": avg_total,
            "vs_average": {"above": above, "below": below, "equal": equal},
        }
    finally:
        conn.close()
