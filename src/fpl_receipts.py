"""
fpl_receipts.py

GET /api/fpl/receipt/{gw}/{team_id} — the comparison object the card renders.

External dependency policy (the first one in this system, so the rules are
explicit):
  * All FPL API access goes through _fetch_json() — one function, injectable
    in tests, never called from anywhere else.
  * A receipt for a finished gameweek is IMMUTABLE: fetched once, cached in
    the receipts table, never refetched. FPL history does not change.
  * On first request for a team, missing graded gameweeks are backfilled
    (sequentially, bounded by season length) so the season H2H is complete —
    a "Model leads 9-3" headline computed from partial history would be a
    false receipt.
  * FPL points semantics: entry_history.points is GROSS (hits not deducted);
    event_transfers_cost is the hit. Net = points - cost, matching how the
    model's net_points is computed. Auto-subs and captaincy are already
    applied inside FPL's number — no reimplementation on the user side.

Read-only toward the ledger; writes only to its own receipts cache table.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import httpx
from fastapi import APIRouter, HTTPException

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPORT_DIR = PROJECT_ROOT / "predictions" / "fpl"
RECEIPTS_DB_PATH = Path(
    os.getenv("RECEIPTS_DB", str(PROJECT_ROOT / "data" / "receipts.db"))
)

FPL_BASE = "https://fantasy.premierleague.com/api"
_HEADERS = {"User-Agent": "themodelsays.com receipt service"}

router = APIRouter(prefix="/api/fpl", tags=["fpl-receipts"])

RECEIPTS_SQL = """
CREATE TABLE IF NOT EXISTS receipts (
    gameweek_id   INTEGER NOT NULL,
    team_id       INTEGER NOT NULL,
    fetched_at    TEXT NOT NULL,
    team_name     TEXT,
    points_gross  INTEGER NOT NULL,
    hit_points    INTEGER NOT NULL,
    points_net    INTEGER NOT NULL,
    PRIMARY KEY (gameweek_id, team_id)
)
"""


def _connect() -> sqlite3.Connection:
    RECEIPTS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(RECEIPTS_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(RECEIPTS_SQL)
    return conn


def _fetch_json(url: str) -> dict:
    """Single chokepoint for FPL API access. Raises HTTPException on failure."""
    try:
        r = httpx.get(url, headers=_HEADERS, timeout=10.0)
    except httpx.HTTPError as e:
        raise HTTPException(502, f"FPL API unreachable: {e}") from e
    if r.status_code == 404:
        raise HTTPException(404, "FPL team not found.")
    if r.status_code != 200:
        raise HTTPException(502, f"FPL API returned {r.status_code}.")
    return r.json()


def _fetch_entry_name(team_id: int) -> str | None:
    data = _fetch_json(f"{FPL_BASE}/entry/{team_id}/")
    return data.get("name")


def _fetch_gw_points(team_id: int, gw: int) -> tuple[int, int]:
    """(gross_points, hit_cost) for a team's finished gameweek."""
    data = _fetch_json(f"{FPL_BASE}/entry/{team_id}/event/{gw}/picks/")
    eh = data.get("entry_history", {})
    return int(eh.get("points", 0)), int(eh.get("event_transfers_cost", 0))


def _graded_gws() -> dict[int, int]:
    """{gameweek_id: model_net_points} from committed result JSON files."""
    out: dict[int, int] = {}
    if not EXPORT_DIR.exists():
        return out
    for f in sorted(EXPORT_DIR.glob("gw*_result.json")):
        data = json.loads(f.read_text())
        gw = data.get("gameweek")
        if gw is not None:
            out[int(gw)] = int(data.get("net_points", 0))
    return out


def _cached_receipts(conn, team_id: int) -> dict[int, sqlite3.Row]:
    return {int(r["gameweek_id"]): r for r in conn.execute(
        "SELECT * FROM receipts WHERE team_id = ?", (team_id,))}


def _ensure_receipt(conn, team_id: int, gw: int,
                    team_name: str | None) -> sqlite3.Row:
    """Fetch-and-cache a single (team, gw) receipt if absent. Immutable after."""
    row = conn.execute(
        "SELECT * FROM receipts WHERE gameweek_id = ? AND team_id = ?",
        (gw, team_id)).fetchone()
    if row is not None:
        return row
    gross, hits = _fetch_gw_points(team_id, gw)
    conn.execute(
        "INSERT INTO receipts VALUES (?,?,?,?,?,?,?)",
        (gw, team_id,
         datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
         team_name, gross, hits, gross - hits))
    conn.commit()
    return conn.execute(
        "SELECT * FROM receipts WHERE gameweek_id = ? AND team_id = ?",
        (gw, team_id)).fetchone()


@router.get("/receipt/{gw}/{team_id}")
def receipt(gw: int, team_id: int) -> dict:
    conn = _connect()
    try:
        graded = _graded_gws()
        if gw not in graded:
            raise HTTPException(
                409, f"GW{gw} is not graded yet — receipts exist only for "
                     "finished, graded gameweeks.")

        cached = _cached_receipts(conn, team_id)
        from_cache = gw in cached

        # Team name: reuse any cached row's name before spending a fetch on it
        team_name = next((r["team_name"] for r in cached.values()
                          if r["team_name"]), None)
        if team_name is None and not from_cache:
            team_name = _fetch_entry_name(team_id)

        # Backfill every graded GW this team is missing (requested one included)
        for g in sorted(graded):
            if g not in cached:
                cached[g] = _ensure_receipt(conn, team_id, g, team_name)

        user_row = cached[gw]
        model_net = graded[gw]
        user_net = int(user_row["points_net"])

        h2h = {"user": 0, "model": 0, "draws": 0}
        for g, m_net in graded.items():
            u_net = int(cached[g]["points_net"])
            if u_net > m_net:
                h2h["user"] += 1
            elif m_net > u_net:
                h2h["model"] += 1
            else:
                h2h["draws"] += 1

        return {
            "gameweek": gw,
            "user": {
                "team_id": team_id,
                "team_name": user_row["team_name"],
                "points_gross": int(user_row["points_gross"]),
                "hit_points": int(user_row["hit_points"]),
                "points_net": user_net,
            },
            "model": {"net_points": model_net},
            "winner": ("user" if user_net > model_net
                       else "model" if model_net > user_net else "draw"),
            "h2h_season": h2h,
            "from_cache": from_cache,
        }
    finally:
        conn.close()
