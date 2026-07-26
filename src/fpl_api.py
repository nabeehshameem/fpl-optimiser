"""
fpl_api.py

Read-only FPL "Beat the Model" endpoints. Mounted into the main api.py app:

    from src.fpl_api import router as fpl_router
    app.include_router(fpl_router)

Data source: committed JSON files in predictions/fpl/
  gw{NN}.json         — written by lock_model_squad.py before the deadline;
                        contains only the squad hash (commitment phase)
  gw{NN}_result.json  — written by grade_model_gw.py after the GW finishes;
                        contains the full squad reveal, result, and display names

Disclosure rule:
  * No gw{NN}.json            -> 404 (never locked)
  * gw{NN}.json, no result    -> commitment only: hash + revealed flag.
                                  revealed=True when deadline has passed so the
                                  frontend can show "result pending" vs "pre-lock"
  * gw{NN}_result.json exists -> full squad, transfers, result (named)

No DB connection: the API is a reader of git-committed artifacts. Railway gets
the data via redeploy-on-push — the lock/grade pushes that already happen for
the public timestamp are the deployment trigger.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPORT_DIR = PROJECT_ROOT / "predictions" / "fpl"

router = APIRouter(prefix="/api/fpl", tags=["fpl"])


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_deadline(s: str) -> datetime:
    return datetime.fromisoformat(str(s).replace("Z", "+00:00"))


def _load_commitment(gw: int) -> dict | None:
    p = EXPORT_DIR / f"gw{gw:02d}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _load_result(gw: int) -> dict | None:
    p = EXPORT_DIR / f"gw{gw:02d}_result.json"
    return json.loads(p.read_text()) if p.exists() else None


@router.get("/model/gw/{gw}")
def model_gw(gw: int) -> dict:
    commitment = _load_commitment(gw)
    if commitment is None:
        raise HTTPException(404, f"GW{gw} has not been locked.")

    base = {
        "gameweek": gw,
        "locked_at_utc": commitment["locked_at_utc"],
        "deadline_utc": commitment["deadline_utc"],
        "squad_hash": commitment["squad_hash"],
    }

    result_data = _load_result(gw)
    if result_data is None:
        revealed = _now() >= _parse_deadline(commitment["deadline_utc"])
        return {**base, "revealed": revealed}

    # squad_display carries names from the grader; fall back to "#pid" if absent
    display = {d["player_id"]: d for d in result_data.get("squad_display", [])}

    def player(pid: int) -> dict:
        d = display.get(pid, {})
        return {
            "player_id": pid,
            "name": d.get("name") or f"#{pid}",
            "position": d.get("position"),
            "team": d.get("team"),
        }

    squad = [{
        **player(r["player_id"]),
        "is_xi": bool(r["is_xi"]),
        "is_captain": bool(r["is_captain"]),
        "is_vice": bool(r["is_vice"]),
        "bench_order": r.get("bench_order", 0),
    } for r in result_data["squad"]]

    raw_tr = result_data.get("transfers", {})
    tr_display = result_data.get("transfers_display", {})
    transfers = {
        "in": tr_display.get("in") or [player(p) for p in raw_tr.get("in", [])],
        "out": tr_display.get("out") or [player(p) for p in raw_tr.get("out", [])],
        "hits": raw_tr.get("hits", 0),
    }

    eff = result_data.get("effective_captain")
    eff_display = result_data.get("effective_captain_display") or (
        player(eff) if eff is not None else None
    )

    raw_subs = result_data.get("autosubs", [])
    autosubs = result_data.get("autosubs_display") or [
        {"out": player(s["out"]), "in": player(s["in"])} for s in raw_subs
    ]

    return {
        **base,
        "revealed": True,
        "squad": squad,
        "transfers": transfers,
        "expected_points": result_data.get("expected_points"),
        "result": {
            "gross_points": result_data["gross_points"],
            "hit_points": result_data["hit_points"],
            "net_points": result_data["net_points"],
            "effective_captain": eff_display,
            "autosubs": autosubs,
            "graded_at_utc": result_data["graded_at_utc"],
        },
    }


@router.get("/projections/{gw}")
def projections(gw: int) -> dict:
    """The model's per-player view for a gameweek, published at lock time.

    Deliberately available BEFORE the deadline, unlike the squad. This is the
    only endpoint that is useful while a manager is still deciding, which is
    when anybody actually cares; everything else here is retrospective. It
    reveals the model's opinion of every player, not which fifteen it bought,
    so the commitment scheme is untouched.
    """
    p = EXPORT_DIR / f"gw{gw:02d}_projections.json"
    if not p.exists():
        raise HTTPException(
            404, f"No projections published for GW{gw} yet. They appear when "
                 "the model locks its squad, about ten hours before the deadline.")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise HTTPException(502, f"projections file unreadable: {exc}") from exc


@router.get("/model/season")
def model_season() -> dict:
    gws: list[dict] = []
    model_total = avg_total = 0
    above = below = equal = 0

    result_files = sorted(EXPORT_DIR.glob("gw*_result.json")) if EXPORT_DIR.exists() else []
    for f in result_files:
        data = json.loads(f.read_text())
        gw_num = data.get("gameweek")
        if gw_num is None:
            continue
        net = int(data.get("net_points", 0))
        hits = int(data.get("hit_points", 0))
        avg = data.get("average_score")
        model_total += net
        if avg is not None:
            avg_total += avg
            if net > avg:
                above += 1
            elif net < avg:
                below += 1
            else:
                equal += 1
        gws.append({
            "gameweek": gw_num,
            "net_points": net,
            "hit_points": hits,
            "fpl_average": avg,
            "cumulative": model_total,
        })

    return {
        "gameweeks": gws,
        "model_total": model_total,
        "average_total": avg_total,
        "vs_average": {"above": above, "below": below, "equal": equal},
    }
