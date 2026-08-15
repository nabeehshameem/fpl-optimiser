"""
build_gw_tools.py

Export predictions/fpl/gwNN_tools.json: four pre-computed interactive FPL tools.

  python scripts/build_gw_tools.py          # next GW
  python scripts/build_gw_tools.py --gw 3   # specific GW
  python scripts/build_gw_tools.py --dry-run

Tools:
  optimal_xi     — highest-projected legal XI from a fresh £100m squad
  captain        — top 8 captain picks with projected delta (the doubling gain)
  gw_predictions — DC scoreline/outcome for every GW fixture
  fixture_ticker — GW window expected-goals grid

All are pre-computed from committed artifacts. The UI reveals a result; it
never computes one. Rule 8: Railway has no DB — everything is done here and
served from the committed JSON.

Rule 4 (claims layer): the note in optimal_xi explicitly distinguishes this
TOOL squad from the model's actual squad (which lives in gwNN.json). They
will differ once the model carries transfer state.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.availability import get_excluded_ids  # noqa: E402
from src.dc_match import predict_match  # noqa: E402
from src.fpl_player_row import load_player_meta, player_row, POSITION_MAP  # noqa: E402
from src.optimiser import SquadOptimiser  # noqa: E402
from scripts.build_fixture_grid import compute_ticker_data  # noqa: E402

DB_PATH = Path(os.getenv("FPL_DB_PATH", PROJECT_ROOT / "data" / "fpl.db"))
DC_PARAMS = PROJECT_ROOT / "models" / "fpl_dc_params.json"
EXPORT_DIR = Path(os.getenv("FPL_EXPORT_DIR", PROJECT_ROOT / "predictions" / "fpl"))

MODEL_NAME = "dc_projection_v1"
CAPTAIN_TOP_N = 8
TICKER_GW_WINDOW = 6


# ── DB helpers ────────────────────────────────────────────────────────────────

def _next_gameweek(conn) -> tuple[int, str]:
    row = conn.execute(
        "SELECT gameweek_id, deadline_time FROM gameweeks WHERE is_next = 1 LIMIT 1"
    ).fetchone()
    if not row:
        row = conn.execute(
            "SELECT gameweek_id, deadline_time FROM gameweeks "
            "WHERE finished = 0 ORDER BY gameweek_id LIMIT 1"
        ).fetchone()
    if not row:
        raise RuntimeError("No upcoming gameweek in DB — run ingest_bootstrap first.")
    return int(row[0]), str(row[1])


def _load_predictions(conn, gw: int):
    """Latest prediction per player for gw, with eligibility features.
    Mirrors lock_model_squad.load_predictions() so the tools are built from
    the identical prediction set as the squad.
    """
    import pandas as pd
    df = pd.read_sql_query(
        """
        SELECT p.player_id, p.predicted_points
        FROM predictions p
        JOIN (
            SELECT player_id, MAX(prediction_time) AS mt
            FROM predictions WHERE gameweek_id = ? AND model_name = ?
            GROUP BY player_id
        ) latest ON latest.player_id = p.player_id AND latest.mt = p.prediction_time
        WHERE p.gameweek_id = ? AND p.model_name = ?
        """,
        conn, params=(gw, MODEL_NAME, gw, MODEL_NAME),
    )
    if df.empty:
        raise RuntimeError(
            f"No {MODEL_NAME} predictions for GW{gw}. "
            "Run scripts/run_predictions.py first."
        )
    hist = pd.read_sql_query(
        """
        SELECT player_id,
               SUM(CASE WHEN minutes >= 60 THEN 1 ELSE 0 END) AS q5,
               SUM(CASE WHEN minutes >= 60 AND recent3 = 1 THEN 1 ELSE 0 END) AS q3
        FROM (
            SELECT h.player_id, h.minutes,
                   CASE WHEN h.gameweek_id > ? - 3 THEN 1 ELSE 0 END AS recent3
            FROM player_gameweek_history h
            WHERE h.gameweek_id > ? - 5
        )
        GROUP BY player_id
        """,
        conn, params=(gw, gw),
    )
    df = df.merge(hist, on="player_id", how="left")
    df["qualifying_games_3"] = df["q3"].fillna(0)
    df["qualifying_games_5"] = df["q5"].fillna(0)
    df["chance_of_playing_next"] = 100

    max_hist_gw = conn.execute(
        "SELECT COALESCE(MAX(gameweek_id), 0) FROM player_gameweek_history"
    ).fetchone()[0]
    if max_hist_gw < 3:
        df["qualifying_games_3"] = 3
        df["qualifying_games_5"] = 5
    return df


# ── tool builders ─────────────────────────────────────────────────────────────

def _build_optimal_xi(
    preds_df, gw: int,
    meta: dict, team_of: dict, opponent: dict,
    db_path: Path = DB_PATH,
) -> dict:
    """Run MILP on the full prediction pool with no transfer state."""
    opt = SquadOptimiser(db_path=db_path)
    result = opt.optimise(preds_df)

    xi = result["starting_xi"]
    cap_pid = int(result["captain"]["player_id"])

    pos_order = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
    xi_rows = []
    for r in xi.itertuples():
        pid = int(r.player_id)
        m = meta.get(pid, {"name": f"#{pid}", "position": 0, "team": "?", "price": 0})
        pos = POSITION_MAP.get(m["position"], "?")
        xi_rows.append({
            "player_id": pid,
            "name": m["name"],
            "team": m["team"],
            "position": pos,
            "price": m["price"] / 10.0,
            "projected_points": round(float(r.predicted_points), 2),
            "is_captain": pid == cap_pid,
        })
    xi_rows.sort(key=lambda p: (pos_order.get(p["position"], 9), -p["projected_points"]))

    counts = {"GK": 0, "DEF": 0, "MID": 0, "FWD": 0}
    for p in xi_rows:
        counts[p["position"]] = counts.get(p["position"], 0) + 1
    formation = f"{counts['DEF']}-{counts['MID']}-{counts['FWD']}"

    cap_pts = round(float(result["captain"]["predicted_points"]), 2)
    total = round(sum(p["projected_points"] for p in xi_rows) + cap_pts, 2)

    return {
        "formation": formation,
        "players": xi_rows,
        "total_projected": total,
        "note": (
            f"Highest-projected legal XI from a fresh £100.0m squad, ignoring "
            f"the model's own transfer state. This is a TOOL, not the model's "
            f"team — the model's actual squad is at /model/gw/{gw}."
        ),
    }


def _build_captain_list(
    preds_df,
    meta: dict, team_of: dict, opponent: dict,
    top_n: int = CAPTAIN_TOP_N,
) -> list[dict]:
    """Top N players by projected points, from those with a fixture."""
    with_fixture = {pid for pid, tid in team_of.items() if tid in opponent}
    ranked = sorted(
        (
            (int(r.player_id), float(r.predicted_points))
            for r in preds_df.itertuples()
            if int(r.player_id) in with_fixture
        ),
        key=lambda t: t[1], reverse=True,
    )[:top_n]

    return [
        {
            **player_row(pid, pts, meta, team_of, opponent),
            "captain_delta": round(pts, 2),
        }
        for pid, pts in ranked
    ]


def _build_gw_predictions(gw: int, dc: dict, db_path: Path) -> list[dict]:
    """DC scoreline predictions for every fixture in the GW."""
    conn = sqlite3.connect(db_path)
    try:
        live_snames = {
            int(r[0]): r[1]
            for r in conn.execute("SELECT team_id, short_name FROM teams")
        }
        rows = conn.execute(
            "SELECT home_team_id, away_team_id, kickoff_time "
            "FROM fixtures WHERE gameweek_id = ? ORDER BY kickoff_time",
            (gw,),
        ).fetchall()
    finally:
        conn.close()

    results = []
    for home_id, away_id, kickoff in rows:
        h_sn = live_snames.get(int(home_id), str(home_id))
        a_sn = live_snames.get(int(away_id), str(away_id))
        pred = predict_match(h_sn, a_sn, dc)
        results.append({
            "home": h_sn,
            "away": a_sn,
            "kickoff_utc": str(kickoff) if kickoff else None,
            "p_home": pred["p_home"],
            "p_draw": pred["p_draw"],
            "p_away": pred["p_away"],
            "top_scoreline": pred["top_scoreline"],
            "xg_home": pred["xg_home"],
            "xg_away": pred["xg_away"],
        })
    return results


def _build_fixture_ticker(from_gw: int, to_gw: int, dc: dict, db_path: Path) -> dict:
    """GW fixture grid, sharing compute_ticker_data() with build_fixture_grid.py."""
    data = compute_ticker_data(from_gw, to_gw, db_path, dc=dc)
    teams = data["teams"]
    grid = data["grid"]

    team_rows = []
    for tid, cells in grid.items():
        if not cells:
            continue
        cell_list = [
            {
                "gw": gw,
                "opponent": cells[gw][0],
                "venue": "H" if cells[gw][1] else "A",
                "xg_for": round(cells[gw][2], 2),
            }
            for gw in range(from_gw, to_gw + 1)
            if gw in cells
        ]
        # total_xg is the sum of the already-rounded cells so the JSON is
        # internally consistent: sum(cells[i].xg_for) == total_xg exactly.
        team_rows.append({
            "team": teams[tid],
            "cells": cell_list,
            "total_xg": round(sum(c["xg_for"] for c in cell_list), 2),
        })
    team_list = sorted(team_rows, key=lambda t: t["total_xg"], reverse=True)

    return {
        "from_gw": from_gw,
        "to_gw": to_gw,
        "teams": team_list,
    }


# ── main export ───────────────────────────────────────────────────────────────

def build(gw: int | None = None, db_path: Path = DB_PATH,
          dc_path: Path = DC_PARAMS) -> dict:
    """Build and return the tools export dict. Raises on hard failures."""
    if not dc_path.exists():
        raise RuntimeError(f"DC params not found: {dc_path} — run train_dc.py first")
    dc = json.loads(dc_path.read_text(encoding="utf-8"))
    if not dc.get("team_params"):
        raise RuntimeError("DC params contain no team_params")

    conn = sqlite3.connect(db_path)
    try:
        if gw is None:
            gw, deadline = _next_gameweek(conn)
        else:
            row = conn.execute(
                "SELECT deadline_time FROM gameweeks WHERE gameweek_id = ?", (gw,)
            ).fetchone()
            deadline = str(row[0]) if row else "unknown"
        preds_df = _load_predictions(conn, gw)
    finally:
        conn.close()

    # Apply the same exclusions as the lock so the tool pool never presents
    # confirmed non-starters. Non-fatal: a stale exclusions file degrades to
    # a warning rather than blocking the tools build (rule 5).
    try:
        excluded = get_excluded_ids(db_path, gw)
        if excluded:
            preds_df = preds_df[~preds_df["player_id"].isin(excluded)]
    except Exception as exc:
        print(f"[WARN] exclusions check skipped — running without: {exc}",
              file=sys.stderr)

    meta, team_of, opponent = load_player_meta(db_path, gw)

    optimal_xi = _build_optimal_xi(preds_df, gw, meta, team_of, opponent, db_path=db_path)
    captain = _build_captain_list(preds_df, meta, team_of, opponent)
    gw_predictions = _build_gw_predictions(gw, dc, db_path)
    ticker = _build_fixture_ticker(gw, gw + TICKER_GW_WINDOW - 1, dc, db_path)

    return {
        "gameweek": gw,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": MODEL_NAME,
        "optimal_xi": optimal_xi,
        "captain": captain,
        "gw_predictions": gw_predictions,
        "fixture_ticker": ticker,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gw", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tools = build(args.gw)
    gw = tools["gameweek"]

    if args.dry_run:
        print(json.dumps(tools, indent=2))
        return

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = EXPORT_DIR / f"gw{gw:02d}_tools.json"
    out.write_text(json.dumps(tools, indent=2))
    print(f"GW{gw} tools exported → {out}")


if __name__ == "__main__":
    main()
