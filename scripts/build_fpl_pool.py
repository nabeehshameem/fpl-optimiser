"""
build_fpl_pool.py

Export predictions/fpl/gwNN_pool.json: the full player pool for the Squad Builder.

  python scripts/build_fpl_pool.py          # next GW
  python scripts/build_fpl_pool.py --gw 1   # specific GW
  python scripts/build_fpl_pool.py --dry-run

Output shape:
  {
    "gameweek": 1,
    "generated_at_utc": "...",
    "budget": 100.0,
    "rules": {
      "squad_size": 15, "xi_size": 11, "max_per_club": 3, "budget": 100.0,
      "positions": {
        "GK":  {"squad": 2, "xi_min": 1, "xi_max": 1},
        "DEF": {"squad": 5, "xi_min": 3, "xi_max": 5},
        "MID": {"squad": 5, "xi_min": 2, "xi_max": 5},
        "FWD": {"squad": 3, "xi_min": 1, "xi_max": 3}
      }
    },
    "players": [
      {
        "player_id": 411,
        "name": "Haaland",
        "team": "MCI",
        "team_id": 11,
        "position": "FWD",
        "price": 15.5,
        "ownership_pct": 62.3,
        "available": true,
        "availability_note": null,
        "projected_points": 8.53,
        "projected": true,
        "opponent": "BOU",
        "venue": "H"
      },
      ...players without projections have projected_points: null, projected: false
    ]
  }

Rule 8: Railway has no DB. This runs locally, commits the JSON, and Railway
serves it from predictions/fpl/. The Squad Builder reads this and enforces
all constraints client-side.

Rules are encoded in the JSON so a future FPL rules change only requires
updating this script, not the frontend.
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

DB_PATH = Path(os.getenv("FPL_DB_PATH", PROJECT_ROOT / "data" / "fpl.db"))
EXPORT_DIR = Path(os.getenv("FPL_EXPORT_DIR", PROJECT_ROOT / "predictions" / "fpl"))

POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

FPL_RULES = {
    "squad_size": 15,
    "xi_size": 11,
    "max_per_club": 3,
    "budget": 100.0,
    "positions": {
        "GK":  {"squad": 2, "xi_min": 1, "xi_max": 1},
        "DEF": {"squad": 5, "xi_min": 3, "xi_max": 5},
        "MID": {"squad": 5, "xi_min": 2, "xi_max": 5},
        "FWD": {"squad": 3, "xi_min": 1, "xi_max": 3},
    },
}


def _next_gameweek(conn) -> tuple[int, str]:
    row = conn.execute(
        "SELECT gameweek_id, deadline_time FROM gameweeks WHERE is_next = 1 LIMIT 1"
    ).fetchone()
    if not row:
        row = conn.execute(
            "SELECT gameweek_id, deadline_time FROM gameweeks ORDER BY gameweek_id LIMIT 1"
        ).fetchone()
    if not row:
        raise RuntimeError("No gameweeks in DB — run ingest_bootstrap.py first")
    return int(row[0]), str(row[1])


def _load_projections(gw: int) -> dict[int, dict]:
    """Return {player_id: {projected_points, opponent, venue}} from the projections file."""
    p = EXPORT_DIR / f"gw{gw:02d}_projections.json"
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    out: dict[int, dict] = {}
    for players in data.get("by_position", {}).values():
        for pl in players:
            out[pl["player_id"]] = {
                "projected_points": pl.get("projected_points"),
                "opponent": pl.get("opponent"),
                "venue": pl.get("venue"),
            }
    return out


def build_pool(gw: int | None = None, db_path: Path = DB_PATH,
               export_dir: Path = EXPORT_DIR, dry_run: bool = False) -> dict:
    conn = sqlite3.connect(db_path)
    try:
        if gw is None:
            gw, deadline = _next_gameweek(conn)
        else:
            row = conn.execute(
                "SELECT deadline_time FROM gameweeks WHERE gameweek_id = ?", (gw,)
            ).fetchone()
            deadline = str(row[0]) if row else "unknown"

        # All players with team + latest snapshot data
        rows = conn.execute("""
            SELECT
                p.player_id,
                p.web_name,
                p.position,
                t.short_name   AS team,
                p.team_id,
                p.current_cost,
                COALESCE(s.selected_by_percent, 0.0) AS ownership_pct,
                s.chance_of_playing_next,
                s.news
            FROM players p
            LEFT JOIN teams t ON t.team_id = p.team_id
            LEFT JOIN player_snapshots s
                ON s.player_id = p.player_id
                AND s.snapshot_id = (
                    SELECT MAX(snapshot_id) FROM player_snapshots
                    WHERE player_id = p.player_id
                )
            ORDER BY p.player_id
        """).fetchall()

        # Opponent data for GW from fixtures
        opponent: dict[int, tuple[str, bool]] = {}
        fix_rows = conn.execute(
            "SELECT f.home_team_id, f.away_team_id, th.short_name, ta.short_name "
            "FROM fixtures f "
            "JOIN teams th ON th.team_id = f.home_team_id "
            "JOIN teams ta ON ta.team_id = f.away_team_id "
            "WHERE f.gameweek_id = ?", (gw,)
        ).fetchall()
        for h, a, hn, an in fix_rows:
            opponent[int(h)] = (an, True)
            opponent[int(a)] = (hn, False)

    finally:
        conn.close()

    projections = _load_projections(gw)

    players = []
    for (player_id, name, pos_int, team, team_id, cost,
         ownership_pct, chance, news) in rows:
        position = POSITION_MAP.get(int(pos_int), "?")
        price = round(int(cost) / 10, 1)

        proj = projections.get(int(player_id))
        has_proj = proj is not None
        projected_points = proj["projected_points"] if has_proj else None

        opp_data = proj if has_proj else None
        if opp_data and opp_data.get("opponent"):
            opp, venue = opp_data["opponent"], opp_data["venue"]
        else:
            opp_tuple = opponent.get(int(team_id) if team_id else -1)
            opp = opp_tuple[0] if opp_tuple else None
            venue = "H" if (opp_tuple and opp_tuple[1]) else ("A" if opp_tuple else None)

        # availability: unavailable if chance_of_playing_next < 50 (same threshold as lock)
        available = True
        availability_note = None
        if chance is not None and int(chance) < 50:
            available = False
            availability_note = news or f"{int(chance)}% chance"
        elif news:
            availability_note = news

        players.append({
            "player_id": int(player_id),
            "name": name,
            "team": team,
            "team_id": int(team_id) if team_id else None,
            "position": position,
            "price": price,
            "ownership_pct": round(float(ownership_pct), 1),
            "available": available,
            "availability_note": availability_note,
            "projected_points": round(projected_points, 2) if projected_points is not None else None,
            "projected": has_proj,
            "opponent": opp,
            "venue": venue,
        })

    pool = {
        "gameweek": gw,
        "deadline_utc": deadline,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rules": FPL_RULES,
        "players": players,
    }

    if dry_run:
        print(json.dumps(pool, indent=2)[:2000] + "\n...(truncated)")
        print(f"\n{len(players)} players in pool, "
              f"{sum(1 for p in players if p['projected'])} with projections")
        return pool

    export_dir.mkdir(parents=True, exist_ok=True)
    out = export_dir / f"gw{gw:02d}_pool.json"
    out.write_text(json.dumps(pool, indent=2), encoding="utf-8")
    print(f"GW{gw} pool exported -> {out} "
          f"({len(players)} players, "
          f"{sum(1 for p in players if p['projected'])} projected)")
    return pool


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gw", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    build_pool(gw=args.gw, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
