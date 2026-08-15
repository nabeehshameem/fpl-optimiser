"""
fpl_player_row.py

Shared helper for building the standard player-row dict used in both the
projections export (lock_model_squad.py) and the tools export
(build_gw_tools.py). Factored out so the two consumers cannot produce
differing formats for the same player.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


def load_player_meta(
    db_path: Path,
    gw: int,
) -> tuple[dict[int, dict], dict[int, int], dict[int, tuple[str, bool]]]:
    """
    Returns:
        meta     {player_id: {name, position (int), team, price (int tenths)}}
        team_of  {player_id: team_id}
        opponent {team_id: (opp_short_name, at_home)}
    """
    conn = sqlite3.connect(db_path)
    try:
        meta = {
            int(r[0]): {
                "name": r[1], "position": int(r[2]),
                "team": r[3], "price": int(r[4]),
            }
            for r in conn.execute(
                "SELECT p.player_id, p.web_name, p.position, t.short_name, "
                "p.current_cost FROM players p "
                "LEFT JOIN teams t ON t.team_id = p.team_id"
            )
        }
        team_of = {
            int(r[0]): int(r[1])
            for r in conn.execute("SELECT player_id, team_id FROM players")
        }
        opponent: dict[int, tuple[str, bool]] = {}
        try:
            rows = conn.execute(
                "SELECT f.home_team_id, f.away_team_id, th.short_name, "
                "ta.short_name FROM fixtures f "
                "JOIN teams th ON th.team_id = f.home_team_id "
                "JOIN teams ta ON ta.team_id = f.away_team_id "
                "WHERE f.gameweek_id = ?", (gw,)
            ).fetchall()
        except sqlite3.Error:
            rows = []
        for h, a, hn, an in rows:
            opponent[int(h)] = (an, True)
            opponent[int(a)] = (hn, False)
    finally:
        conn.close()

    return meta, team_of, opponent


def player_row(
    pid: int,
    pts: float,
    meta: dict[int, dict],
    team_of: dict[int, int],
    opponent: dict[int, tuple[str, bool]],
) -> dict:
    """Standard player-row dict for projections and tools exports."""
    m = meta.get(pid, {"name": f"#{pid}", "position": 0, "team": "?", "price": 0})
    opp_info = opponent.get(team_of.get(pid, -1))
    opp, home = opp_info if opp_info else (None, None)
    return {
        "player_id": pid,
        "name": m["name"],
        "team": m["team"],
        "position": POSITION_MAP.get(m["position"], "?"),
        "price": m["price"] / 10.0,
        "projected_points": round(float(pts), 2),
        "opponent": opp,
        "venue": None if home is None else ("H" if home else "A"),
    }
