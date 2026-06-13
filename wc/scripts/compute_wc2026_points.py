"""
compute_wc2026_points.py
Compute WC2026 fantasy points from ingested match stats, events, and lineups.

Scoring system (WC Fantasy 2022 rules):
  Appearance ≥60 min        : 2 pts  (all positions)
  Appearance  <60 min       : 1 pt
  Goal (FWD)                : 4 pts
  Goal (MID)                : 5 pts
  Goal (GK/DEF)             : 6 pts
  Assist                    : 3 pts
  Clean sheet ≥60 min (GK/DEF): 4 pts
  Clean sheet ≥60 min (MID) : 1 pt
  Yellow card               : -1 pt
  Red card                  : -3 pts
  Every 2 goals conceded (GK/DEF, ≥60 min): -1 pt

Run after ingest_match_stats.py:
  python wc/scripts/compute_wc2026_points.py

Outputs to wc2026_player_points table and prints a summary.
"""

import sqlite3
import sys
from pathlib import Path

# Windows terminal may not handle non-ASCII names
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from wc.scripts.init_db import DB_PATH, init_db

SCORING = {
    "GK":  {"app_60": 2, "app_u60": 1, "goal": 6, "assist": 3, "clean_sheet": 4, "yellow": -1, "red": -3, "per2_conceded": -1},
    "DEF": {"app_60": 2, "app_u60": 1, "goal": 6, "assist": 3, "clean_sheet": 4, "yellow": -1, "red": -3, "per2_conceded": -1},
    "MID": {"app_60": 2, "app_u60": 1, "goal": 5, "assist": 3, "clean_sheet": 1, "yellow": -1, "red": -3, "per2_conceded": 0},
    "FWD": {"app_60": 2, "app_u60": 1, "goal": 4, "assist": 3, "clean_sheet": 0, "yellow": -1, "red": -3, "per2_conceded": 0},
}


def _pts(pos: str, minutes: int, goals: int, assists: int,
         yellow_cards: int, red_cards: int, clean_sheet: bool,
         goals_conceded: int) -> float:
    s = SCORING.get(pos, SCORING["MID"])
    pts = 0.0
    if minutes >= 60:
        pts += s["app_60"]
    elif minutes > 0:
        pts += s["app_u60"]
    else:
        return 0.0  # unused sub
    pts += goals * s["goal"]
    pts += assists * s["assist"]
    if clean_sheet and minutes >= 60:
        pts += s["clean_sheet"]
    pts += yellow_cards * s["yellow"]
    pts += red_cards * s["red"]
    if s["per2_conceded"] and minutes >= 60:
        pts += (goals_conceded // 2) * s["per2_conceded"]
    return pts


def _compute_fixture(conn: sqlite3.Connection, fixture_id: int) -> int:
    stat = conn.execute(
        "SELECT match_date, home_team, away_team, home_score, away_score "
        "FROM match_stats WHERE api_fixture_id = ?",
        (fixture_id,),
    ).fetchone()
    if not stat:
        return 0
    match_date, home_team, away_team, home_score, away_score = stat

    conceded = {home_team: away_score or 0, away_team: home_score or 0}

    # Build player map from lineups
    players: dict[str, dict] = {}
    for player_name, team, pos, minutes, _ in conn.execute(
        "SELECT player_name, team_name, position, minutes_played, is_starter "
        "FROM match_lineups WHERE api_fixture_id = ?",
        (fixture_id,),
    ).fetchall():
        players[player_name] = {
            "team": team, "pos": pos or "MID", "minutes": minutes or 0,
            "goals": 0, "assists": 0, "yellow_cards": 0, "red_cards": 0,
        }

    # Accumulate events
    for player_name, event_type in conn.execute(
        "SELECT player_name, event_type FROM match_events WHERE api_fixture_id = ?",
        (fixture_id,),
    ).fetchall():
        if player_name not in players:
            continue
        p = players[player_name]
        if event_type == "goal":
            p["goals"] += 1
        elif event_type == "assist":
            p["assists"] += 1
        elif event_type == "yellow_card":
            p["yellow_cards"] += 1
        elif event_type == "red_card":
            p["red_cards"] += 1

    conn.execute("DELETE FROM wc2026_player_points WHERE api_fixture_id = ?", (fixture_id,))

    count = 0
    for player_name, p in players.items():
        if p["minutes"] == 0:
            continue
        team = p["team"]
        gc = conceded.get(team, 0)
        cs = gc == 0
        fantasy_pts = _pts(p["pos"], p["minutes"], p["goals"], p["assists"],
                           p["yellow_cards"], p["red_cards"], cs, gc)
        conn.execute(
            """
            INSERT OR REPLACE INTO wc2026_player_points
              (api_fixture_id, match_date, team_name, player_name,
               position, minutes, goals, assists,
               yellow_cards, red_cards, clean_sheet, goals_conceded, fantasy_pts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (fixture_id, match_date, team, player_name,
             p["pos"], p["minutes"], p["goals"], p["assists"],
             p["yellow_cards"], p["red_cards"], int(cs), gc, fantasy_pts),
        )
        count += 1

    conn.commit()
    return count


def compute() -> None:
    init_db()
    conn = sqlite3.connect(DB_PATH)

    fixture_ids = [r[0] for r in conn.execute("SELECT api_fixture_id FROM match_stats").fetchall()]
    if not fixture_ids:
        print("No fixtures in match_stats. Run ingest_match_stats.py first.")
        conn.close()
        return

    total_players = 0
    for fid in fixture_ids:
        n = _compute_fixture(conn, fid)
        row = conn.execute(
            "SELECT home_team, away_team, home_score, away_score, match_date "
            "FROM match_stats WHERE api_fixture_id = ?",
            (fid,),
        ).fetchone()
        if row and n:
            home, away, hs, as_, date = row
            print(f"  {home} {hs}-{as_} {away} ({date}): {n} players")
        total_players += n

    conn.close()

    # Print top scorers summary
    conn = sqlite3.connect(DB_PATH)
    print("\nTop fantasy performers:")
    print(f"  {'Player':25s}  {'Team':22s}  {'Pos':3s}  {'Pts':>5}  {'G':>2}  {'A':>2}  {'Min':>3}")
    print("  " + "-" * 70)
    for row in conn.execute(
        """
        SELECT player_name, team_name, position,
               SUM(fantasy_pts), SUM(goals), SUM(assists), SUM(minutes)
        FROM wc2026_player_points
        GROUP BY player_name, team_name
        ORDER BY SUM(fantasy_pts) DESC
        LIMIT 20
        """
    ).fetchall():
        name, team, pos, pts, g, a, mins = row
        print(f"  {name:25s}  {team:22s}  {pos:3s}  {pts:5.1f}  {g:2d}  {a:2d}  {mins:3d}")
    conn.close()
    print(f"\nTotal: {total_players} player-match records written to wc2026_player_points.")


if __name__ == "__main__":
    compute()
