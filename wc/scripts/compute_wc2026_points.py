"""
compute_wc2026_points.py
Compute WC2026 fantasy points from ingested match stats, events, and lineups.

Scoring system (WC Fantasy 2026 actual rules):
  Appearance <60 min         : 1 pt  (all positions)
  Appearance ≥60 min         : 2 pts (all positions)
  Assist                     : 3 pts
  Yellow card                : -1 pt
  Red card                   : -2 pts
  Own goal                   : -2 pts
  Winning a penalty          : +2 pts  [tracked via events where available]
  Conceding a penalty        : -1 pt
  Goal from direct free-kick : +1 bonus (in addition to goal pts)

  GK:  goal=9, CS(≥60)=5, first GC=0 / each additional=-1, pen_save=+3, per3_saves=+1
  DEF: goal=7, CS(≥60)=5, first GC=0 / each additional=-1
  MID: goal=6, CS(≥60)=1, per3_tackles=+1, per2_bcc=+1
  FWD: goal=5, per2_sot=+1

Run after ingest_match_stats.py:
  python wc/scripts/compute_wc2026_points.py

Outputs to wc2026_player_points table and prints a summary.
"""

import sqlite3
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from wc.scripts.init_db import DB_PATH, init_db

SCORING = {
    "GK":  {"app_60": 2, "app_u60": 1, "goal": 9, "assist": 3, "clean_sheet": 5,
             "yellow": -1, "red": -2, "penalty_save": 3, "save_per3": 1},
    "DEF": {"app_60": 2, "app_u60": 1, "goal": 7, "assist": 3, "clean_sheet": 5,
             "yellow": -1, "red": -2},
    "MID": {"app_60": 2, "app_u60": 1, "goal": 6, "assist": 3, "clean_sheet": 1,
             "yellow": -1, "red": -2, "tackle_per3": 1, "bcc_per2": 1},
    "FWD": {"app_60": 2, "app_u60": 1, "goal": 5, "assist": 3, "clean_sheet": 0,
             "yellow": -1, "red": -2, "sot_per2": 1},
}


def _pts(pos: str, minutes: int, goals: int, assists: int,
         yellow_cards: int, red_cards: int, own_goals: int,
         clean_sheet: bool, goals_conceded: int,
         saves: int, shots_on_target: int, tackles: int,
         big_chances_created: int, penalty_saves: int,
         penalty_conceded: int, penalty_won: int) -> float:
    s = SCORING.get(pos, SCORING["MID"])
    if minutes == 0:
        return 0.0

    pts = s["app_60"] if minutes >= 60 else s["app_u60"]
    pts += goals * s["goal"]
    pts += assists * s["assist"]
    pts += own_goals * -2
    pts += yellow_cards * s["yellow"]
    pts += red_cards * s["red"]
    pts += penalty_conceded * -1
    pts += penalty_won * 2

    if clean_sheet and minutes >= 60:
        pts += s["clean_sheet"]

    # GC: first goal = 0, each additional = -1
    if pos in ("GK", "DEF") and minutes >= 60 and goals_conceded >= 2:
        pts += -(goals_conceded - 1)

    # Position-specific bonus stats
    if pos == "GK":
        pts += (saves // 3) * s["save_per3"]
        pts += penalty_saves * s["penalty_save"]
    elif pos == "MID":
        pts += (tackles // 3) * s["tackle_per3"]
        pts += (big_chances_created // 2) * s["bcc_per2"]
    elif pos == "FWD":
        pts += (shots_on_target // 2) * s["sot_per2"]

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

    # Build player map from lineups (now includes bonus stat columns)
    players: dict[str, dict] = {}
    for row in conn.execute(
        """SELECT player_name, team_name, position, minutes_played,
                  saves, shots_on_target, tackles, big_chances_created,
                  penalty_saves, penalty_conceded, penalty_won
           FROM match_lineups WHERE api_fixture_id = ?""",
        (fixture_id,),
    ).fetchall():
        (player_name, team, pos, minutes,
         saves, sot, tackles, bcc, pen_saves, pen_conceded, pen_won) = row
        players[player_name] = {
            "team": team, "pos": pos or "MID", "minutes": minutes or 0,
            "goals": 0, "assists": 0, "yellow_cards": 0, "red_cards": 0,
            "own_goals": 0,
            "saves": saves or 0, "shots_on_target": sot or 0,
            "tackles": tackles or 0, "big_chances_created": bcc or 0,
            "penalty_saves": pen_saves or 0, "penalty_conceded": pen_conceded or 0,
            "penalty_won": pen_won or 0,
        }

    # Accumulate events — distinguish own goals from regular goals
    for player_name, event_type, event_detail in conn.execute(
        "SELECT player_name, event_type, event_detail FROM match_events WHERE api_fixture_id = ?",
        (fixture_id,),
    ).fetchall():
        if player_name not in players:
            continue
        p = players[player_name]
        if event_type == "goal":
            if event_detail == "ownGoal":
                p["own_goals"] += 1
            else:
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
        fantasy_pts = _pts(
            p["pos"], p["minutes"], p["goals"], p["assists"],
            p["yellow_cards"], p["red_cards"], p["own_goals"],
            cs, gc,
            p["saves"], p["shots_on_target"], p["tackles"],
            p["big_chances_created"], p["penalty_saves"], p["penalty_conceded"],
            p["penalty_won"],
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO wc2026_player_points
              (api_fixture_id, match_date, team_name, player_name,
               position, minutes, goals, assists,
               yellow_cards, red_cards, own_goals, clean_sheet, goals_conceded,
               saves, shots_on_target, tackles, big_chances_created,
               penalty_saves, penalty_conceded, penalty_won, fantasy_pts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (fixture_id, match_date, team, player_name,
             p["pos"], p["minutes"], p["goals"], p["assists"],
             p["yellow_cards"], p["red_cards"], p["own_goals"], int(cs), gc,
             p["saves"], p["shots_on_target"], p["tackles"], p["big_chances_created"],
             p["penalty_saves"], p["penalty_conceded"], p["penalty_won"], fantasy_pts),
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
