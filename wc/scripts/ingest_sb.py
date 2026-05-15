"""
ingest_sb.py
Load StatsBomb open data (WC2022 + WC2018) into the local DB.

StatsBomb open data: https://github.com/statsbomb/open-data
No API key needed — data is public on GitHub.
Install: pip install statsbombpy

Competition IDs (StatsBomb):
  43  = FIFA World Cup
  Season IDs:
    3   = 2018 World Cup
    106 = 2022 World Cup

Usage:
  python wc/scripts/ingest_sb.py
  python wc/scripts/ingest_sb.py --tournaments WC2022
  python wc/scripts/ingest_sb.py --tournaments WC2018,WC2022
"""

import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from wc.scripts.init_db import init_db, DB_PATH

try:
    from statsbombpy import sb as _sb
except ImportError:
    print("[ERROR] statsbombpy not installed. Run: pip install statsbombpy")
    sys.exit(1)

COMPETITIONS = {
    "WC2022": (43, 106),
    "WC2018": (43, 3),
}

# StatsBomb position name → fantasy position
POSITION_MAP = {
    "Goalkeeper":               "GK",
    "Center Back":              "DEF",  "Left Back":               "DEF",
    "Right Back":               "DEF",  "Left Center Back":        "DEF",
    "Right Center Back":        "DEF",  "Left Wing Back":          "DEF",
    "Right Wing Back":          "DEF",
    "Left Midfield":            "MID",  "Right Midfield":          "MID",
    "Center Midfield":          "MID",  "Left Center Midfield":    "MID",
    "Right Center Midfield":    "MID",  "Left Defensive Midfield": "MID",
    "Right Defensive Midfield": "MID",  "Center Defensive Midfield":"MID",
    "Center Attacking Midfield":"MID",  "Left Attacking Midfield": "MID",
    "Right Attacking Midfield": "MID",
    "Left Forward":             "FWD",  "Right Forward":           "FWD",
    "Center Forward":           "FWD",  "Secondary Striker":       "FWD",
    "Left Wing":                "MID",  "Right Wing":              "MID",
    "Attacking Midfield":       "MID",  "Defensive Midfield":      "MID",
}

# Fantasy scoring (FIFA WCF 2022 approximation — update for 2026 once announced)
SCORING = {
    "min_60": 2, "min_any": 1,
    "goal": {"GK": 10, "DEF": 10, "MID": 9, "FWD": 8},
    "assist": 5,
    "cs_full_gkdef": 4,   # GK/DEF clean sheet, 90+ min
    "cs_half_gkdef": 2,   # GK/DEF clean sheet, 45-89 min
    "cs_mid": 1,          # MID clean sheet, 60+ min
    "save_3": 1,          # GK: per 3 saves
    "yellow": -1,
    "red": -3,
}


# ---------- Helpers ----------

def _d(cell, *keys):
    """Safely navigate nested dicts from a DataFrame cell."""
    val = cell
    for k in keys:
        if isinstance(val, dict):
            val = val.get(k)
        else:
            return None
    return val


def _col(series, *keys):
    """Apply _d to a whole Series."""
    return series.apply(lambda x: _d(x, *keys))


def _norm(name: str) -> str:
    return name.lower().strip() if name else ""


def _parse_lineup_minutes(positions) -> int:
    """
    Compute total minutes played from StatsBomb lineup positions list.
    Each entry has 'from' and 'to' as "MM:SS" or "MM:SS.ms".
    """
    if not positions or not isinstance(positions, list):
        return 0
    total = 0.0
    for p in positions:
        try:
            def to_min(t):
                if not t:
                    return 0.0
                parts = str(t).split(":")
                if len(parts) == 2:
                    m, s = parts
                    return int(m) + float(s.split(".")[0]) / 60
                return 0.0

            start = to_min(p.get("from", "0:00"))
            end   = to_min(p.get("to",   "0:00"))
            total += max(0.0, end - start)
        except Exception:
            pass
    return min(int(round(total)), 95)


# ---------- Core extraction ----------

def compute_match_stats(match_id: int, match_row: dict) -> pd.DataFrame:
    """
    Parse events + lineups for one match and return a DataFrame of per-player stats.

    Returns columns:
      player_id, player_name, team_id, position, minutes,
      goals, assists, xg, xa, shots, key_passes,
      yellow_cards, red_cards, saves, clean_sheet, fantasy_pts
    """
    try:
        events   = _sb.events(match_id=match_id)
        lineups  = _sb.lineups(match_id=match_id)
    except Exception as e:
        print(f"    [warn] match {match_id}: {e}")
        return pd.DataFrame()

    if events is None or events.empty:
        return pd.DataFrame()

    # ── Normalise key event columns ────────────────────────────────────────
    events["_type"]       = _col(events["type"],   "name")
    events["_player_id"]  = _col(events["player"], "id")
    events["_team_id"]    = _col(events["team"],   "id")

    if "shot" in events.columns:
        events["_shot_outcome"] = _col(events["shot"], "outcome", "name")
        events["_shot_xg"]      = _col(events["shot"], "statsbomb_xg")
    else:
        events["_shot_outcome"] = None
        events["_shot_xg"]      = None

    if "pass" in events.columns:
        events["_pass_goal_assist"] = _col(events["pass"], "goal_assist")
        events["_pass_shot_assist"] = _col(events["pass"], "shot_assist")
        events["_pass_xa"]          = _col(events["pass"], "xa")
    else:
        events["_pass_goal_assist"] = None
        events["_pass_shot_assist"] = None
        events["_pass_xa"]          = None

    fc_card = _col(events.get("foul_committed", pd.Series([None] * len(events))), "card", "name") \
              if "foul_committed" in events.columns else pd.Series([None] * len(events))
    bb_card = _col(events.get("bad_behaviour", pd.Series([None] * len(events))), "card", "name") \
              if "bad_behaviour" in events.columns else pd.Series([None] * len(events))
    events["_card"] = fc_card.combine_first(bb_card)

    if "goalkeeper" in events.columns:
        events["_gk_outcome"] = _col(events["goalkeeper"], "outcome", "name")
    else:
        events["_gk_outcome"] = None

    # ── Event aggregations ─────────────────────────────────────────────────
    shots_df = events[events["_type"] == "Shot"]
    goals      = shots_df[shots_df["_shot_outcome"] == "Goal"].groupby("_player_id").size().rename("goals")
    xg         = shots_df.groupby("_player_id")["_shot_xg"].sum().rename("xg")
    shots_cnt  = shots_df.groupby("_player_id").size().rename("shots")

    passes_df = events[events["_type"] == "Pass"]
    assists    = passes_df[passes_df["_pass_goal_assist"] == True].groupby("_player_id").size().rename("assists")
    key_passes = passes_df[passes_df["_pass_shot_assist"] == True].groupby("_player_id").size().rename("key_passes")
    xa         = passes_df.groupby("_player_id")["_pass_xa"].sum().rename("xa")

    yellows = events[events["_card"] == "Yellow Card"].groupby("_player_id").size().rename("yellow_cards")
    reds    = events[events["_card"].isin(["Red Card", "Second Yellow"])].groupby("_player_id").size().rename("red_cards")

    gk_saves = events[
        (events["_type"] == "Goal Keeper") &
        events["_gk_outcome"].isin(["Saved", "Saved to Post", "Saved Twice"])
    ].groupby("_player_id").size().rename("saves")

    # ── Build player rows from lineups ─────────────────────────────────────
    home_team_id = match_row.get("home_team_id")
    away_team_id = match_row.get("away_team_id")
    home_score   = match_row.get("home_score", 0) or 0
    away_score   = match_row.get("away_score", 0) or 0

    player_rows = []
    for team_name, lineup_df in lineups.items():
        if lineup_df is None or lineup_df.empty:
            continue
        # Determine team_id from events
        team_events = events[_col(events["team"], "name") == team_name]
        team_id = int(_col(team_events["team"], "id").dropna().iloc[0]) \
                  if not team_events.empty else None

        # Goals conceded = goals scored by the other team
        if team_id == home_team_id:
            goals_conceded = away_score
        elif team_id == away_team_id:
            goals_conceded = home_score
        else:
            goals_conceded = 1  # unknown — assume no clean sheet

        for _, player in lineup_df.iterrows():
            positions = player.get("positions", [])
            if not positions:
                continue
            minutes = _parse_lineup_minutes(positions)
            if minutes == 0:
                continue
            primary = positions[0]
            pos_dict = primary.get("position", {}) if isinstance(primary, dict) else {}
            pos_name = pos_dict.get("name", "") if isinstance(pos_dict, dict) else str(pos_dict)
            position = POSITION_MAP.get(pos_name, "MID")
            clean_sheet = 1 if (goals_conceded == 0 and minutes >= 45) else 0

            player_rows.append({
                "player_id":   int(player["player_id"]),
                "player_name": player["player_name"],
                "team_id":     team_id,
                "position":    position,
                "minutes":     minutes,
                "clean_sheet": clean_sheet,
            })

    if not player_rows:
        return pd.DataFrame()

    df = pd.DataFrame(player_rows)

    # ── Merge event stats ──────────────────────────────────────────────────
    for stat in [goals, xg, shots_cnt, assists, key_passes, xa, yellows, reds, gk_saves]:
        df = df.merge(
            stat.reset_index().rename(columns={"_player_id": "player_id"}),
            on="player_id", how="left",
        )

    int_cols = ["goals", "assists", "shots", "key_passes", "yellow_cards", "red_cards", "saves"]
    for c in int_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0).astype(int)
    for c in ["xg", "xa"]:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)

    # ── Fantasy points ─────────────────────────────────────────────────────
    def _pts(row):
        pos  = row["position"]
        mins = row.get("minutes", 0)
        p    = 0
        p   += SCORING["min_60"] if mins >= 60 else (SCORING["min_any"] if mins > 0 else 0)
        p   += row.get("goals", 0) * SCORING["goal"].get(pos, 8)
        p   += row.get("assists", 0) * SCORING["assist"]
        cs   = row.get("clean_sheet", 0)
        if cs:
            if pos in ("GK", "DEF"):
                p += SCORING["cs_full_gkdef"] if mins >= 90 else (SCORING["cs_half_gkdef"] if mins >= 45 else 0)
            elif pos == "MID" and mins >= 60:
                p += SCORING["cs_mid"]
        p   += row.get("yellow_cards", 0) * SCORING["yellow"]
        p   += row.get("red_cards", 0) * SCORING["red"]
        if pos == "GK":
            p += (row.get("saves", 0) // 3) * SCORING["save_3"]
        return float(p)

    df["fantasy_pts"] = df.apply(_pts, axis=1)
    df["match_id"]    = match_id
    return df


# ---------- DB writers ----------

def _upsert_teams(conn, teams_dict: dict) -> None:
    conn.executemany(
        "INSERT OR IGNORE INTO teams (team_id, name) VALUES (?, ?)",
        [(tid, tname) for tid, tname in teams_dict.items()],
    )


def _upsert_players(conn, players: list) -> None:
    conn.executemany(
        """
        INSERT OR IGNORE INTO players (player_id, name, name_norm, team_id, position)
        VALUES (?, ?, ?, ?, ?)
        """,
        players,
    )


def _upsert_match(conn, match_id, tournament, stage, matchday,
                  home_id, away_id, home_score, away_score, date) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO matches
          (match_id, tournament, stage, matchday, home_team_id, away_team_id,
           home_score, away_score, match_date)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (match_id, tournament, stage, matchday,
         home_id, away_id, home_score, away_score, date),
    )


def _upsert_player_stats(conn, df: pd.DataFrame, match_id: int) -> None:
    for _, row in df.iterrows():
        conn.execute(
            """
            INSERT OR REPLACE INTO player_match_stats
              (player_id, match_id, team_id, minutes, goals, assists, xg, xa,
               shots, key_passes, yellow_cards, red_cards, saves, clean_sheet, fantasy_pts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (int(row["player_id"]), match_id, row.get("team_id"),
             int(row.get("minutes", 0)), int(row.get("goals", 0)),
             int(row.get("assists", 0)), float(row.get("xg", 0)),
             float(row.get("xa", 0)), int(row.get("shots", 0)),
             int(row.get("key_passes", 0)), int(row.get("yellow_cards", 0)),
             int(row.get("red_cards", 0)), int(row.get("saves", 0)),
             int(row.get("clean_sheet", 0)), float(row.get("fantasy_pts", 0))),
        )


def _stage_to_matchday(stage: str) -> int:
    mapping = {
        "Group Stage": None,  # resolved per-match from match index within group
        "Round of 16": 4, "Last 16": 4,
        "Quarter-finals": 5, "Semi-finals": 6,
        "3rd Place Final": 7, "Final": 8,
    }
    return mapping.get(stage, 9)


# ---------- Main ingest ----------

def ingest_tournament(tournament_key: str) -> None:
    comp_id, season_id = COMPETITIONS[tournament_key]
    print(f"\nIngesting {tournament_key} (competition={comp_id}, season={season_id})...")

    matches_df = _sb.matches(competition_id=comp_id, season_id=season_id)
    if matches_df is None or matches_df.empty:
        print("  [warn] No matches returned — check competition/season IDs.")
        return

    print(f"  {len(matches_df)} matches found.")

    # Collect unique teams
    teams = {}
    for _, m in matches_df.iterrows():
        ht = m.get("home_team", {})
        at = m.get("away_team", {})
        if isinstance(ht, dict):
            teams[ht["home_team_id"]] = ht["home_team_name"]
        if isinstance(at, dict):
            teams[at["away_team_id"]] = at["away_team_name"]

    conn = sqlite3.connect(DB_PATH)
    _upsert_teams(conn, teams)
    conn.commit()

    processed = 0
    for _, m in matches_df.iterrows():
        match_id   = int(m["match_id"])
        ht         = m.get("home_team", {})
        at         = m.get("away_team", {})
        home_id    = ht.get("home_team_id")   if isinstance(ht, dict) else None
        away_id    = at.get("away_team_id")   if isinstance(at, dict) else None
        home_score = m.get("home_score")
        away_score = m.get("away_score")
        stage      = m.get("competition_stage", {})
        stage_name = stage.get("name", "Group Stage") if isinstance(stage, dict) else str(stage)
        matchday   = _stage_to_matchday(stage_name)
        # For group stage, use match_week as matchday
        if stage_name == "Group Stage":
            matchday = int(m.get("match_week", 1))
        date_str   = str(m.get("match_date", ""))

        match_row = {
            "home_team_id": home_id, "away_team_id": away_id,
            "home_score": home_score, "away_score": away_score,
        }

        _upsert_match(conn, match_id, tournament_key, stage_name, matchday,
                      home_id, away_id, home_score, away_score, date_str)

        stats_df = compute_match_stats(match_id, match_row)
        if stats_df.empty:
            conn.commit()
            continue

        # Upsert players seen in this match
        player_tuples = [
            (int(r["player_id"]), r["player_name"], _norm(r["player_name"]),
             r.get("team_id"), r.get("position"))
            for _, r in stats_df.iterrows()
        ]
        _upsert_players(conn, player_tuples)
        _upsert_player_stats(conn, stats_df, match_id)
        conn.commit()

        processed += 1
        if processed % 10 == 0:
            print(f"  {processed}/{len(matches_df)} matches processed")

    conn.close()
    print(f"  Done — {processed} matches ingested for {tournament_key}.")


def main():
    parser = argparse.ArgumentParser(description="Ingest StatsBomb WC data")
    parser.add_argument("--tournaments", type=str, default="WC2022,WC2018",
                        help="Comma-separated: WC2022, WC2018 (default: both)")
    args = parser.parse_args()

    init_db()
    keys = [t.strip() for t in args.tournaments.split(",") if t.strip()]
    for key in keys:
        if key not in COMPETITIONS:
            print(f"[warn] Unknown tournament: {key}. Options: {list(COMPETITIONS)}")
            continue
        ingest_tournament(key)


if __name__ == "__main__":
    main()
