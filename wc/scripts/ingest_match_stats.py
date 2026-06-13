"""
ingest_match_stats.py
Pull WC2026 match xG, scorers, assists, cards, and lineups from API-Football.

Source: API-Football via RapidAPI (free tier: 100 req/day)
  https://rapidapi.com/api-sports/api/api-football

Setup:
  1. Sign up at https://rapidapi.com (free)
  2. Subscribe to "API-Football" free plan
  3. Copy your RapidAPI key
  4. Set env var: RAPIDAPI_KEY=your_key

Usage:
  python wc/scripts/ingest_match_stats.py               # all completed WC2026 matches
  python wc/scripts/ingest_match_stats.py --dry-run     # show what would be fetched
  python wc/scripts/ingest_match_stats.py --fixture 123 # specific fixture ID

What gets stored:
  match_stats   — xG, shots, possession per match (used by DC model)
  match_events  — goals, assists, cards per player per match (used by fantasy)
  match_lineups — starting XI + subs with minutes played (used by fantasy)
"""

import argparse
import os
import sqlite3
import sys
import time
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from wc.scripts.init_db import DB_PATH, init_db

API_BASE = "https://api-football-v1.p.rapidapi.com/v3"
WC_LEAGUE_ID = 1
WC_SEASON = 2026

# minutes each substituted-on player gets (approximation when exact data unavailable)
_DEFAULT_SUB_MINUTES = 30

# API-Football position codes → our position names
POSITION_MAP = {
    "G": "GK", "D": "DEF", "M": "MID", "F": "FWD",
    "Goalkeeper": "GK", "Defender": "DEF", "Midfielder": "MID", "Attacker": "FWD",
}


def _headers(api_key: str) -> dict:
    return {
        "X-RapidAPI-Key": api_key,
        "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com",
    }


def _get(endpoint: str, params: dict, api_key: str) -> dict | None:
    url = f"{API_BASE}/{endpoint}"
    resp = requests.get(url, headers=_headers(api_key), params=params, timeout=30)
    if resp.status_code == 429:
        print("  [rate-limit] API-Football — daily limit hit, try tomorrow.")
        return None
    if not resp.ok:
        print(f"  [error] {resp.status_code}: {resp.text[:200]}")
        return None
    return resp.json()


def _get_completed_fixtures(api_key: str) -> list[dict]:
    """Return all finished WC2026 fixtures from API-Football."""
    data = _get("fixtures", {"league": WC_LEAGUE_ID, "season": WC_SEASON, "status": "FT"}, api_key)
    if not data:
        return []
    return data.get("response", [])


def _get_statistics(fixture_id: int, api_key: str) -> dict | None:
    data = _get("fixtures/statistics", {"fixture": fixture_id}, api_key)
    if not data:
        return None
    return data.get("response", [])


def _get_events(fixture_id: int, api_key: str) -> list[dict]:
    data = _get("fixtures/events", {"fixture": fixture_id}, api_key)
    if not data:
        return []
    return data.get("response", [])


def _get_lineups(fixture_id: int, api_key: str) -> list[dict]:
    data = _get("fixtures/lineups", {"fixture": fixture_id}, api_key)
    if not data:
        return []
    return data.get("response", [])


def _already_ingested(conn: sqlite3.Connection, fixture_id: int) -> bool:
    row = conn.execute(
        "SELECT 1 FROM match_stats WHERE api_fixture_id = ?", (fixture_id,)
    ).fetchone()
    return row is not None


def _ingest_one(conn: sqlite3.Connection, fixture: dict, api_key: str) -> None:
    fid = fixture["fixture"]["id"]
    match_date = fixture["fixture"]["date"][:10]
    home_team = fixture["teams"]["home"]["name"]
    away_team = fixture["teams"]["away"]["name"]
    home_score = fixture["goals"]["home"]
    away_score = fixture["goals"]["away"]

    print(f"  {home_team} vs {away_team} ({match_date})...")

    # ── Statistics (xG, shots, possession) ──────────────────────────────────
    stats_resp = _get_statistics(fid, api_key)
    time.sleep(0.3)

    xg_home = xg_away = None
    shots_home = shots_away = shots_on_home = shots_on_away = None
    poss_home = poss_away = None

    if stats_resp:
        for team_stats in stats_resp:
            team_name = team_stats["team"]["name"]
            is_home = team_name == home_team
            for s in team_stats.get("statistics", []):
                stype = s["type"]
                val = s["value"]
                if val is None:
                    continue
                # possession comes as "55%" string
                if isinstance(val, str) and val.endswith("%"):
                    try:
                        val = int(val.rstrip("%"))
                    except ValueError:
                        val = None
                elif isinstance(val, str):
                    try:
                        val = float(val)
                    except ValueError:
                        val = None
                if stype == "expected_goals":
                    if is_home:
                        xg_home = float(val) if val is not None else None
                    else:
                        xg_away = float(val) if val is not None else None
                elif stype == "Total Shots":
                    if is_home:
                        shots_home = int(val) if val is not None else None
                    else:
                        shots_away = int(val) if val is not None else None
                elif stype == "Shots on Goal":
                    if is_home:
                        shots_on_home = int(val) if val is not None else None
                    else:
                        shots_on_away = int(val) if val is not None else None
                elif stype == "Ball Possession":
                    if is_home:
                        poss_home = int(val) if val is not None else None
                    else:
                        poss_away = int(val) if val is not None else None

    conn.execute(
        """
        INSERT OR REPLACE INTO match_stats
          (api_fixture_id, match_date, home_team, away_team,
           home_score, away_score,
           xg_home, xg_away,
           shots_home, shots_away, shots_on_home, shots_on_away,
           possession_home, possession_away)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (fid, match_date, home_team, away_team,
         home_score, away_score,
         xg_home, xg_away,
         shots_home, shots_away, shots_on_home, shots_on_away,
         poss_home, poss_away),
    )

    xg_str = f"xG {xg_home:.2f}-{xg_away:.2f}" if xg_home is not None else "xG n/a"
    print(f"    Score {home_score}-{away_score}  {xg_str}")

    # ── Events (goals, assists, cards) ───────────────────────────────────────
    events = _get_events(fid, api_key)
    time.sleep(0.3)

    # Delete stale events for this fixture before re-inserting
    conn.execute("DELETE FROM match_events WHERE api_fixture_id = ?", (fid,))

    goals = assists = cards = 0
    for ev in events:
        etype = ev.get("type", "")
        detail = ev.get("detail", "")
        minute = ev.get("time", {}).get("elapsed")
        team_name = ev.get("team", {}).get("name", "")
        player_name = (ev.get("player") or {}).get("name", "")
        assist_name = (ev.get("assist") or {}).get("name")

        if not player_name:
            continue

        # Normalise event type
        if etype == "Goal":
            stored_type = "goal"
            goals += 1
        elif etype == "Card":
            if "Yellow" in detail:
                stored_type = "yellow_card"
            elif "Red" in detail or "Second Yellow" in detail:
                stored_type = "red_card"
            else:
                stored_type = "card"
            cards += 1
        elif etype == "subst":
            stored_type = "substitution"
        else:
            continue

        conn.execute(
            """
            INSERT OR IGNORE INTO match_events
              (api_fixture_id, match_date, team_name, player_name,
               event_type, event_detail, minute, assist_player)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (fid, match_date, team_name, player_name,
             stored_type, detail, minute, assist_name),
        )

        # Record assist as a separate event row
        if stored_type == "goal" and assist_name:
            conn.execute(
                """
                INSERT OR IGNORE INTO match_events
                  (api_fixture_id, match_date, team_name, player_name,
                   event_type, event_detail, minute, assist_player)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (fid, match_date, team_name, assist_name,
                 "assist", detail, minute, None),
            )
            assists += 1

    print(f"    Events: {goals} goals, {assists} assists, {cards} cards")

    # ── Lineups (minutes played) ─────────────────────────────────────────────
    lineups = _get_lineups(fid, api_key)
    time.sleep(0.3)

    conn.execute("DELETE FROM match_lineups WHERE api_fixture_id = ?", (fid,))

    total_players = 0
    for team_lineup in lineups:
        team_name = team_lineup["team"]["name"]

        # Starters get full 90 minutes (adjusted below if subbed off)
        starters = {
            p["player"]["name"]: {
                "pos": POSITION_MAP.get(p.get("pos", ""), "MID"),
                "minutes": 90,
                "is_starter": 1,
            }
            for p in team_lineup.get("startXI", [])
            if p.get("player", {}).get("name")
        }

        # Substitutes: minutes = 90 - substitution_minute (approx)
        subs = {}
        for p in team_lineup.get("substitutes", []):
            name = (p.get("player") or {}).get("name")
            if not name:
                continue
            subs[name] = {
                "pos": POSITION_MAP.get(p.get("pos", ""), "MID"),
                "minutes": 0,
                "is_starter": 0,
            }

        # Apply substitution events to adjust minutes
        team_events = [
            ev for ev in events
            if ev.get("type") == "subst"
            and (ev.get("team") or {}).get("name") == team_name
        ]
        for ev in team_events:
            minute = ev.get("time", {}).get("elapsed", 90)
            sub_in = (ev.get("assist") or {}).get("name")   # player coming on
            sub_out = (ev.get("player") or {}).get("name")  # player going off
            if sub_out and sub_out in starters:
                starters[sub_out]["minutes"] = minute
            if sub_in and sub_in in subs:
                subs[sub_in]["minutes"] = max(1, 90 - minute)

        for name, info in {**starters, **subs}.items():
            conn.execute(
                """
                INSERT OR IGNORE INTO match_lineups
                  (api_fixture_id, match_date, team_name, player_name,
                   position, minutes_played, is_starter)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (fid, match_date, team_name, name,
                 info["pos"], info["minutes"], info["is_starter"]),
            )
            total_players += 1

    print(f"    Lineups: {total_players} players")
    conn.commit()


def ingest(api_key: str, dry_run: bool = False, fixture_id: int | None = None) -> None:
    init_db()
    conn = sqlite3.connect(DB_PATH)

    if fixture_id:
        # Fetch single fixture metadata
        data = _get("fixtures", {"id": fixture_id}, api_key)
        if not data or not data.get("response"):
            print(f"[error] Fixture {fixture_id} not found.")
            conn.close()
            return
        fixtures = data["response"]
    else:
        print("Fetching completed WC2026 fixtures from API-Football...")
        fixtures = _get_completed_fixtures(api_key)
        if not fixtures:
            print("  No completed fixtures found (or API error).")
            conn.close()
            return
        print(f"  {len(fixtures)} completed fixtures found.")

    new_count = skipped = 0
    for fix in fixtures:
        fid = fix["fixture"]["id"]
        if not fixture_id and _already_ingested(conn, fid):
            skipped += 1
            continue
        if dry_run:
            home = fix["teams"]["home"]["name"]
            away = fix["teams"]["away"]["name"]
            score = f"{fix['goals']['home']}-{fix['goals']['away']}"
            print(f"  [dry-run] Would ingest fixture {fid}: {home} vs {away} {score}")
            new_count += 1
            continue
        _ingest_one(conn, fix, api_key)
        new_count += 1

    conn.close()
    if dry_run:
        print(f"\nDry run: {new_count} new fixtures would be ingested, {skipped} already done.")
    else:
        print(f"\nDone: {new_count} fixtures ingested, {skipped} already up-to-date.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest WC2026 match xG, scorers, cards from API-Football")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be fetched without writing")
    parser.add_argument("--fixture", type=int, default=None, help="Ingest a single fixture by API-Football ID")
    args = parser.parse_args()

    api_key = os.environ.get("RAPIDAPI_KEY", "")
    if not api_key:
        print(
            "[ERROR] Set RAPIDAPI_KEY environment variable.\n"
            "  1. Sign up free at https://rapidapi.com\n"
            "  2. Subscribe to 'API-Football' free plan (100 req/day)\n"
            "  3. Copy your key from the API-Football dashboard\n"
            "  4. Run: $env:RAPIDAPI_KEY = 'your_key_here'\n"
            "     then: python wc/scripts/ingest_match_stats.py"
        )
        sys.exit(1)

    ingest(api_key, dry_run=args.dry_run, fixture_id=args.fixture)


if __name__ == "__main__":
    main()
