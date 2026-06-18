"""
ingest_match_stats.py
Pull WC2026 match xG, scorers, assists, cards, and lineups from SportAPI7 (SofaScore data).

Source: SportAPI7 via RapidAPI
  https://rapidapi.com/fluis.lacasse/api/sportapi7

Setup:
  1. Sign up at https://rapidapi.com (free)
  2. Subscribe to "SportAPI7"
  3. Copy your RapidAPI key
  4. Set env var: RAPIDAPI_KEY=your_key

Usage:
  python wc/scripts/ingest_match_stats.py               # all completed WC2026 matches
  python wc/scripts/ingest_match_stats.py --dry-run     # show what would be fetched
  python wc/scripts/ingest_match_stats.py --fixture 123 # specific SofaScore event ID

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
from datetime import datetime, timezone
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from wc.scripts.init_db import DB_PATH, init_db

API_BASE = "https://sportapi7.p.rapidapi.com/api/v1"
API_HOST = "sportapi7.p.rapidapi.com"

# FIFA World Cup (men's) unique tournament ID on SofaScore
WC_TOURNAMENT_ID = 16
WC2026_SEASON_ID = 58210  # saves 1 API call per run

POSITION_MAP = {
    "G": "GK", "D": "DEF", "M": "MID", "F": "FWD",
    "Goalkeeper": "GK", "Defender": "DEF", "Midfielder": "MID", "Attacker": "FWD",
}


def _headers(api_key: str) -> dict:
    return {
        "X-RapidAPI-Key": api_key,
        "X-RapidAPI-Host": API_HOST,
    }


def _get(endpoint: str, params: dict, api_key: str) -> dict | None:
    url = f"{API_BASE}/{endpoint}"
    resp = requests.get(url, headers=_headers(api_key), params=params, timeout=30)
    if resp.status_code == 429:
        print("  [rate-limit] SportAPI7 — daily limit hit, try tomorrow.")
        return None
    if not resp.ok:
        print(f"  [error] {resp.status_code}: {resp.text[:200]}")
        return None
    return resp.json()


def _find_wc2026_season_id(api_key: str) -> int | None:
    """Find the SofaScore season ID for FIFA World Cup 2026."""
    data = _get(f"unique-tournament/{WC_TOURNAMENT_ID}/seasons", {}, api_key)
    if not data:
        return None
    for season in data.get("seasons", []):
        if "2026" in season.get("name", ""):
            print(f"  Found season: {season['name']} (id={season['id']})")
            return season["id"]
    # Fallback: print available seasons to help diagnose
    names = [s.get("name") for s in data.get("seasons", [])]
    print(f"  [warn] No 2026 season found. Available: {names[:8]}")
    return None


def _get_completed_fixtures(season_id: int, api_key: str) -> list[dict]:
    """Return all finished WC2026 fixtures by paginating last-events."""
    fixtures = []
    page = 0
    while True:
        data = _get(
            f"unique-tournament/{WC_TOURNAMENT_ID}/season/{season_id}/events/last/{page}",
            {}, api_key,
        )
        if not data:
            break
        events = data.get("events", [])
        if not events:
            break
        fixtures.extend(events)
        if not data.get("hasNextPage", False):
            break
        page += 1
        time.sleep(0.3)
    return fixtures


def _parse_float(val) -> float | None:
    if val is None:
        return None
    try:
        return float(str(val).rstrip("%").strip())
    except (ValueError, AttributeError):
        return None


def _parse_int(val) -> int | None:
    f = _parse_float(val)
    return int(f) if f is not None else None


def _already_ingested(conn: sqlite3.Connection, event_id: int) -> bool:
    row = conn.execute(
        "SELECT 1 FROM match_stats WHERE api_fixture_id = ?", (event_id,)
    ).fetchone()
    if not row:
        return False
    # Also require events and lineups — if those are missing the match was only
    # partially ingested (hit API rate limit mid-match) and needs to be retried.
    has_events = conn.execute(
        "SELECT 1 FROM match_events WHERE api_fixture_id = ? LIMIT 1", (event_id,)
    ).fetchone()
    has_lineups = conn.execute(
        "SELECT 1 FROM match_lineups WHERE api_fixture_id = ? LIMIT 1", (event_id,)
    ).fetchone()
    return bool(has_events and has_lineups)


def _ingest_one(conn: sqlite3.Connection, fixture: dict, api_key: str) -> None:
    event_id = fixture["id"]
    ts = fixture.get("startTimestamp", 0)
    match_date = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d") if ts else "unknown"
    home_team = fixture["homeTeam"]["name"]
    away_team = fixture["awayTeam"]["name"]
    home_score = (fixture.get("homeScore") or {}).get("current")
    away_score = (fixture.get("awayScore") or {}).get("current")

    print(f"  {home_team} vs {away_team} ({match_date})...")

    # ── Statistics (xG, shots, possession) ──────────────────────────────────
    stats_resp = _get(f"event/{event_id}/statistics", {}, api_key)
    time.sleep(0.3)

    xg_home = xg_away = None
    shots_home = shots_away = shots_on_home = shots_on_away = None
    poss_home = poss_away = None

    if stats_resp:
        for period in stats_resp.get("statistics", []):
            if period.get("period") != "ALL":
                continue
            for group in period.get("groups", []):
                for item in group.get("statisticsItems", []):
                    name = item.get("name", "").lower()
                    h, a = item.get("home"), item.get("away")
                    if "expected goals" in name or name == "xg":
                        xg_home, xg_away = _parse_float(h), _parse_float(a)
                    elif "total shots" in name:
                        shots_home, shots_away = _parse_int(h), _parse_int(a)
                    elif "shots on target" in name:
                        shots_on_home, shots_on_away = _parse_int(h), _parse_int(a)
                    elif "ball possession" in name:
                        poss_home = _parse_int(str(h).rstrip("%"))
                        poss_away = _parse_int(str(a).rstrip("%"))

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
        (event_id, match_date, home_team, away_team,
         home_score, away_score,
         xg_home, xg_away,
         shots_home, shots_away, shots_on_home, shots_on_away,
         poss_home, poss_away),
    )

    xg_str = f"xG {xg_home:.2f}-{xg_away:.2f}" if xg_home is not None else "xG n/a"
    print(f"    Score {home_score}-{away_score}  {xg_str}")

    # ── Incidents (goals, cards, substitutions) ──────────────────────────────
    incidents_resp = _get(f"event/{event_id}/incidents", {}, api_key)
    time.sleep(0.3)

    conn.execute("DELETE FROM match_events WHERE api_fixture_id = ?", (event_id,))

    goals = assists = cards = 0
    for inc in (incidents_resp or {}).get("incidents", []):
        inc_type = inc.get("incidentType", "")
        inc_class = inc.get("incidentClass", "")
        is_home = inc.get("isHome", True)
        team_name = home_team if is_home else away_team
        minute = inc.get("time")

        if inc_type == "goal":
            player_name = (inc.get("player") or {}).get("name", "")
            assist_name = (inc.get("assist1") or {}).get("name")
            if not player_name:
                continue
            conn.execute(
                """
                INSERT OR IGNORE INTO match_events
                  (api_fixture_id, match_date, team_name, player_name,
                   event_type, event_detail, minute, assist_player)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (event_id, match_date, team_name, player_name,
                 "goal", inc_class, minute, assist_name),
            )
            goals += 1
            if assist_name:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO match_events
                      (api_fixture_id, match_date, team_name, player_name,
                       event_type, event_detail, minute, assist_player)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (event_id, match_date, team_name, assist_name,
                     "assist", inc_class, minute, None),
                )
                assists += 1

        elif inc_type == "card":
            player_name = (inc.get("player") or {}).get("name", "")
            if not player_name:
                continue
            stored_type = "yellow_card" if inc_class == "yellow" else "red_card"
            conn.execute(
                """
                INSERT OR IGNORE INTO match_events
                  (api_fixture_id, match_date, team_name, player_name,
                   event_type, event_detail, minute, assist_player)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (event_id, match_date, team_name, player_name,
                 stored_type, inc_class, minute, None),
            )
            cards += 1

        elif inc_type == "substitution":
            player_in = (inc.get("playerIn") or {}).get("name")
            player_out = (inc.get("playerOut") or {}).get("name")
            if player_out:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO match_events
                      (api_fixture_id, match_date, team_name, player_name,
                       event_type, event_detail, minute, assist_player)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (event_id, match_date, team_name, player_out,
                     "substitution", "out", minute, player_in),
                )
            if player_in:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO match_events
                      (api_fixture_id, match_date, team_name, player_name,
                       event_type, event_detail, minute, assist_player)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (event_id, match_date, team_name, player_in,
                     "substitution", "in", minute, player_out),
                )

    print(f"    Events: {goals} goals, {assists} assists, {cards} cards")

    # ── Lineups (minutes played come directly from SofaScore player stats) ───
    lineups_resp = _get(f"event/{event_id}/lineups", {}, api_key)
    time.sleep(0.3)

    conn.execute("DELETE FROM match_lineups WHERE api_fixture_id = ?", (event_id,))

    total_players = 0
    if lineups_resp:
        for side, team_name in [("home", home_team), ("away", away_team)]:
            for p in (lineups_resp.get(side) or {}).get("players", []):
                player_name = (p.get("player") or {}).get("name", "")
                if not player_name:
                    continue
                pos = POSITION_MAP.get(p.get("position", ""), "MID")
                is_starter = 0 if p.get("substitute", False) else 1
                minutes = (p.get("statistics") or {}).get("minutesPlayed") or 0
                conn.execute(
                    """
                    INSERT OR IGNORE INTO match_lineups
                      (api_fixture_id, match_date, team_name, player_name,
                       position, minutes_played, is_starter)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (event_id, match_date, team_name, player_name,
                     pos, minutes, is_starter),
                )
                total_players += 1

    print(f"    Lineups: {total_players} players")
    conn.commit()


def ingest(api_key: str, dry_run: bool = False, fixture_id: int | None = None) -> None:
    init_db()
    conn = sqlite3.connect(DB_PATH)

    if fixture_id:
        data = _get(f"event/{fixture_id}", {}, api_key)
        if not data or not data.get("event"):
            print(f"[error] Event {fixture_id} not found.")
            conn.close()
            return
        fixtures = [data["event"]]
    else:
        if WC2026_SEASON_ID:
            season_id = WC2026_SEASON_ID
            print(f"Using hardcoded WC2026 season id={season_id}")
        else:
            print("Finding WC2026 season on SportAPI7...")
            season_id = _find_wc2026_season_id(api_key)
            if not season_id:
                print("[error] Could not find WC2026 season. Check WC_TOURNAMENT_ID constant.")
                conn.close()
                return
        print("Fetching completed WC2026 fixtures...")
        fixtures = _get_completed_fixtures(season_id, api_key)
        if not fixtures:
            print("  No completed fixtures found (or API error).")
            conn.close()
            return
        print(f"  {len(fixtures)} completed fixtures found.")

    new_count = skipped = 0
    for fix in fixtures:
        fid = fix["id"]
        if not fixture_id and _already_ingested(conn, fid):
            skipped += 1
            continue
        if dry_run:
            home = fix["homeTeam"]["name"]
            away = fix["awayTeam"]["name"]
            h_score = (fix.get("homeScore") or {}).get("current", "?")
            a_score = (fix.get("awayScore") or {}).get("current", "?")
            print(f"  [dry-run] Would ingest event {fid}: {home} vs {away} {h_score}-{a_score}")
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
    parser = argparse.ArgumentParser(description="Ingest WC2026 match xG, scorers, cards from SportAPI7")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be fetched without writing")
    parser.add_argument("--fixture", type=int, default=None, help="Ingest a single event by SofaScore ID")
    args = parser.parse_args()

    api_key = os.environ.get("RAPIDAPI_KEY", "")
    if not api_key:
        print(
            "[ERROR] Set RAPIDAPI_KEY environment variable.\n"
            "  1. Sign up free at https://rapidapi.com\n"
            "  2. Subscribe to 'SportAPI7'\n"
            "  3. Copy your key from the dashboard\n"
            "  4. Run: $env:RAPIDAPI_KEY = 'your_key_here'\n"
            "     then: python wc/scripts/ingest_match_stats.py"
        )
        sys.exit(1)

    ingest(api_key, dry_run=args.dry_run, fixture_id=args.fixture)


if __name__ == "__main__":
    main()
