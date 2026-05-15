"""
ingest_fixtures.py
Load World Cup 2026 fixtures and team FIFA rankings.

Two modes:
  1. football-data.org API (free, requires a free API key)
     Sign up at: https://www.football-data.org/client/register
     Set env var: FD_API_KEY=your_key
  2. JSON fallback — provide a fixtures.json file (see --json flag)

Football-data.org endpoints:
  GET https://api.football-data.org/v4/competitions/WC/matches
  GET https://api.football-data.org/v4/competitions/WC/teams

Usage:
  $env:FD_API_KEY = "your_key"
  python wc/scripts/ingest_fixtures.py
  python wc/scripts/ingest_fixtures.py --json path/to/fixtures.json
"""

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from wc.scripts.init_db import DB_PATH

FD_BASE   = "https://api.football-data.org/v4"
WC_COMP   = "WC"
HEADERS   = lambda key: {"X-Auth-Token": key}


# ── FIFA Rankings (2026 WC, approximate) ─────────────────────────────────────
# Update this table with the final FIFA rankings before the tournament.
# Source: https://www.fifa.com/fifa-world-ranking
FIFA_RANKINGS = {
    # fmt: off  (name → rank, add more as needed)
    "Argentina":  1,  "France":      2,  "England":     3,
    "Belgium":    4,  "Brazil":      5,  "Portugal":    6,
    "Spain":      7,  "Netherlands": 8,  "Italy":       9,
    "Germany":   10,  "Croatia":    11,  "Morocco":    12,
    "USA":       13,  "Mexico":     14,  "Japan":      15,
    "South Korea":16, "Senegal":    17,  "Ecuador":    18,
    "Uruguay":   19,  "Colombia":   20,
    # add all 48 qualified teams
}


def _fd_get(endpoint: str, api_key: str) -> dict | None:
    url  = f"{FD_BASE}/{endpoint}"
    resp = requests.get(url, headers=HEADERS(api_key), timeout=30)
    if resp.status_code == 429:
        print("  [rate-limit] football-data.org — try again in a minute.")
        return None
    if not resp.ok:
        print(f"  [error] {resp.status_code}: {resp.text[:200]}")
        return None
    return resp.json()


def ingest_from_api(api_key: str) -> None:
    print("  Fetching WC2026 fixtures from football-data.org...")
    data = _fd_get(f"competitions/{WC_COMP}/matches", api_key)
    if not data:
        return

    teams_data = _fd_get(f"competitions/{WC_COMP}/teams", api_key)

    conn = sqlite3.connect(DB_PATH)

    # Upsert teams
    if teams_data and "teams" in teams_data:
        for t in teams_data["teams"]:
            name     = t.get("name", "")
            short    = t.get("shortName") or t.get("tla") or name[:4]
            rank     = FIFA_RANKINGS.get(name)
            team_id  = t["id"]
            conn.execute(
                """
                INSERT OR REPLACE INTO teams (team_id, name, short_name, fifa_rank, source)
                VALUES (?, ?, ?, ?, 'football-data')
                """,
                (team_id, name, short, rank),
            )

    # Upsert fixtures
    conn.execute("DELETE FROM fixtures")
    for m in data.get("matches", []):
        gd          = m.get("group") or ""
        stage       = m.get("stage", "")
        matchday    = m.get("matchday") or 1
        kickoff     = m.get("utcDate", "")
        home        = m.get("homeTeam", {}).get("id")
        away        = m.get("awayTeam", {}).get("id")
        if not home or not away:
            continue
        if "GROUP" in stage.upper() or "ROUND" in stage.upper():
            conn.execute(
                """
                INSERT INTO fixtures (matchday, home_team_id, away_team_id, kickoff_time, group_id)
                VALUES (?, ?, ?, ?, ?)
                """,
                (matchday, home, away, kickoff, gd),
            )

    conn.commit()
    conn.close()
    print("  Fixtures ingested from football-data.org.")


def ingest_from_json(path: str) -> None:
    """
    Ingest from a local JSON file.
    Expected format:
    {
      "teams": [{"id": 1, "name": "Argentina", "short_name": "ARG", "group": "A"}],
      "fixtures": [{"matchday": 1, "home": 1, "away": 2, "kickoff": "2026-06-11T18:00:00Z", "group": "A"}]
    }
    """
    data = json.loads(Path(path).read_text())
    conn = sqlite3.connect(DB_PATH)

    for t in data.get("teams", []):
        rank = FIFA_RANKINGS.get(t.get("name", ""))
        conn.execute(
            """
            INSERT OR REPLACE INTO teams (team_id, name, short_name, fifa_rank, group_id, source)
            VALUES (?, ?, ?, ?, ?, 'json')
            """,
            (t["id"], t["name"], t.get("short_name", t["name"][:4]), rank, t.get("group")),
        )

    conn.execute("DELETE FROM fixtures")
    for f in data.get("fixtures", []):
        conn.execute(
            "INSERT INTO fixtures (matchday, home_team_id, away_team_id, kickoff_time, group_id) VALUES (?, ?, ?, ?, ?)",
            (f["matchday"], f["home"], f["away"], f.get("kickoff"), f.get("group")),
        )

    conn.commit()
    conn.close()
    print(f"  {len(data.get('teams', []))} teams and {len(data.get('fixtures', []))} fixtures loaded from JSON.")


def main():
    parser = argparse.ArgumentParser(description="Ingest WC2026 fixtures and team data")
    parser.add_argument("--json", type=str, default="",
                        help="Path to a fixtures.json file (skips API)")
    args = parser.parse_args()

    if args.json:
        ingest_from_json(args.json)
        return

    api_key = os.environ.get("FD_API_KEY", "")
    if not api_key:
        print(
            "[ERROR] Set FD_API_KEY environment variable.\n"
            "  Get a free key at: https://www.football-data.org/client/register\n"
            "  Or use: python ingest_fixtures.py --json path/to/fixtures.json"
        )
        sys.exit(1)

    ingest_from_api(api_key)


if __name__ == "__main__":
    main()
