"""
ingest_fantasy_players.py
Load the World Cup Fantasy player roster into `fantasy_players`.

The WC Fantasy game (FIFA's official game or the most popular platform)
typically launches 2-3 weeks before the tournament.

TWO MODES:

1. JSON file (recommended for v1):
   Download the player list manually and provide a JSON file.
   Expected format:
   [
     {"id": 1, "name": "Kylian Mbappe", "team": "France", "pos": "FWD",
      "price": 120, "ownership": 45.2},
     ...
   ]
   Usage: python wc/scripts/ingest_fantasy_players.py --json players.json

2. FIFA Fantasy API (when available):
   The FIFA Fantasy API is undocumented but the endpoint structure is usually:
     GET https://fantasy.fifa.com/api/v2/players
   or similar. Check the browser's Network tab when loading the player page.
   Usage: Set FIFA_FANTASY_URL in this file, then run without --json.

Ownership matching to StatsBomb players:
   Players are matched to the `players` table by normalised name.
   Unmatched players are stored with player_id = NULL — they still get
   predictions based on qualifying stats if qualifying_stats has their name.
"""

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from wc.scripts.init_db import DB_PATH

FIFA_PLAYERS_URL = "https://play.fifa.com/json/fantasy/players.json"
FIFA_SQUADS_URL  = "https://play.fifa.com/json/fantasy/squads_fifa.json"


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", name.lower().strip())


def _match_player_id(conn, name: str) -> int | None:
    """Try to match by exact normalised name, then by last name."""
    norm = _norm(name)
    row = conn.execute(
        "SELECT player_id FROM players WHERE name_norm = ?", (norm,)
    ).fetchone()
    if row:
        return row[0]
    # Last-name fallback
    last = norm.split()[-1] if norm.split() else ""
    if len(last) >= 4:
        rows = conn.execute(
            "SELECT player_id FROM players WHERE name_norm LIKE ?",
            (f"%{last}%",)
        ).fetchall()
        if len(rows) == 1:
            return rows[0][0]
    return None


def _upsert_team(conn, team_name: str) -> int | None:
    row = conn.execute(
        "SELECT team_id FROM teams WHERE name = ? OR short_name = ?",
        (team_name, team_name)
    ).fetchone()
    return row[0] if row else None


POSITION_MAP = {
    "GK": "GK", "GKP": "GK", "Goalkeeper": "GK",
    "DEF": "DEF", "Defender": "DEF", "DF": "DEF",
    "MID": "MID", "Midfielder": "MID", "MF": "MID",
    "FWD": "FWD", "Forward": "FWD", "FW": "FWD", "ATT": "FWD",
}


def ingest_from_json(path: str) -> int:
    data = json.loads(Path(path).read_text())
    if not isinstance(data, list):
        data = data.get("players", data.get("elements", []))

    conn  = sqlite3.connect(DB_PATH)
    count = 0
    unmatched = []

    for p in data:
        name    = str(p.get("name") or p.get("web_name") or "").strip()
        if not name:
            continue
        team_name = str(p.get("team") or "").strip()
        pos_raw   = str(p.get("pos") or p.get("position") or p.get("element_type") or "MID").strip()
        position  = POSITION_MAP.get(pos_raw, "MID")
        price     = int(float(p.get("price") or p.get("now_cost") or 0))
        ownership = float(p.get("ownership") or p.get("selected_by_percent") or 0.0)
        fid       = int(p.get("id") or p.get("fantasy_id") or 0)
        if not fid:
            continue

        player_id = _match_player_id(conn, name)
        team_id   = _upsert_team(conn, team_name)
        name_norm = _norm(name)

        if player_id is None:
            unmatched.append(name)

        conn.execute(
            """
            INSERT OR REPLACE INTO fantasy_players
              (fantasy_id, player_id, name, name_norm, team_id, position, price, ownership)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (fid, player_id, name, name_norm, team_id, position, price, ownership),
        )
        count += 1

    conn.commit()
    conn.close()

    print(f"  {count} fantasy players stored.")
    if unmatched:
        print(f"  {len(unmatched)} unmatched (no StatsBomb player_id):")
        for n in unmatched[:20]:
            print(f"    - {n}")
        if len(unmatched) > 20:
            print(f"    ... and {len(unmatched) - 20} more")
    return count


def ingest_from_fifa_api() -> int:
    """Pull live player + price data from play.fifa.com/json/fantasy/."""
    print(f"  Fetching squads from {FIFA_SQUADS_URL}")
    squads_resp = requests.get(FIFA_SQUADS_URL, timeout=30)
    squads_resp.raise_for_status()
    squads_data = squads_resp.json()
    squads_list = squads_data if isinstance(squads_data, list) else squads_data.get("squads", [])
    squad_map = {s["id"]: s["name"] for s in squads_list if "id" in s and "name" in s}

    print(f"  Fetching players from {FIFA_PLAYERS_URL}")
    players_resp = requests.get(FIFA_PLAYERS_URL, timeout=30)
    players_resp.raise_for_status()
    raw = players_resp.json()
    raw_players = raw if isinstance(raw, list) else raw.get("players", [])

    # Normalise to the format ingest_from_json expects
    normalised = []
    for p in raw_players:
        known = (p.get("knownName") or "").strip()
        first = (p.get("firstName") or "").strip()
        last  = (p.get("lastName") or "").strip()
        name  = known or f"{first} {last}".strip()
        if not name:
            continue
        team = squad_map.get(p.get("squadId"), "")
        price_raw = p.get("price", 0)
        normalised.append({
            "id":    p.get("id") or p.get("fifaId") or 0,
            "name":  name,
            "team":  team,
            "pos":   p.get("position", "MID"),
            "price": int(round(float(price_raw) * 10)),  # store as int (×10 = $0.1m units)
            "ownership": float(p.get("percentSelected") or 0.0),
        })

    print(f"  {len(normalised)} players fetched, {len(squad_map)} squads resolved.")
    tmp = Path("/tmp/wc_fantasy_players.json")
    tmp.write_text(json.dumps(normalised))
    return ingest_from_json(str(tmp))


def ingest_from_api(url: str) -> int:
    print(f"  Fetching from {url}")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    players = data if isinstance(data, list) else data.get("players", [])
    tmp = Path("/tmp/wc_fantasy_players.json")
    tmp.write_text(json.dumps(players))
    return ingest_from_json(str(tmp))


def main():
    parser = argparse.ArgumentParser(description="Ingest WC Fantasy player roster")
    parser.add_argument("--json", type=str, default="",
                        help="Path to a players JSON file")
    args = parser.parse_args()

    if args.json:
        ingest_from_json(args.json)
        return

    ingest_from_fifa_api()


if __name__ == "__main__":
    main()
