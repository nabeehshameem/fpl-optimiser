"""
ingest_qualifying.py
Scrape FBRef qualifying campaign stats and store in `qualifying_stats`.

FBRef player stats pages have standard HTML tables — we use pandas.read_html().
Each confederation has its own qualifying competition URL.

HOW TO FIND URLs:
  1. Go to https://fbref.com/en/comps/
  2. Search for "World Cup Qualifying" for your confederation
  3. Click the competition → go to "Player Standard Stats" tab
  4. Copy the URL and add it to QUALIFYING_URLS below.

Example URLs (2026 qualifying — update if FBRef changes them):
  CONMEBOL: https://fbref.com/en/comps/CONMEBOL-WCQ/stats/
  UEFA:     https://fbref.com/en/comps/UEFA-WCQ/stats/
  (exact IDs vary — look them up on FBRef)

Usage:
  python wc/scripts/ingest_qualifying.py
  python wc/scripts/ingest_qualifying.py --url "https://fbref.com/..." --tournament WCQ2026
"""

import argparse
import re
import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from wc.scripts.init_db import DB_PATH

# ── Update these URLs before running ──────────────────────────────────────────
# Find them at: https://fbref.com/en/comps/ → search "World Cup Qualifying"
# Copy the "Player Standard Stats" page URL for each confederation.
QUALIFYING_URLS = {
    # "CONMEBOL_WCQ_2026": "https://fbref.com/en/comps/XXXX/stats/...",
    # "UEFA_WCQ_2026":     "https://fbref.com/en/comps/XXXX/stats/...",
}

TOURNAMENT_KEY = "WCQ2026"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

POSITION_MAP = {
    "GK": "GK", "DF": "DEF", "MF": "MID", "FW": "FWD",
    "DF,MF": "DEF", "MF,FW": "MID", "DF,FW": "DEF",
    "MF,DF": "MID", "FW,MF": "FWD",
}

# Mapping FBRef column names → our schema names
COL_ALIASES = {
    "MP": "matches", "Min": "minutes", "Gls": "goals", "Ast": "assists",
    "xG": "xg", "xAG": "xa", "Sh": "shots",
    # sometimes columns appear with suffixes (FBRef dedup)
    "MP.1": "matches", "Min.1": "minutes",
}


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", name.lower().strip())


def _fetch_fbref_table(url: str) -> pd.DataFrame | None:
    """
    Fetch the first meaningful player stats table from an FBRef URL.
    FBRef uses multi-level headers; we flatten them.
    """
    time.sleep(3.5)  # polite delay — FBRef rate-limits aggressively
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  [error] fetch failed: {e}")
        return None

    try:
        tables = pd.read_html(resp.text, header=[0, 1])
    except ValueError:
        try:
            tables = pd.read_html(resp.text)
        except ValueError:
            print("  [error] no tables found in page")
            return None

    # Find the table with a 'Player' column
    for tbl in tables:
        # Flatten multi-level headers
        if isinstance(tbl.columns, pd.MultiIndex):
            tbl.columns = [
                c[1] if c[1] and c[1] != c[0] else c[0]
                for c in tbl.columns
            ]
        cols_lower = [str(c).strip() for c in tbl.columns]
        if "Player" in cols_lower or "player" in [c.lower() for c in cols_lower]:
            tbl.columns = cols_lower
            return tbl

    return None


def parse_and_store(url: str, confederation: str, tournament: str = TOURNAMENT_KEY) -> int:
    print(f"  Fetching {confederation} from FBRef...")
    tbl = _fetch_fbref_table(url)
    if tbl is None:
        return 0

    # Rename columns
    tbl = tbl.rename(columns={k: v for k, v in COL_ALIASES.items() if k in tbl.columns})

    # Drop totals rows (FBRef repeats header mid-table with 'Player'=='Player')
    tbl = tbl[tbl.get("Player", tbl.iloc[:, 0]).astype(str) != "Player"]
    tbl = tbl.dropna(subset=["Player"] if "Player" in tbl.columns else [tbl.columns[0]])

    player_col = "Player" if "Player" in tbl.columns else tbl.columns[0]
    squad_col  = next((c for c in tbl.columns if c in ("Squad", "squad", "Nation", "Comp")), None)
    pos_col    = next((c for c in tbl.columns if c.lower() in ("pos", "position")), None)

    conn = sqlite3.connect(DB_PATH)
    inserted = 0

    for _, row in tbl.iterrows():
        name = str(row.get(player_col, "")).strip()
        if not name or name == "nan":
            continue

        name_norm = _norm(name)
        squad     = str(row.get(squad_col, "")).strip() if squad_col else ""
        pos_raw   = str(row.get(pos_col, "")).strip() if pos_col else ""
        position  = POSITION_MAP.get(pos_raw.split(",")[0], "MID")

        # Find or create player by normalised name
        player_row = conn.execute(
            "SELECT player_id FROM players WHERE name_norm = ?", (name_norm,)
        ).fetchone()

        if player_row:
            player_id = player_row[0]
        else:
            # Find team by short_name or create a placeholder player (ID = -hash)
            team_row = conn.execute(
                "SELECT team_id FROM teams WHERE short_name = ? OR name = ?",
                (squad, squad)
            ).fetchone()
            team_id = team_row[0] if team_row else None

            # Use a negative hash-based ID for FBRef-only players
            import hashlib
            phash = int(hashlib.md5(name_norm.encode()).hexdigest()[:8], 16)
            player_id = -(phash % 10_000_000)  # negative = FBRef-sourced

            conn.execute(
                "INSERT OR IGNORE INTO players (player_id, name, name_norm, team_id, position) VALUES (?, ?, ?, ?, ?)",
                (player_id, name, name_norm, team_id, position),
            )

        def _int(col):
            try:
                v = row.get(col)
                return int(float(v)) if v is not None and str(v) not in ("nan", "") else 0
            except (ValueError, TypeError):
                return 0

        def _float(col):
            try:
                v = row.get(col)
                return float(v) if v is not None and str(v) not in ("nan", "") else 0.0
            except (ValueError, TypeError):
                return 0.0

        conn.execute(
            """
            INSERT OR REPLACE INTO qualifying_stats
              (player_id, tournament, matches, minutes, goals, assists, xg, xa, shots)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (player_id, tournament, _int("matches"), _int("minutes"),
             _int("goals"), _int("assists"), _float("xg"), _float("xa"), _int("shots")),
        )
        inserted += 1

    conn.commit()
    conn.close()
    print(f"    {inserted} players stored for {confederation}.")
    return inserted


def main():
    parser = argparse.ArgumentParser(description="Ingest FBRef qualifying stats")
    parser.add_argument("--url", type=str, default="",
                        help="Single FBRef player stats URL to ingest")
    parser.add_argument("--confederation", type=str, default="CUSTOM",
                        help="Label for this confederation (used in logs)")
    parser.add_argument("--tournament", type=str, default=TOURNAMENT_KEY,
                        help=f"Tournament key (default: {TOURNAMENT_KEY})")
    args = parser.parse_args()

    if args.url:
        parse_and_store(args.url, args.confederation, args.tournament)
        return

    if not QUALIFYING_URLS:
        print(
            "No URLs configured. Edit QUALIFYING_URLS in this file, or pass --url.\n"
            "FBRef URL guide is in the module docstring."
        )
        return

    total = 0
    for conf, url in QUALIFYING_URLS.items():
        total += parse_and_store(url, conf, args.tournament)
    print(f"\nDone. {total} total qualifying player records stored.")


if __name__ == "__main__":
    main()
