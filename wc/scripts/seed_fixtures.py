"""
seed_fixtures.py — Seed WC2026 group-stage fixtures into the DB.

Uses the official WC2026 group draw (12 groups × 4 teams).
Round-robin schedule per group:
  MD1: seed1 vs seed2, seed3 vs seed4
  MD2: seed1 vs seed3, seed2 vs seed4
  MD3: seed1 vs seed4, seed2 vs seed3

Kickoff times are approximate (UTC) — exact times can be updated once
the full official calendar is published by FIFA.

Run:
    python wc/scripts/seed_fixtures.py
"""

import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from wc.scripts.init_db import DB_PATH

# Standard round-robin pairings for a 4-team group (0-indexed seeds)
_PAIRINGS = {
    1: [(0, 1), (2, 3)],
    2: [(0, 2), (1, 3)],
    3: [(0, 3), (1, 2)],
}

# Official WC2026 groups (seed order: pot1, pot2, pot3, pot4 per draw)
# Name strings must match teams.name in the DB exactly.
WC2026_GROUPS = {
    "A": ["Mexico",        "South Korea",           "South Africa",          "Czech Republic"],
    "B": ["Canada",        "Switzerland",           "Qatar",                 "Bosnia and Herzegovina"],
    "C": ["Brazil",        "Morocco",               "Scotland",              "Haiti"],
    "D": ["United States", "Australia",             "Paraguay",              "Turkey"],
    "E": ["Germany",       "Curaçao",               "Ivory Coast",           "Ecuador"],
    "F": ["Netherlands",   "Japan",                 "Tunisia",               "Sweden"],
    "G": ["Belgium",       "Iran",                  "Egypt",                 "New Zealand"],
    "H": ["Spain",         "Uruguay",               "Saudi Arabia",          "Cape Verde"],
    "I": ["France",        "Senegal",               "Norway",                "Iraq"],
    "J": ["Argentina",     "Austria",               "Algeria",               "Jordan"],
    "K": ["Portugal",      "Colombia",              "Uzbekistan",            "DR Congo"],
    "L": ["England",       "Croatia",               "Panama",                "Ghana"],
}

# Approximate kickoff windows per matchday (UTC).
# Format: group → matchday → [kickoff_iso, kickoff_iso] (one per fixture in _PAIRINGS)
# Times are placeholders that keep the ordering correct; update with official schedule.
_KICKOFFS: dict[str, dict[int, list[str]]] = {
    "A": {1: ["2026-06-12T17:00:00Z", "2026-06-12T20:00:00Z"],
          2: ["2026-06-17T17:00:00Z", "2026-06-17T20:00:00Z"],
          3: ["2026-06-23T21:00:00Z", "2026-06-23T21:00:00Z"]},
    "B": {1: ["2026-06-13T17:00:00Z", "2026-06-13T20:00:00Z"],
          2: ["2026-06-18T17:00:00Z", "2026-06-18T20:00:00Z"],
          3: ["2026-06-24T21:00:00Z", "2026-06-24T21:00:00Z"]},
    "C": {1: ["2026-06-13T23:00:00Z", "2026-06-14T02:00:00Z"],
          2: ["2026-06-18T23:00:00Z", "2026-06-19T02:00:00Z"],
          3: ["2026-06-24T02:00:00Z", "2026-06-24T02:00:00Z"]},
    "D": {1: ["2026-06-14T17:00:00Z", "2026-06-14T20:00:00Z"],
          2: ["2026-06-19T17:00:00Z", "2026-06-19T20:00:00Z"],
          3: ["2026-06-25T21:00:00Z", "2026-06-25T21:00:00Z"]},
    "E": {1: ["2026-06-14T23:00:00Z", "2026-06-15T02:00:00Z"],
          2: ["2026-06-19T23:00:00Z", "2026-06-20T02:00:00Z"],
          3: ["2026-06-25T02:00:00Z", "2026-06-25T02:00:00Z"]},
    "F": {1: ["2026-06-15T17:00:00Z", "2026-06-15T20:00:00Z"],
          2: ["2026-06-20T17:00:00Z", "2026-06-20T20:00:00Z"],
          3: ["2026-06-26T21:00:00Z", "2026-06-26T21:00:00Z"]},
    "G": {1: ["2026-06-15T23:00:00Z", "2026-06-16T02:00:00Z"],
          2: ["2026-06-20T23:00:00Z", "2026-06-21T02:00:00Z"],
          3: ["2026-06-26T02:00:00Z", "2026-06-26T02:00:00Z"]},
    "H": {1: ["2026-06-16T17:00:00Z", "2026-06-16T20:00:00Z"],
          2: ["2026-06-21T17:00:00Z", "2026-06-21T20:00:00Z"],
          3: ["2026-06-27T21:00:00Z", "2026-06-27T21:00:00Z"]},
    "I": {1: ["2026-06-16T23:00:00Z", "2026-06-17T02:00:00Z"],
          2: ["2026-06-21T23:00:00Z", "2026-06-22T02:00:00Z"],
          3: ["2026-06-27T02:00:00Z", "2026-06-27T02:00:00Z"]},
    "J": {1: ["2026-06-11T23:00:00Z", "2026-06-12T02:00:00Z"],
          2: ["2026-06-16T23:00:00Z", "2026-06-17T02:00:00Z"],  # overlaps with I — MD1 opens June 11
          3: ["2026-06-22T21:00:00Z", "2026-06-22T21:00:00Z"]},
    "K": {1: ["2026-06-12T23:00:00Z", "2026-06-13T02:00:00Z"],
          2: ["2026-06-17T23:00:00Z", "2026-06-18T02:00:00Z"],
          3: ["2026-06-23T02:00:00Z", "2026-06-23T02:00:00Z"]},
    "L": {1: ["2026-06-11T17:00:00Z", "2026-06-11T20:00:00Z"],
          2: ["2026-06-16T17:00:00Z", "2026-06-16T20:00:00Z"],
          3: ["2026-06-22T02:00:00Z", "2026-06-22T02:00:00Z"]},
}


def _name_to_id(conn: sqlite3.Connection) -> dict[str, int]:
    rows = conn.execute("SELECT team_id, name FROM teams").fetchall()
    return {name: tid for tid, name in rows}


def seed(overwrite: bool = True) -> None:
    conn   = sqlite3.connect(DB_PATH)
    n2id   = _name_to_id(conn)

    if overwrite:
        conn.execute("DELETE FROM fixtures")

    inserted = 0
    missing  = []
    for grp, teams in WC2026_GROUPS.items():
        ids = []
        for t in teams:
            tid = n2id.get(t)
            if tid is None:
                missing.append(t)
                ids.append(None)
            else:
                ids.append(tid)

        for md, pairs in _PAIRINGS.items():
            ko_list = _KICKOFFS.get(grp, {}).get(md, [None, None])
            for pair_idx, (i, j) in enumerate(pairs):
                home_id = ids[i]
                away_id = ids[j]
                if home_id is None or away_id is None:
                    continue
                ko = ko_list[pair_idx] if pair_idx < len(ko_list) else None
                conn.execute(
                    "INSERT INTO fixtures (matchday, home_team_id, away_team_id, kickoff_time, group_id) VALUES (?,?,?,?,?)",
                    (md, home_id, away_id, ko, grp),
                )
                inserted += 1

    conn.commit()
    conn.close()

    if missing:
        print(f"[warn] Teams not found in DB: {missing}")
    print(f"Seeded {inserted} fixtures across 12 groups × 3 matchdays.")


if __name__ == "__main__":
    seed()
