"""
set_fifa_ranks.py
Populates fifa_rank in the teams table using approximate 2025/2026 rankings.

Ranks matter because the DC model uses them as a regularisation prior —
they prevent small-sample WC results from producing wildly distorted
team attack/defence estimates.
"""

import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from wc.scripts.init_db import DB_PATH

# Approximate FIFA world rankings, May 2026.
# Key source: public rankings + WC2026 squad form knowledge.
# Teams are matched case-insensitively against the DB.
RANKS: dict[str, int] = {
    "Argentina":     1,
    "France":        2,
    "England":       3,
    "Belgium":       4,
    "Brazil":        5,
    "Spain":         6,
    "Portugal":      7,
    "Netherlands":   8,
    "Croatia":       9,
    "Colombia":      11,
    "Germany":       12,
    "United States": 13,
    "Morocco":       14,
    "Mexico":        16,
    "Japan":         17,
    "Uruguay":       18,
    "Senegal":       19,
    "Switzerland":   20,
    "Denmark":       21,
    "Iran":          22,
    "South Korea":   23,
    "Australia":     24,
    "Serbia":        25,
    "Wales":         27,
    "Poland":        27,
    "Tunisia":       29,
    "Norway":        31,
    "Sweden":        30,
    "Ecuador":       33,
    "Egypt":         34,
    "Qatar":         37,
    "Nigeria":       38,
    "Canada":        41,
    "Cameroon":      48,
    "Costa Rica":    49,
    "Ghana":         53,
    "Russia":        63,
    "Saudi Arabia":  57,
    "Panama":        62,
    "Iceland":       72,
    "Peru":          88,
    "Austria":       22,
    "Turkey":        38,
    "Czech Republic":40,
    "Scotland":      42,
    "Algeria":       45,
    "Paraguay":      55,
    "DR Congo":      58,
    "Uzbekistan":    65,
    "New Zealand":  102,
    # WC2026 additions
    "South Africa":          58,
    "Bosnia and Herzegovina":65,
    "Haiti":                115,
    "Curaçao":               94,
    "Ivory Coast":           48,
    "Cape Verde":            80,
    "Iraq":                  73,
    "Jordan":                91,
}


def _ensure_teams(conn: sqlite3.Connection, names: list[str]) -> None:
    """Insert teams that exist in recent_results but not yet in the teams table."""
    for name in names:
        exists = conn.execute(
            "SELECT 1 FROM teams WHERE LOWER(name) = LOWER(?)", (name,)
        ).fetchone()
        if not exists:
            conn.execute(
                "INSERT INTO teams (name, short_name, source) VALUES (?, ?, 'recent_results')",
                (name, name[:3].upper()),
            )
            print(f"  [info] Inserted missing team '{name}' into teams table")


def run():
    conn = sqlite3.connect(DB_PATH)

    # Teams that qualify for WC2026 but weren't in WC2018/2022 StatsBomb data
    _ensure_teams(conn, [
        "Norway", "Austria", "Turkey", "Czech Republic",
        "Scotland", "Algeria", "Paraguay", "DR Congo",
        "Uzbekistan", "New Zealand",
        "South Africa", "Bosnia and Herzegovina", "Haiti",
        "Curaçao", "Ivory Coast", "Cape Verde", "Iraq", "Jordan",
    ])
    conn.commit()

    updated = 0
    for name, rank in RANKS.items():
        cur = conn.execute(
            "UPDATE teams SET fifa_rank = ? WHERE LOWER(name) = LOWER(?)",
            (rank, name),
        )
        if cur.rowcount:
            updated += 1
        else:
            # Try partial match for known aliases (e.g. "Korea Republic" vs "South Korea")
            rows = conn.execute(
                "SELECT team_id, name FROM teams WHERE LOWER(name) LIKE ?",
                (f"%{name.lower().split()[-1]}%",),
            ).fetchall()
            if len(rows) == 1:
                conn.execute(
                    "UPDATE teams SET fifa_rank = ? WHERE team_id = ?",
                    (rank, rows[0][0]),
                )
                updated += 1
            else:
                print(f"  [warn] no unique match for '{name}' (found {len(rows)})")

    conn.commit()

    # Verify
    rows = conn.execute(
        "SELECT name, fifa_rank FROM teams WHERE fifa_rank IS NOT NULL ORDER BY fifa_rank"
    ).fetchall()
    conn.close()
    print(f"Updated {updated} teams. Rankings set:")
    for name, rank in rows:
        print(f"  {rank:3}  {name}")


if __name__ == "__main__":
    run()
