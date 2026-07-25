"""
availability.py

Returns the set of player_ids to exclude from squad optimisation.

Two sources:
  1. config/player_exclusions.txt — manual list (backup GKs, known non-starters)
  2. player_snapshots.chance_of_playing_next < CHANCE_THRESHOLD — FPL injury flag

Pre-season the FPL field is usually NULL (unpopulated), so the manual list
carries the load in GW1; the API-driven filter kicks in once FPL starts
publishing injury data.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXCLUSIONS_FILE = PROJECT_ROOT / "config" / "player_exclusions.txt"

CHANCE_THRESHOLD = 50  # exclude if chance_of_playing_next < this value


def get_excluded_ids(db_path: Path, gameweek: int) -> set[int]:
    """Return player_ids that should be excluded from optimisation."""
    excluded: set[int] = set()

    # 1. Manual exclusions file
    if EXCLUSIONS_FILE.exists():
        for line in EXCLUSIONS_FILE.read_text().splitlines():
            line = line.split("#")[0].strip()
            if line:
                try:
                    excluded.add(int(line))
                except ValueError:
                    pass

    # 2. FPL injury flag — only where the field is populated
    try:
        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            """
            SELECT player_id FROM player_snapshots
            WHERE gameweek_id = ?
              AND chance_of_playing_next IS NOT NULL
              AND chance_of_playing_next < ?
            """,
            (gameweek, CHANCE_THRESHOLD),
        ).fetchall()
        conn.close()
        excluded.update(r[0] for r in rows)
    except Exception:
        pass

    return excluded
