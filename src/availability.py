"""
availability.py

Returns the set of player_ids to exclude from squad optimisation.

Two sources:
  1. config/player_exclusions.txt â€” manual list (backup GKs, known non-starters)
  2. player_snapshots.chance_of_playing_next < CHANCE_THRESHOLD â€” FPL injury flag

Pre-season the FPL field is usually NULL (unpopulated), so the manual list
carries the load in GW1; the API-driven filter kicks in once FPL starts
publishing injury data.

THE COMMENTS IN THE EXCLUSIONS FILE ARE ASSERTIONS, NOT DECORATION.

Each manual line must be written as:

    302  # Button (IPS) â€” 3rd choice GK, age 37

and the loader checks that player_id 302 really is Button at Ipswich in the
live database. If it is not, the lock FAILS rather than silently removing
somebody else. This matters because FPL reassigns player_ids between seasons
(the bug that once handed a goalkeeper a striker's expected goals), and a
hand-maintained ID list is exactly where a stale number does its damage
quietly: a mis-excluded starter is invisible in the output and permanent in
the hash-committed squad.

A hard failure at lock time is the correct trade. The lock runs 10 hours
before the deadline with an alert attached, so a bad line costs an evening's
attention; a wrong exclusion costs a gameweek and cannot be withdrawn.
"""

from __future__ import annotations

import re
import sqlite3
import unicodedata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXCLUSIONS_FILE = PROJECT_ROOT / "config" / "player_exclusions.txt"

CHANCE_THRESHOLD = 50  # exclude if chance_of_playing_next < this value

# "302  # Button (IPS) â€” reason"  ->  id=302, name=Button, team=IPS
_LINE_RE = re.compile(r"^\s*(\d+)\s*#\s*([^(]+?)\s*\(([A-Za-z0-9]{2,5})\)")


def _norm(text: str) -> str:
    """Casefold and strip accents so GroÃŸ == Gross and O'Reilly == o'reilly."""
    stripped = "".join(c for c in unicodedata.normalize("NFKD", text)
                       if not unicodedata.combining(c))
    return stripped.casefold().strip()


class ExclusionError(RuntimeError):
    """An exclusions line does not match the live database."""


def parse_exclusions_file(path: Path | None = None) -> list[tuple[int, str, str]]:
    """[(player_id, expected_name, expected_team)] from the manual file.

    A bare id with no "# Name (TEAM)" comment is rejected: an unlabelled id
    cannot be validated, which is the whole point of the format.
    """
    path = path or EXCLUSIONS_FILE
    if not path.exists():
        return []
    out: list[tuple[int, str, str]] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        m = _LINE_RE.match(raw)
        if not m:
            raise ExclusionError(
                f"{path.name} line {lineno}: {raw.strip()!r} is not in the "
                "required form '<id>  # <Name> (<TEAM>) â€” <reason>'. An "
                "unlabelled id cannot be checked against the database."
            )
        out.append((int(m.group(1)), m.group(2), m.group(3)))
    return out


def validate_exclusions(db_path: Path,
                        path: Path | None = None) -> list[tuple[int, str, str]]:
    """Check every manual exclusion against the live players table.

    Raises ExclusionError naming every mismatch at once â€” fixing them one
    failed run at a time would be needlessly slow at lock time.
    """
    entries = parse_exclusions_file(path)
    if not entries:
        return []

    ids = [e[0] for e in entries]
    ph = ",".join("?" * len(ids))
    conn = sqlite3.connect(db_path)
    try:
        actual = {
            int(r[0]): (r[1], r[2]) for r in conn.execute(
                f"SELECT p.player_id, p.web_name, t.short_name FROM players p "
                f"LEFT JOIN teams t ON t.team_id = p.team_id "
                f"WHERE p.player_id IN ({ph})", ids)
        }
    finally:
        conn.close()

    problems: list[str] = []
    for pid, want_name, want_team in entries:
        if pid not in actual:
            problems.append(
                f"  id {pid} ({want_name}, {want_team}) is not in the database "
                "â€” stale id from a previous season?")
            continue
        got_name, got_team = actual[pid]
        if _norm(got_name) != _norm(want_name):
            problems.append(
                f"  id {pid} is {got_name!r} ({got_team}), not {want_name!r} "
                f"({want_team}) â€” excluding this id would remove the wrong player")
        elif want_team and got_team and _norm(got_team) != _norm(want_team):
            problems.append(
                f"  id {pid} {got_name} is at {got_team}, not {want_team} "
                "â€” transferred? update the comment")

    if problems:
        raise ExclusionError(
            "config/player_exclusions.txt does not match the database:\n"
            + "\n".join(problems)
            + "\n\nRefusing to optimise: a wrong exclusion silently removes a "
              "real player and the resulting squad is permanent once committed."
        )
    return entries


def get_excluded_ids(db_path: Path, gameweek: int) -> set[int]:
    """Return player_ids that should be excluded from optimisation."""
    excluded: set[int] = set()

    # 1. Manual exclusions file â€” validated against the DB before use
    for pid, _name, _team in validate_exclusions(db_path):
        excluded.add(pid)

    # 2. FPL injury flag â€” only where the field is populated
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

