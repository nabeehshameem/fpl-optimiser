"""
test_availability.py

Tests for src/availability.py. Run: python scripts/test_availability.py

The exclusions file is hand-maintained ids. FPL reassigns ids between seasons,
so the comments beside each id are treated as assertions and checked against
the live players table. These tests prove the check has teeth.

V1  A correct file loads and returns exactly those ids
V2  Wrong NAME for an id -> ExclusionError naming the actual player
V3  Wrong TEAM (player transferred) -> ExclusionError
V4  Id absent from the database (stale from last season) -> ExclusionError
V5  Bare id with no "# Name (TEAM)" label -> rejected, because an unlabelled
    id cannot be validated at all
V6  Accents and case do not cause false failures (Groß vs Gross)
V7  chance_of_playing_next < threshold is merged in, and NULL is not treated
    as unavailable (pre-season the field is empty for everyone)
"""

from __future__ import annotations

import sqlite3
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import src.availability as avail  # noqa: E402
from src.availability import (  # noqa: E402
    ExclusionError, get_excluded_ids, validate_exclusions,
)

PLAYERS = [
    # (player_id, web_name, team_id, position)
    (2, "Arrizabalaga", 1, 1),
    (3, "Meslier", 1, 1),
    (10, "Raya", 1, 1),
    (302, "Button", 2, 1),
    (303, "Walton", 2, 1),
    (385, "Trafford", 3, 1),
    (400, "Groß", 4, 3),
    (500, "Semenyo", 3, 3),
]
TEAMS = [(1, "ARS"), (2, "IPS"), (3, "MCI"), (4, "BHA")]


def build_db(tmp: Path, chances: dict[int, int | None] | None = None) -> Path:
    db = tmp / "fpl.db"
    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE TABLE teams (team_id INTEGER PRIMARY KEY, short_name TEXT);
        CREATE TABLE players (player_id INTEGER PRIMARY KEY, web_name TEXT,
            team_id INT, position INT);
        CREATE TABLE player_snapshots (snapshot_id INTEGER PRIMARY KEY
            AUTOINCREMENT, player_id INT, gameweek_id INT,
            chance_of_playing_next INT);
    """)
    conn.executemany("INSERT INTO teams VALUES (?,?)", TEAMS)
    conn.executemany("INSERT INTO players VALUES (?,?,?,?)", PLAYERS)
    for pid, chance in (chances or {}).items():
        conn.execute("INSERT INTO player_snapshots "
                     "(player_id, gameweek_id, chance_of_playing_next) "
                     "VALUES (?,1,?)", (pid, chance))
    conn.commit()
    conn.close()
    return db


def write_file(tmp: Path, body: str) -> Path:
    p = tmp / "player_exclusions.txt"
    p.write_text(body, encoding="utf-8")
    return p


def check(label, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {label}" + (f"  ({detail})" if detail else ""))
    return cond


def expect_error(label, fn, must_contain):
    try:
        fn()
        return check(label, False, "no error raised")
    except ExclusionError as e:
        return check(label, must_contain in str(e), str(e).replace("\n", " ")[:90])


def main():
    ok = True
    tmp = Path(tempfile.mkdtemp())
    db = build_db(tmp)

    # V1 — a correct file
    good = write_file(tmp, """
# comment line ignored
2    # Arrizabalaga (ARS) — backup behind Raya
302  # Button (IPS) — 3rd choice GK
""")
    entries = validate_exclusions(db, good)
    ok &= check("V1 correct file validates", {e[0] for e in entries} == {2, 302},
                str([e[0] for e in entries]))

    # V2 — wrong name (the ID-reassignment case)
    d2 = tmp / "v2"; d2.mkdir(exist_ok=True)
    bad_name = write_file(d2, "302  # Trafford (IPS) — wrong name\n")
    ok &= expect_error("V2 wrong name rejected",
                       lambda: validate_exclusions(db, bad_name),
                       "not 'Trafford'")

    # V3 — wrong team (transferred)
    d3 = tmp / "v3"; d3.mkdir(exist_ok=True)
    bad_team = write_file(d3, "500  # Semenyo (BOU) — stale team\n")
    ok &= expect_error("V3 wrong team rejected",
                       lambda: validate_exclusions(db, bad_team),
                       "is at MCI")

    # V4 — id not in the database at all
    d4 = tmp / "v4"; d4.mkdir(exist_ok=True)
    stale = write_file(d4, "99999  # Ghost (ARS) — left the league\n")
    ok &= expect_error("V4 stale id rejected",
                       lambda: validate_exclusions(db, stale),
                       "not in the database")

    # V5 — unlabelled id cannot be validated, so it is refused
    d5 = tmp / "v5"; d5.mkdir(exist_ok=True)
    bare = write_file(d5, "302\n")
    ok &= expect_error("V5 unlabelled id rejected",
                       lambda: validate_exclusions(db, bare),
                       "not in the required form")

    # V6 — accents and case must not cause false alarms
    d6 = tmp / "v6"; d6.mkdir(exist_ok=True)
    accent = write_file(d6, "400  # Gross (bha) — accent-stripped, lowercase team\n")
    try:
        e = validate_exclusions(db, accent)
        ok &= check("V6 accent/case tolerated", {x[0] for x in e} == {400})
    except ExclusionError as exc:
        ok &= check("V6 accent/case tolerated", False, str(exc)[:80])

    # V7 — injury flag merges in; NULL is not "unavailable"
    d7 = tmp / "v7"; d7.mkdir(exist_ok=True)
    db2 = build_db(d7, chances={10: 25, 303: 75, 500: None})
    avail.EXCLUSIONS_FILE = good
    got = get_excluded_ids(db2, 1)
    ok &= check("V7 low chance_of_playing excluded", 10 in got)
    ok &= check("V7 75% not excluded", 303 not in got)
    ok &= check("V7 NULL chance not excluded", 500 not in got)
    ok &= check("V7 manual file merged in", {2, 302} <= got, sorted(got))

    avail.EXCLUSIONS_FILE = PROJECT_ROOT / "config" / "player_exclusions.txt"

    print("\n" + ("ALL PASS" if ok else "FAILURES PRESENT"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
