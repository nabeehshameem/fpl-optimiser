"""
Synthetic tests for UCL ingest layer (init_db.py schema).
No API key required — tests use in-memory / tempdir databases only.
Run: python ucl/test_ucl_ingest.py

I1  init_db creates all required tables
I2  Schema is idempotent (second run doesn't raise)
I3  Fixture round-trip: insert + query returns correct data
I4  Foreign key constraint on match_stats is enforced
I5  short_name UNIQUE on teams is enforced
"""

from __future__ import annotations

import sqlite3
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def check(label, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {label}" + (f"  ({detail})" if detail else ""))
    return cond


REQUIRED_TABLES = {"teams", "fixtures", "match_stats", "dc_params", "league_standings"}


def _make_db(tmp: Path) -> Path:
    from ucl.init_db import SCHEMA, INDEXES
    db = tmp / "ucl.db"
    conn = sqlite3.connect(db)
    conn.execute("PRAGMA foreign_keys = ON")
    for stmt in SCHEMA:
        conn.execute(stmt)
    for idx in INDEXES:
        conn.execute(idx)
    conn.commit()
    conn.close()
    return db


def main():
    ok = True
    tmp = Path(tempfile.mkdtemp())

    # I1: all tables created
    db = _make_db(tmp)
    conn = sqlite3.connect(db)
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )}
    conn.close()
    ok &= check("I1 all required tables exist",
                REQUIRED_TABLES.issubset(tables), str(tables))

    # I2: idempotent — second call must not raise
    try:
        _make_db(tmp)
        ok &= check("I2 schema creation is idempotent", True)
    except Exception as e:
        ok &= check("I2 schema creation is idempotent", False, str(e))

    # I3: fixture round-trip
    conn = sqlite3.connect(db)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("INSERT INTO teams VALUES (1, 'Real Madrid', 'RMA')")
    conn.execute("INSERT INTO teams VALUES (2, 'Manchester City', 'MCI')")
    conn.execute(
        "INSERT INTO fixtures VALUES "
        "(42, 'League Phase', 1, 2, '2026-09-17T20:00:00Z', 'NS', NULL, NULL, '2026-09-15T00:00:00Z')"
    )
    conn.commit()
    row = conn.execute(
        "SELECT f.fixture_id, th.short_name, ta.short_name, f.round_name "
        "FROM fixtures f "
        "JOIN teams th ON th.team_id = f.home_team_id "
        "JOIN teams ta ON ta.team_id = f.away_team_id "
        "WHERE f.fixture_id = 42"
    ).fetchone()
    conn.close()
    ok &= check("I3 fixture round-trip: id", row and row[0] == 42)
    ok &= check("I3 fixture round-trip: home short_name", row and row[1] == "RMA", str(row))
    ok &= check("I3 fixture round-trip: away short_name", row and row[2] == "MCI", str(row))
    ok &= check("I3 fixture round-trip: round_name", row and row[3] == "League Phase", str(row))

    # I4: match_stats FK — inserting with non-existent fixture_id must be refused
    conn = sqlite3.connect(db)
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        conn.execute(
            "INSERT INTO match_stats VALUES (999, 12, 7, 5, 3, 60, 40, 1.8, 0.9, '2026-09-17T22:30:00Z')"
        )
        conn.commit()
        ok &= check("I4 FK constraint on match_stats", False, "insert succeeded — FK not enforced")
    except sqlite3.IntegrityError:
        ok &= check("I4 FK constraint on match_stats", True)
    finally:
        conn.close()

    # I5: teams.short_name UNIQUE
    conn = sqlite3.connect(db)
    try:
        conn.execute("INSERT INTO teams VALUES (3, 'Real Madrid B', 'RMA')")
        conn.commit()
        ok &= check("I5 teams.short_name UNIQUE enforced", False, "duplicate insert succeeded")
    except sqlite3.IntegrityError:
        ok &= check("I5 teams.short_name UNIQUE enforced", True)
    finally:
        conn.close()

    print("\n" + ("ALL PASS" if ok else "FAILURES PRESENT"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
