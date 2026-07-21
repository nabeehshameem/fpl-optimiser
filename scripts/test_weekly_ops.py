"""
Tests for scripts/weekly_ops.py against a throwaway git repo with a real
bare "origin". Run: python scripts/test_weekly_ops.py

O1  commit_push_verify: happy path — commit reaches origin, sha matches
O2  UNVERIFIED PUSH: push "succeeds" but the remote does not advance ->
    must exit non-zero and alert CRITICAL (the session's core lesson,
    encoded: a green push exit code is not proof)
O3  Missing export file -> refuses rather than committing nothing
O4  Idempotent: staging an already-committed file is a no-op, still verified
O5  phase_lock refuses after the deadline and says UNVERIFIED, not "retry"
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import scripts.weekly_ops as ops  # noqa: E402


def sh(*args, cwd):
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True)


def make_repo() -> tuple[Path, Path]:
    """(worktree, bare_origin) with one commit already pushed."""
    root = Path(tempfile.mkdtemp())
    bare = root / "origin.git"
    work = root / "work"
    sh("git", "init", "--bare", "-b", "main", str(bare), cwd=root)
    sh("git", "init", "-b", "main", str(work), cwd=root)
    work.mkdir(exist_ok=True)
    sh("git", "config", "user.email", "t@t.t", cwd=work)
    sh("git", "config", "user.name", "T", cwd=work)
    sh("git", "remote", "add", "origin", str(bare), cwd=work)
    (work / "README").write_text("x")
    sh("git", "add", "README", cwd=work)
    sh("git", "commit", "-m", "init", cwd=work)
    sh("git", "push", "origin", "main", cwd=work)
    return work, bare


def check(label, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {label}" + (f"  ({detail})" if detail else ""))
    return cond


def main():
    ok = True
    alerts: list[tuple] = []
    ops.alert = lambda p, s, m: alerts.append((p, s, m))

    # ── O1 happy path ────────────────────────────────────────────────────
    work, bare = make_repo()
    ops.PROJECT_ROOT = work
    ops.BRANCH = "main"
    export = work / "predictions" / "fpl" / "gw01.json"
    export.parent.mkdir(parents=True)
    export.write_text('{"gameweek": 1, "squad_hash": "abc"}')

    sha = ops.commit_push_verify([export], "GW1 lock", "lock")
    remote = sh("git", "ls-remote", str(bare), "main", cwd=work).stdout.split()[0]
    ok &= check("O1 commit verified on origin", sha == remote, f"{sha[:9]}")

    # ── O2 the silent failure: push exits 0 but remote never advances ────
    export.write_text('{"gameweek": 2, "squad_hash": "def"}')
    real_run = subprocess.run

    def fake_run(cmd, *a, **kw):
        if isinstance(cmd, list) and len(cmd) > 1 and cmd[1] == "push":
            class R:  # pretends the push worked
                returncode, stdout, stderr = 0, "", ""
            return R()
        return real_run(cmd, *a, **kw)

    alerts.clear()
    subprocess.run = fake_run
    try:
        ops.commit_push_verify([export], "GW2 lock", "lock")
        ok &= check("O2 unverified push exits non-zero", False, "returned!")
    except SystemExit as e:
        crit = [a for a in alerts if a[1] == "CRITICAL"]
        ok &= check("O2 unverified push exits non-zero", e.code == 1)
        ok &= check("O2 alerts CRITICAL with UNVERIFIED",
                    bool(crit) and "UNVERIFIED" in crit[0][2],
                    crit[0][2][:60] if crit else "no alert")
    finally:
        subprocess.run = real_run

    # ── O3 missing export ────────────────────────────────────────────────
    alerts.clear()
    try:
        ops.commit_push_verify([work / "nope.json"], "x", "lock")
        ok &= check("O3 missing export refused", False, "returned!")
    except SystemExit as e:
        ok &= check("O3 missing export refused", e.code == 1
                    and "missing" in alerts[0][2])

    # ── O4 idempotent re-verify ──────────────────────────────────────────
    work2, bare2 = make_repo()
    ops.PROJECT_ROOT = work2
    e2 = work2 / "predictions" / "fpl" / "gw03.json"
    e2.parent.mkdir(parents=True)
    e2.write_text("{}")
    first = ops.commit_push_verify([e2], "GW3 lock", "lock")
    again = ops.commit_push_verify([e2], "GW3 lock (re-verify)", "lock")
    ok &= check("O4 re-run is a verified no-op", first == again, f"{again[:9]}")

    # ── O5 passed deadline ───────────────────────────────────────────────
    import sqlite3
    from datetime import datetime, timedelta, timezone
    db = Path(tempfile.mkdtemp()) / "fpl.db"
    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE TABLE gameweeks (gameweek_id INT PRIMARY KEY, deadline_time TEXT,
            is_current INT, is_next INT, finished INT, average_score INT);
        CREATE TABLE model_squad_log (gameweek_id INT PRIMARY KEY,
            locked_at_utc TEXT, deadline_utc TEXT, squad_json TEXT,
            transfers_json TEXT, free_transfers INT, bank INT,
            expected_points REAL, squad_hash TEXT);
    """)
    past = (datetime.now(timezone.utc) - timedelta(hours=3)).strftime(
        "%Y-%m-%dT%H:%M:%SZ")
    conn.execute("INSERT INTO gameweeks VALUES (1,?,0,1,0,NULL)", (past,))
    conn.commit(); conn.close()

    import scripts.lock_model_squad as lk
    lk.DB_PATH = db
    alerts.clear()
    try:
        ops.phase_lock()
        ok &= check("O5 post-deadline lock refused", False, "returned!")
    except SystemExit as e:
        msg = alerts[-1][2] if alerts else ""
        ok &= check("O5 post-deadline lock refused", e.code == 1)
        ok &= check("O5 directs to UNVERIFIED, not backfill",
                    "UNVERIFIED" in msg and "backfill" in msg, msg[:70])

    print("\n" + ("ALL PASS" if ok else "FAILURES PRESENT"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
