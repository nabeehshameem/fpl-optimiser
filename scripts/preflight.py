"""
preflight.py

Check everything the GW lock depends on, without locking anything.

    python scripts/preflight.py

Run it today, run it again at D-3, and let Task Scheduler run it weekly. Every
check here corresponds to something that would otherwise fail at D-10h, when
there is an alert and ten hours instead of weeks. Nothing is written; the
ledger, the export directory and the remote are all read-only here except for a
`git push --dry-run`, which authenticates without transferring anything.

WHAT THIS CANNOT CHECK

It cannot tell you whether the machine wakes from sleep, or whether git can
authenticate from a non-interactive Task Scheduler session. Those two are the
remaining untested failure modes and only the unattended dry run covers them --
see docs/dry_run_week.md, step 5. A green preflight is not a substitute.

Exit codes: 0 all clear (warnings allowed), 1 at least one FAIL.
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DB_PATH = Path(os.getenv("FPL_DB_PATH", PROJECT_ROOT / "data" / "fpl.db"))
EXPORT_DIR = Path(os.getenv("FPL_EXPORT_DIR", PROJECT_ROOT / "predictions" / "fpl"))
BRANCH = os.getenv("GIT_BRANCH", "main")

results: list[tuple[str, str, str]] = []


def ok(name, detail=""): results.append(("PASS", name, detail))
def warn(name, detail=""): results.append(("WARN", name, detail))
def fail(name, detail=""): results.append(("FAIL", name, detail))


def check_env() -> None:
    """Alerting is what turns a failed lock into a fixable problem."""
    if os.getenv("ALERT_WEBHOOK"):
        ok("ALERT_WEBHOOK set")
    else:
        fail("ALERT_WEBHOOK missing",
             "a failing lock would be silent -- set it in the Task Scheduler action")
    if os.getenv("HEARTBEAT_URL_LOCK") or os.getenv("HEARTBEAT_URL"):
        ok("lock heartbeat configured")
    else:
        warn("no lock heartbeat",
             "a lock that never RUNS cannot alert; healthchecks.io covers that")


def check_db() -> tuple[int, datetime] | None:
    if not DB_PATH.exists():
        fail("database missing", str(DB_PATH))
        return None
    conn = sqlite3.connect(DB_PATH)
    try:
        counts = {}
        for t in ("players", "teams", "fixtures", "gameweeks"):
            try:
                counts[t] = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            except sqlite3.Error:
                fail(f"table {t} unreadable")
                return None
        if counts["teams"] != 20:
            fail("teams table", f"{counts['teams']} teams, expected 20")
        else:
            ok("20 teams")
        if counts["players"] < 400:
            fail("players table", f"only {counts['players']} players")
        else:
            ok(f"{counts['players']} players")

        row = conn.execute(
            "SELECT gameweek_id, deadline_time FROM gameweeks WHERE is_next = 1"
        ).fetchone() or conn.execute(
            "SELECT gameweek_id, deadline_time FROM gameweeks "
            "WHERE finished = 0 ORDER BY gameweek_id LIMIT 1").fetchone()
        if not row:
            fail("no upcoming gameweek", "run ingest_bootstrap.py")
            return None
        gw = int(row[0])
        deadline = datetime.fromisoformat(str(row[1]).replace("Z", "+00:00"))
        hours = (deadline - datetime.now(timezone.utc)).total_seconds() / 3600
        if hours < 0:
            fail(f"GW{gw} deadline has passed", str(row[1]))
        else:
            ok(f"next is GW{gw}", f"{hours:.1f}h away ({row[1]})")

        n_fix = conn.execute(
            "SELECT COUNT(*) FROM fixtures WHERE gameweek_id = ?", (gw,)).fetchone()[0]
        if n_fix == 0:
            fail(f"no fixtures for GW{gw}", "run ingest_fixtures.py")
        else:
            ok(f"{n_fix} fixtures for GW{gw}")

        # Already locked? Not a failure -- the lock is idempotent -- but you want
        # to know before you go looking for a bug.
        try:
            locked = conn.execute(
                "SELECT locked_at_utc FROM model_squad_log WHERE gameweek_id = ?",
                (gw,)).fetchone()
            if locked:
                warn(f"GW{gw} already locked", f"at {locked[0]} -- lock is a no-op")
            else:
                ok(f"GW{gw} not yet locked")
        except sqlite3.Error:
            ok("ledger empty (first run)")

        return gw, deadline
    finally:
        conn.close()


def check_predictions(gw: int) -> None:
    """The lock reads only the bake-off winner's rows and fails loudly without
    them -- better to discover that now than at D-10h."""
    try:
        from scripts.lock_model_squad import MODEL_NAME
    except Exception as exc:
        fail("cannot import lock job", str(exc)[:90])
        return
    conn = sqlite3.connect(DB_PATH)
    try:
        n = conn.execute(
            "SELECT COUNT(DISTINCT player_id) FROM predictions "
            "WHERE gameweek_id = ? AND model_name = ?", (gw, MODEL_NAME)).fetchone()[0]
    except sqlite3.Error as exc:
        fail("predictions table unreadable", str(exc)[:80])
        return
    finally:
        conn.close()
    if n == 0:
        fail(f"no {MODEL_NAME} predictions for GW{gw}", "run run_predictions.py")
    elif n < 300:
        warn(f"only {n} predicted players for GW{gw}", "expected several hundred")
    else:
        ok(f"{n} {MODEL_NAME} predictions for GW{gw}")


def check_models() -> None:
    dc = PROJECT_ROOT / "models" / "fpl_dc_params.json"
    if dc.exists():
        ok("DC parameters present")
    else:
        fail("models/fpl_dc_params.json missing", "run train_dc.py --db <archive>")

    # Cold start reads the archive while the live DB has <3 gameweeks.
    conn = sqlite3.connect(DB_PATH) if DB_PATH.exists() else None
    live_gws = 0
    if conn:
        try:
            live_gws = conn.execute(
                "SELECT COUNT(DISTINCT gameweek_id) FROM player_gameweek_history"
            ).fetchone()[0] or 0
        except sqlite3.Error:
            pass
        finally:
            conn.close()
    archives = sorted((PROJECT_ROOT / "data").glob("fpl_*.db"))
    if live_gws < 3:
        if archives:
            ok("cold start ready", f"{live_gws} live GWs, archive {archives[-1].name}")
        else:
            fail("cold start impossible",
                 f"{live_gws} live GWs and no data/fpl_*.db archive -- every "
                 "player would project 0.0")
    else:
        ok("warm start", f"{live_gws} gameweeks of live history")


def check_exclusions() -> None:
    """The comments in the exclusions file are assertions; verify them now."""
    try:
        from src.availability import validate_exclusions
        entries = validate_exclusions(DB_PATH)
        ok("exclusions validate", f"{len(entries)} entries match the database")
    except Exception as exc:
        fail("exclusions do not match the database",
             str(exc).replace("\n", " ")[:120])


def check_export_dir() -> None:
    try:
        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        probe = EXPORT_DIR / ".preflight"
        probe.write_text("ok")
        probe.unlink()
        ok("export directory writable", str(EXPORT_DIR))
    except Exception as exc:
        fail("export directory not writable", f"{EXPORT_DIR}: {exc}")


def check_git() -> None:
    def git(*a):
        return subprocess.run(["git", *a], cwd=PROJECT_ROOT,
                              capture_output=True, text=True)

    r = git("rev-parse", "--abbrev-ref", "HEAD")
    if r.returncode != 0:
        fail("not a git repository")
        return
    ok("on branch", r.stdout.strip())

    r = git("status", "--porcelain")
    dirty = [l for l in r.stdout.splitlines() if l and not l.startswith("??")]
    if dirty:
        warn(f"{len(dirty)} modified tracked file(s)",
             "the lock commits only its export, but review before the deadline")
    else:
        ok("working tree clean")

    # Authenticates against the remote without transferring anything. This is
    # the check most likely to catch a rotten credential.
    r = git("push", "--dry-run", "origin", f"HEAD:{BRANCH}")
    if r.returncode == 0:
        ok(f"push to origin/{BRANCH} authenticates", "--dry-run succeeded")
    else:
        fail(f"cannot push to origin/{BRANCH}",
             (r.stderr or r.stdout).strip().splitlines()[-1][:110]
             if (r.stderr or r.stdout).strip() else "unknown error")


def _safe(text: str) -> str:
    """Strip characters the console may not be able to encode.

    Upstream messages contain em-dashes and curly quotes. On a Windows console
    still defaulting to cp1252, printing those raises UnicodeEncodeError -- so
    the check that exists to reassure you would itself crash, and on the one
    machine that actually runs the lock. Downgrade to ASCII rather than risk it.
    """
    return (text.replace("—", "-").replace("–", "-")
                .replace("’", "'").replace("“", '"')
                .replace("”", '"').replace("→", "->")
                .encode("ascii", "replace").decode("ascii"))


def main() -> None:
    print("=== lock preflight ===\n")
    check_env()
    gwinfo = check_db()
    if gwinfo:
        check_predictions(gwinfo[0])
    check_models()
    check_exclusions()
    check_export_dir()
    check_git()

    width = max(len(n) for _, n, _ in results) + 2
    for status, name, detail in results:
        print(_safe(f"[{status}] {name:<{width}} {detail}"))

    fails = sum(1 for s, _, _ in results if s == "FAIL")
    warns = sum(1 for s, _, _ in results if s == "WARN")
    print(f"\n{len(results)} checks  {fails} fail  {warns} warn")
    if fails:
        print(_safe(
            "\nFix the failures before the deadline. Reminder: a clean "
            "preflight still does not prove the machine wakes from sleep or "
            "that git authenticates from a non-interactive scheduled task - "
            "only the unattended dry run covers those."))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
