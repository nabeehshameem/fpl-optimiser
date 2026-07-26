"""
weekly_ops.py

The scheduled path through a gameweek. Cron runs this; you read the alerts.

    python scripts/weekly_ops.py refresh   # D-24h: ingest + predictions
    python scripts/weekly_ops.py lock      # D-10h: lock, commit, push, VERIFY
    python scripts/weekly_ops.py grade     # GW end: grade, commit, push, VERIFY

Design rules, each earned the hard way:

1. A PUSH IS NOT DONE UNTIL THE REMOTE SAYS SO.
   Every git phase ends with `git ls-remote origin <branch>` compared against
   local HEAD. "Committed" and "pushed" are different states and only the
   second one creates a public timestamp. A commit that didn't reach origin
   is reported as CRITICAL, never as success.

2. IDEMPOTENT. Cron retries, machines sleep through schedules, you re-run by
   hand at 3am. Re-running a completed phase exits 0 with "already done"
   rather than erroring or double-writing. The ledger's append-only guards
   are the backstop; this is the polite layer above them.

3. FAILURES ARE LOUD AND SPECIFIC. Non-zero exit (so cron's MAILTO fires),
   stderr detail, and an optional webhook via ALERT_WEBHOOK. Severity is in
   the message: a failed refresh has 14 hours of slack; a failed lock has a
   deadline.

Environment:
    ALERT_WEBHOOK   optional; POSTed a JSON {phase, severity, message}
    GIT_BRANCH      default "main"
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

EXPORT_DIR = Path(os.getenv("FPL_EXPORT_DIR", PROJECT_ROOT / "predictions" / "fpl"))
BRANCH = os.getenv("GIT_BRANCH", "main")
ALERT_WEBHOOK = os.getenv("ALERT_WEBHOOK")
# Vercel deploy hook URL. POST triggers a rebuild so the prerendered HTML
# gains the new graded gameweek. Get it from Vercel Project Settings →
# Git → Deploy Hooks, add to Railway env alongside ALERT_WEBHOOK.
VERCEL_DEPLOY_HOOK = os.getenv("VERCEL_DEPLOY_HOOK")

# Dead-man's switch. An on-failure alert only fires if the job RUNS; it is
# structurally blind to the likeliest failure of all — the job never
# executing (machine asleep, scheduler misconfigured, laptop shut). Set
# HEARTBEAT_URL_<PHASE> to a healthchecks.io-style check URL and the phase
# pings it on success; if the ping doesn't arrive, that service alerts you.
def heartbeat_url(phase: str) -> str | None:
    return os.getenv(f"HEARTBEAT_URL_{phase.upper()}") or os.getenv("HEARTBEAT_URL")


def heartbeat(phase: str) -> None:
    """Ping the dead-man's switch. Never let this failure mask a real success."""
    url = heartbeat_url(phase)
    if not url:
        return
    try:
        urllib.request.urlopen(url, timeout=10)
        print(f"heartbeat sent ({phase})")
    except Exception as exc:
        print(f"[WARN] heartbeat failed: {exc}", file=sys.stderr)


def trigger_vercel_rebuild() -> None:
    """POST to the Vercel deploy hook so the prerendered HTML gains the new result.

    Never let a failed hook mask a successful grade — log and move on.
    """
    if not VERCEL_DEPLOY_HOOK:
        return
    try:
        req = urllib.request.Request(
            VERCEL_DEPLOY_HOOK, data=b"",
            headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=15)
        print("Vercel rebuild triggered")
    except Exception as exc:
        print(f"[WARN] Vercel deploy hook failed: {exc}", file=sys.stderr)


REFRESH_STEPS = [
    "scripts/ingest_bootstrap.py",
    "scripts/ingest_fixtures.py",
    "scripts/run_predictions.py",
]
GRADE_INGEST_STEPS = [
    "scripts/ingest_bootstrap.py",   # refreshes gameweeks.finished
    "scripts/backfill_history.py",   # the GW's player results
]


# ── reporting ────────────────────────────────────────────────────────────────

def alert(phase: str, severity: str, message: str) -> None:
    line = f"[{severity}] {phase}: {message}"
    print(line, file=sys.stderr)
    if not ALERT_WEBHOOK:
        return
    try:
        payload = json.dumps({
            "phase": phase, "severity": severity, "message": message,
            "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }).encode()
        req = urllib.request.Request(
            ALERT_WEBHOOK, data=payload,
            headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10)
    except Exception as exc:  # an alert failing must not mask the real error
        print(f"[WARN] alert webhook failed: {exc}", file=sys.stderr)


def die(phase: str, message: str, severity: str = "CRITICAL") -> None:
    alert(phase, severity, message)
    sys.exit(1)


# ── shell helpers ────────────────────────────────────────────────────────────

def run_step(script: str, phase: str, timeout: int = 1800,
             args: list[str] | None = None) -> None:
    print(f"--- {script} {' '.join(args or [])}")
    try:
        r = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / script)] + (args or []),
            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        die(phase, f"{script} timed out after {timeout}s")
    if r.stdout:
        print(r.stdout.rstrip())
    if r.returncode != 0:
        die(phase, f"{script} exited {r.returncode}: "
                   f"{(r.stderr or r.stdout).strip()[-500:]}")


def git(*args: str, check: bool = True) -> str:
    r = subprocess.run(["git", *args], cwd=PROJECT_ROOT,
                       capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {r.stderr.strip()}")
    return r.stdout.strip()


def commit_push_verify(paths: list[Path], message: str, phase: str) -> str:
    """Stage, commit, push, then PROVE the remote has it. Returns commit sha."""
    def _rel(p: Path) -> str:
        try:
            return str(p.resolve().relative_to(PROJECT_ROOT))
        except ValueError:
            raise ValueError(
                f"{p} is outside the project root — FPL_EXPORT_DIR must be a "
                f"subdirectory of the project for git to stage it. "
                f"Use e.g. predictions/fpl-dryrun/ for dry runs."
            ) from None
    existing = [_rel(p) for p in paths if p.exists()]
    if not existing:
        die(phase, f"expected export file(s) missing: "
                   f"{[str(p) for p in paths]} — nothing to publish")

    git("add", *existing)
    if not git("diff", "--cached", "--name-only"):
        print("nothing staged (files already committed)")
    else:
        git("commit", "-m", message)

    local = git("rev-parse", "HEAD")

    r = subprocess.run(["git", "push", "origin", f"HEAD:{BRANCH}"],
                       cwd=PROJECT_ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        die(phase, f"PUSH FAILED — commit {local[:9]} is LOCAL ONLY and no "
                   f"public timestamp exists: {r.stderr.strip()[-300:]}")

    # The whole point. Trust the remote, not the exit code of push.
    remote_line = git("ls-remote", "origin", BRANCH)
    remote = remote_line.split()[0] if remote_line else ""
    if remote != local:
        die(phase, f"PUSH UNVERIFIED — local HEAD {local[:9]} but "
                   f"origin/{BRANCH} is {remote[:9] or 'EMPTY'}. Treat the "
                   f"public record as NOT published.")
    print(f"verified on origin/{BRANCH}: {local[:9]}")
    return local


# ── phases ───────────────────────────────────────────────────────────────────

def phase_refresh() -> None:
    for s in REFRESH_STEPS:
        run_step(s, "refresh")
    print("\nrefresh OK — predictions written for the next gameweek")
    heartbeat("refresh")


def phase_lock() -> None:
    import scripts.lock_model_squad as lk

    conn_gw = None
    try:
        import sqlite3
        conn = sqlite3.connect(lk.DB_PATH)
        lk.ensure_ledger(conn)
        conn_gw, deadline = lk.next_gameweek(conn)
        already = conn.execute(
            "SELECT locked_at_utc FROM model_squad_log WHERE gameweek_id = ?",
            (conn_gw,)).fetchone()
        conn.close()
    except Exception as exc:
        die("lock", f"could not read ledger state: {exc}")

    export = EXPORT_DIR / f"gw{conn_gw:02d}.json"

    if already:
        print(f"GW{conn_gw} already locked at {already[0]} — idempotent no-op")
        # Still verify the commitment actually reached origin.
        commit_push_verify([export], f"GW{conn_gw} lock (re-verify)", "lock")
        return

    deadline_dt = datetime.fromisoformat(str(deadline).replace("Z", "+00:00"))
    remaining = (deadline_dt - datetime.now(timezone.utc)).total_seconds() / 3600
    if remaining <= 0:
        die("lock", f"GW{conn_gw} deadline ({deadline}) has PASSED. Do not "
                    f"backfill a timestamp — mark the gameweek UNVERIFIED per "
                    f"the runbook and publish the reason.")
    if remaining < 2:
        alert("lock", "WARNING",
              f"only {remaining:.1f}h to the GW{conn_gw} deadline")

    run_step("scripts/lock_model_squad.py", "lock", timeout=600)
    sha = commit_push_verify([export], f"GW{conn_gw} lock", "lock")
    print(f"\nlock OK — GW{conn_gw} commitment public at {sha[:9]}, "
          f"{remaining:.1f}h before deadline")
    heartbeat("lock")


def phase_grade() -> None:
    import sqlite3
    import scripts.lock_model_squad as lk

    import scripts.grade_model_gw as grd
    conn = sqlite3.connect(lk.DB_PATH)
    conn.execute(grd.RESULTS_SQL)
    row = conn.execute("""
        SELECT l.gameweek_id FROM model_squad_log l
        JOIN gameweeks g ON g.gameweek_id = l.gameweek_id
        WHERE g.finished = 1
          AND l.gameweek_id NOT IN (SELECT gameweek_id FROM model_gw_results)
        ORDER BY l.gameweek_id DESC LIMIT 1
    """).fetchone()
    conn.close()

    if row is None:
        # Not an error: the GW isn't finished yet, or it's already graded.
        print("no finished, locked, ungraded gameweek — nothing to do")
        return

    gw = int(row[0])
    for s in GRADE_INGEST_STEPS:
        run_step(s, "grade")
    run_step("scripts/grade_model_gw.py", "grade", args=["--gw", str(gw)])

    result = EXPORT_DIR / f"gw{gw:02d}_result.json"
    sha = commit_push_verify([result], f"GW{gw} graded", "grade")
    print(f"\ngrade OK — GW{gw} result public at {sha[:9]}")
    trigger_vercel_rebuild()
    heartbeat("grade")


PHASES = {"refresh": phase_refresh, "lock": phase_lock, "grade": phase_grade}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=sorted(PHASES))
    args = ap.parse_args()
    started = datetime.now(timezone.utc)
    print(f"=== weekly_ops {args.phase} @ {started:%Y-%m-%d %H:%M UTC} ===")
    PHASES[args.phase]()


if __name__ == "__main__":
    main()
