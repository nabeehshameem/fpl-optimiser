"""
Archive-and-reset for a new FPL season. One command, guarded at every step:

  1. ARCHIVE  data/fpl.db -> data/fpl_<label>.db  (refuses to overwrite;
              refuses to proceed if the archive copy fails verification)
  2. INIT     fresh schema via scripts/init_db.py
  3. INGEST   bootstrap -> fixtures (subprocess, same as run_gw.py)
  4. VERIFY   the new DB actually looks like a new season:
                - 20 teams
                - 38 gameweeks, GW1 deadline in the future
                - player_gameweek_history EMPTY (it's a new season)
                - players ingested (> 400)
                - model artifacts still present in models/
     A failed verification names the failure and points at the archive —
     nothing is deleted at any point, so rollback is `mv` back.

Usage:
  python scripts/season_rollover.py --label 2526          # archive as fpl_2526.db
  python scripts/season_rollover.py --label 2526 --no-ingest   # steps 1-2 only

After a successful rollover:
  * The 1001+ upcoming_fixtures convention is obsolete — new-season fixtures
    live in `fixtures` with honest gameweek IDs. Delete the workaround.
  * Retrain/refit against the ARCHIVE for pre-season models; the fresh DB has
    no history until GW1 completes (lock_model_squad.py's max_hist_gw < 3 gate
    handles the cold start).
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DB = DATA_DIR / "fpl.db"
MODELS = ["models/lightgbm_v1.txt", "models/fpl_dc_params.json"]
INGEST_SEQUENCE = ["scripts/ingest_bootstrap.py", "scripts/ingest_fixtures.py"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fail(msg: str) -> None:
    print(f"\nROLLOVER FAILED: {msg}")
    sys.exit(1)


def step_archive(label: str) -> Path:
    if not DB.exists():
        fail(f"{DB} does not exist — nothing to archive. "
             "If this is a genuinely fresh machine, run init_db + ingest directly.")
    archive = DATA_DIR / f"fpl_{label}.db"
    if archive.exists():
        fail(f"{archive} already exists — refusing to overwrite an archive. "
             "Pick a different --label or move the existing file yourself.")
    src_hash = sha256(DB)
    shutil.copy2(DB, archive)
    if sha256(archive) != src_hash:
        fail("archive copy hash mismatch — disk problem? Old DB untouched.")
    print(f"[1/4] Archived {DB.name} -> {archive.name} (sha256 verified)")
    return archive


def step_init() -> None:
    DB.unlink()  # safe: archive verified above
    r = subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "init_db.py")],
                       cwd=PROJECT_ROOT)
    if r.returncode != 0 or not DB.exists():
        fail("init_db.py failed — restore with: "
             "mv data/fpl_<label>.db data/fpl.db")
    print("[2/4] Fresh schema initialised")


def step_ingest() -> None:
    for script in INGEST_SEQUENCE:
        print(f"      running {script} ...")
        r = subprocess.run([sys.executable, str(PROJECT_ROOT / script)],
                           cwd=PROJECT_ROOT)
        if r.returncode != 0:
            fail(f"{script} failed. Old season is safe in the archive; "
                 "fix the ingest and re-run it directly (no need to re-archive).")
    print("[3/4] Ingest sequence complete")


def step_verify(ingested: bool) -> None:
    conn = sqlite3.connect(DB)
    problems = []

    def q(sql):
        return conn.execute(sql).fetchone()[0]

    hist = q("SELECT COUNT(*) FROM player_gameweek_history")
    if hist != 0:
        problems.append(f"player_gameweek_history has {hist} rows — a fresh "
                        "season must start empty; the reset did not happen.")

    if ingested:
        teams = q("SELECT COUNT(*) FROM teams")
        if teams != 20:
            problems.append(f"{teams} teams (expected 20)")
        gws = q("SELECT COUNT(*) FROM gameweeks")
        if gws != 38:
            problems.append(f"{gws} gameweeks (expected 38)")
        players = q("SELECT COUNT(*) FROM players")
        if players < 400:
            problems.append(f"only {players} players ingested")
        row = conn.execute(
            "SELECT deadline_time FROM gameweeks WHERE gameweek_id = 1"
        ).fetchone()
        if not row or not row[0]:
            problems.append("GW1 has no deadline_time")
        else:
            dl = datetime.fromisoformat(str(row[0]).replace("Z", "+00:00"))
            if dl <= datetime.now(timezone.utc):
                problems.append(
                    f"GW1 deadline {row[0]} is in the past — this looks like "
                    "LAST season's bootstrap. Check the FPL API has switched over.")
        fixtures = q("SELECT COUNT(*) FROM fixtures")
        if fixtures < 380:
            problems.append(f"only {fixtures} fixtures (expected 380)")

    conn.close()
    for m in MODELS:
        if not (PROJECT_ROOT / m).exists():
            problems.append(f"model artifact missing: {m}")

    if problems:
        fail("verification failed:\n  - " + "\n  - ".join(problems)
             + "\nOld season is intact in the archive.")
    print("[4/4] Verification passed — new season DB is live.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True,
                    help="archive label, e.g. 2526 -> data/fpl_2526.db")
    ap.add_argument("--no-ingest", action="store_true",
                    help="archive + init only; run ingests manually after")
    args = ap.parse_args()

    step_archive(args.label)
    step_init()
    if not args.no_ingest:
        step_ingest()
    step_verify(ingested=not args.no_ingest)

    print("\nDone. Next: refit DC params and confirm the LightGBM artifact "
          "loads, then delete the upcoming_fixtures workaround "
          "(ingest_fixtures_2627.py) — it is obsolete after rollover.")


if __name__ == "__main__":
    main()
