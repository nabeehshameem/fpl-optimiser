"""
export_projections.py

Export predictions/fpl/gwNN_projections.json independently of the squad lock.

    python scripts/export_projections.py          # next GW
    python scripts/export_projections.py --gw 3   # specific GW
    python scripts/export_projections.py --dry-run

Run during the refresh phase (D-24h) so ProjectionsPanel shows data before
the lock. Harmless to re-run: overwrites the same file with the latest
predictions run.

Rule 8: Railway has no DB — everything is computed here, served from committed JSON.
Rule 5: failures are non-fatal (warn_step in weekly_ops.py) — a failed projections
export must never block the daily refresh or the lock. The lock rebuilds projections
independently anyway; this script closes the gap in the days before lock day.

No ledger writes. No squad optimisation. Cannot be confused with a lock.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import scripts.lock_model_squad as lk  # noqa: E402
from src.availability import get_excluded_ids  # noqa: E402

DB_PATH = Path(os.getenv("FPL_DB_PATH", PROJECT_ROOT / "data" / "fpl.db"))
EXPORT_DIR = Path(os.getenv("FPL_EXPORT_DIR", PROJECT_ROOT / "predictions" / "fpl"))


def export(
    gw: int | None = None,
    db_path: Path = DB_PATH,
    export_dir: Path = EXPORT_DIR,
    dry_run: bool = False,
) -> dict:
    """Build and write (or print) the projections JSON.

    Uses the same build_projections() and load_predictions() as the lock so
    the pre-lock projections come from exactly the same computation path — no
    divergence between the early publish and the lock-time republish.
    """
    conn = sqlite3.connect(db_path)
    try:
        if gw is None:
            gw, deadline = lk.next_gameweek(conn)
        else:
            row = conn.execute(
                "SELECT deadline_time FROM gameweeks WHERE gameweek_id = ?", (gw,)
            ).fetchone()
            deadline = str(row[0]) if row else "unknown"
        preds = lk.load_predictions(conn, gw)
    finally:
        conn.close()

    try:
        excluded = get_excluded_ids(db_path, gw)
    except Exception as exc:
        print(f"[WARN] exclusions skipped — running without: {exc}", file=sys.stderr)
        excluded = set()

    if excluded:
        before = len(preds)
        preds = preds[~preds["player_id"].isin(excluded)].copy()
        print(f"  Availability filter: {before - len(preds)} players removed "
              f"({len(preds)} eligible)")

    proj = lk.build_projections(gw, deadline, preds, excluded, conn_for_meta=db_path)

    if dry_run:
        print(json.dumps(proj, indent=2))
        return proj

    export_dir.mkdir(parents=True, exist_ok=True)
    out = export_dir / f"gw{gw:02d}_projections.json"
    out.write_text(json.dumps(proj, indent=2))
    print(f"GW{gw} projections exported -> {out}")
    return proj


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gw", type=int, default=None,
                    help="Gameweek number (default: next upcoming)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print JSON without writing to disk")
    args = ap.parse_args()
    export(gw=args.gw, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
