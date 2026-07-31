"""check_team_id_drift.py

Do the team ids in models/fpl_dc_params.json still mean what they meant when the
model was fitted?

    python scripts/check_team_id_drift.py

WHY THIS EXISTS

fpl_dc_params.json is fitted on the ARCHIVED season and keyed by that season's
team ids. Everything that consumes it - dc_predictor's fixture adjustments, the
published fixture grid - looks those ids up against the LIVE season's database.
FPL numbers teams alphabetically within a season, so three promotions and three
relegations reorder the list and the ids stop referring to the same clubs.

The result is silent: every team still resolves to a parameter, so nothing
errors. The parameters are just the wrong club's. A promoted side can inherit a
title challenger's attack rating and vice versa, and the only symptom is a
projection table that looks subtly implausible.

This is the same failure as the player-id reassignment already fixed in the
cold-start path, one level up. Exit 1 if any club's id has moved.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LIVE_DB = PROJECT_ROOT / "data" / "fpl.db"
DC_PARAMS = PROJECT_ROOT / "models" / "fpl_dc_params.json"


def teams_of(db: Path) -> dict[int, str]:
    conn = sqlite3.connect(db)
    try:
        return {int(r[0]): r[1] for r in
                conn.execute("SELECT team_id, short_name FROM teams")}
    finally:
        conn.close()


def main() -> None:
    archives = sorted((PROJECT_ROOT / "data").glob("fpl_*.db"))
    if not archives:
        raise SystemExit("no data/fpl_*.db archive found")
    archive = archives[-1]

    if not LIVE_DB.exists():
        raise SystemExit(f"missing {LIVE_DB}")
    if not DC_PARAMS.exists():
        raise SystemExit(f"missing {DC_PARAMS}")

    live = teams_of(LIVE_DB)
    old = teams_of(archive)
    params = json.loads(DC_PARAMS.read_text(encoding="utf-8"))
    tp = params.get("team_params", {})

    print(f"live      : {LIVE_DB.name}   ({len(live)} teams)")
    print(f"archive   : {archive.name}   ({len(old)} teams)")
    print(f"parameters: {len(tp)} team entries\n")

    # Detect format: new params are keyed by short_name (non-numeric strings);
    # old params were keyed by int-as-string ("1", "2", ...).
    keys = list(tp.keys())
    new_format = bool(keys) and not keys[0].lstrip("-").isdigit()

    if new_format:
        # Verify coverage: every live club should have a param or be a known
        # promoted side with no PL history.
        live_snames = set(live.values())
        archive_snames = set(old.values())
        promoted = live_snames - archive_snames   # no PL history — mean fallback expected
        missing_with_history = [
            sn for sn in live_snames - promoted if sn not in tp
        ]
        covered = [sn for sn in live_snames if sn in tp]
        missing_promoted = [sn for sn in promoted if sn not in tp]

        print(f"Format  : short_name keys (new format — cross-season safe)")
        print(f"Covered : {len(covered)}/{len(live_snames)} live clubs have parameters")
        if missing_promoted:
            print(f"No params (promoted, expected): "
                  + ", ".join(sorted(missing_promoted)))
        if missing_with_history:
            print(f"\nWARNING: {len(missing_with_history)} club(s) with PL history but no params:")
            for sn in sorted(missing_with_history):
                print(f"  {sn}")
            print("\nVERDICT: retrain to fill missing clubs.")
            sys.exit(1)
        print("\nVERDICT: parameters are keyed by short_name; every club with "
              "PL history has a param. Promoted sides will use the league mean.")
        return

    # Old int-string format — check for ID drift.
    moved, kept, gone, new_clubs = [], [], [], []
    for tid, name in sorted(live.items()):
        was = old.get(tid)
        if was is None:
            new_clubs.append((tid, name))
        elif was != name:
            moved.append((tid, was, name))
        else:
            kept.append((tid, name))

    for tid, name in sorted(old.items()):
        if name not in live.values():
            gone.append((tid, name))

    print(f"Format  : int-string keys (old format — vulnerable to ID drift)")
    print(f"{len(kept)} clubs kept their id")
    if moved:
        print(f"\n{len(moved)} ids now refer to a DIFFERENT club:")
        for tid, was, now in moved:
            a = tp.get(str(tid), {})
            print(f"  id {tid:>2}: was {was:<4} now {now:<4}   "
                  f"parameters attack={a.get('attack', float('nan')):.2f} "
                  f"defense={a.get('defense', float('nan')):.2f}  "
                  f"<- {now} is currently using {was}'s rating")
    if new_clubs:
        print(f"\n{len(new_clubs)} ids not present in the archive at all: "
              + ", ".join(f"{n} (id {t})" for t, n in new_clubs))
    if gone:
        print(f"\n{len(gone)} archived clubs no longer in the league: "
              + ", ".join(f"{n} (id {t})" for t, n in gone))

    if moved or new_clubs:
        print("\nVERDICT: the parameter file is keyed by ids that have shifted.")
        print("Every fixture adjustment built from it is using another club's")
        print("strength. Fix by keying team_params on short_name rather than")
        print("team_id, the same way player rates were fixed for the cold start,")
        print("and refit or remap before the next lock.")
        sys.exit(1)

    print("\nVERDICT: ids are stable; parameters refer to the clubs they were "
          "fitted on.")


if __name__ == "__main__":
    main()
