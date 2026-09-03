"""
correct_model_gw.py

Amend a published GW result file in a traceable, append-only way.

Rule 11: published artifacts are never silently replaced. This script is
the ONLY sanctioned way to change a value in an already-published result.
It records every amendment in the corrections array so any reader can see
exactly what changed, why, and which commit introduced the fix.

Usage:
  python scripts/correct_model_gw.py --gw 1 \\
      --field net_points --from 51 --to 50 \\
      --reason "Grader used wrong bench order; Calvert-Lewin subbed in, not Groß" \\
      --commit 33dda99

  python scripts/correct_model_gw.py --gw 1 --dry-run ...

Limits:
  - Refuses if the field already has the target value (no-op correction).
  - Refuses if corrections would exceed 3 entries for one GW. Three
    amendments to one result means the grader is wrong, not the data.
  - Does NOT touch the SQLite ledger (which records the grader's original
    calculation). The correction is an amendment to the published JSON only.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

EXPORT_DIR = Path(os.getenv("FPL_EXPORT_DIR", PROJECT_ROOT / "predictions" / "fpl"))
MAX_CORRECTIONS = 3


def correct(gw: int, field: str, from_val, to_val, reason: str,
            commit: str, dry_run: bool = False) -> None:
    path = EXPORT_DIR / f"gw{gw:02d}_result.json"
    if not path.exists():
        sys.exit(f"ERROR: {path} does not exist — cannot correct an unpublished result.")

    result = json.loads(path.read_text(encoding="utf-8"))

    existing = result.get("corrections", [])
    if len(existing) >= MAX_CORRECTIONS:
        sys.exit(
            f"ERROR: GW{gw} already has {len(existing)} corrections "
            f"(max {MAX_CORRECTIONS}). This many amendments indicate a "
            f"systematic grader problem — fix the grader and re-grade from "
            f"scratch rather than patching the published artifact further."
        )

    current = result.get(field)
    if current == to_val:
        sys.exit(
            f"ERROR: {field} is already {to_val!r} — no correction needed."
        )
    if current != from_val:
        sys.exit(
            f"ERROR: {field} is {current!r}, not {from_val!r}. "
            f"Check --from matches the current published value."
        )

    entry = {
        "revised_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "field": field,
        "from": from_val,
        "to": to_val,
        "reason": reason,
        "commit": commit,
    }

    if dry_run:
        print("DRY RUN — would write:")
        print(json.dumps(entry, indent=2))
        print(f"\n{field}: {from_val!r} → {to_val!r}")
        return

    result[field] = to_val
    result.setdefault("corrections", []).append(entry)

    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"GW{gw} corrected: {field} {from_val!r} → {to_val!r}")
    print(f"Correction recorded in {path.name}. Commit and push to publish.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gw", type=int, required=True)
    ap.add_argument("--field", required=True, help="Top-level field to correct")
    ap.add_argument("--from", dest="from_val", required=True,
                    help="Current published value (must match exactly)")
    ap.add_argument("--to", dest="to_val", required=True,
                    help="Corrected value")
    ap.add_argument("--reason", required=True, help="Plain-English explanation")
    ap.add_argument("--commit", required=True,
                    help="Short hash of the commit that introduced the fix")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    def _cast(v: str):
        try:
            return int(v)
        except ValueError:
            try:
                return float(v)
            except ValueError:
                return v

    correct(
        gw=args.gw,
        field=args.field,
        from_val=_cast(args.from_val),
        to_val=_cast(args.to_val),
        reason=args.reason,
        commit=args.commit,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
