"""
poll_api_flip.py

Polls the FPL bootstrap-static endpoint once and exits:
  - exit 0, prints the GW1 deadline line → API has flipped to 26/27 (rollover is go)
  - exit 1, prints status → still showing last season (or endpoint is down)

Run daily (Task Scheduler / cron) until it exits 0 and fires an alert.
That exit 0 is the signal to begin the rollover cluster:
  season_rollover.py --label 2526 → train_dc.py --db data/fpl_2526.db → etc.

Example scheduler one-liner (pipe exit code to an alert):
  python scripts/poll_api_flip.py || echo "not yet" && echo "API IS LIVE — run rollover"
"""

import json
import sys
from datetime import datetime, timezone
from urllib.request import urlopen

URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
FLIP_YEAR = 2026   # GW1 deadline must be in this calendar year to count as flipped


def main() -> int:
    try:
        with urlopen(URL, timeout=15) as r:
            data = json.load(r)
    except Exception as exc:
        print(f"ENDPOINT ERROR: {exc}", file=sys.stderr)
        return 1

    events = data.get("events", [])
    if not events:
        print("ERROR: no events in response", file=sys.stderr)
        return 1

    gw1 = events[0]
    deadline_str = gw1.get("deadline_time", "")
    finished     = gw1.get("finished", True)
    name         = gw1.get("name", "GW?")

    try:
        deadline = datetime.fromisoformat(deadline_str.replace("Z", "+00:00"))
        deadline_year = deadline.year
    except (ValueError, AttributeError):
        deadline_year = 0

    flipped = (deadline_year >= FLIP_YEAR) and (not finished)

    status = "LIVE — rollover is go" if flipped else "not yet"
    print(f"{name} | deadline: {deadline_str} | finished: {finished} | {status}")

    return 0 if flipped else 1


if __name__ == "__main__":
    sys.exit(main())
