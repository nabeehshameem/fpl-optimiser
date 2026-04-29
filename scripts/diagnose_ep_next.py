"""
diagnose_ep_next.py
Check the dtype distribution of ep_next in player_snapshots.
"""

import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "fpl.db"

conn = sqlite3.connect(DB_PATH)
print("typeof(ep_next) distribution across player_snapshots:")
for row in conn.execute("SELECT typeof(ep_next), COUNT(*) FROM player_snapshots GROUP BY typeof(ep_next)"):
    print(f"  {row[0]}: {row[1]}")

print("\nSample of distinct ep_next values stored:")
for row in conn.execute("SELECT DISTINCT ep_next FROM player_snapshots LIMIT 15"):
    print(f"  {row[0]!r} (type: {type(row[0]).__name__})")

print("\ntypeof(ep_next) by gameweek:")
for row in conn.execute("""
    SELECT gameweek_id, typeof(ep_next), COUNT(*)
    FROM player_snapshots
    GROUP BY gameweek_id, typeof(ep_next)
    ORDER BY gameweek_id
"""):
    print(f"  GW{row[0]}: {row[1]} ({row[2]} rows)")

conn.close()