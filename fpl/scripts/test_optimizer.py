"""Quick end-to-end test: DC model + FPL optimizer."""
import sys
sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fpl.src.fpl_optimizer import optimise

print("Running FPL optimizer (full season baseline — no specific fixture)...")
result = optimise(budget=1000)

print(f"\n{'='*60}")
print(f"Optimal FPL Squad  |  Projected: {result['total_pts']} pts  |  Cost: £{result['total_cost']/10:.1f}m")
print(f"{'='*60}")

pos_order = ["GK", "DEF", "MID", "FWD"]
current_pos = None
for p in result["squad"]:
    if p["pos"] != current_pos:
        current_pos = p["pos"]
        print(f"\n  {current_pos}:")
    starter_mark = "★" if p["is_starter"] else " "
    cap_mark     = "(C)" if p["is_captain"] else "   "
    cs_mark      = f"CS={p.get('fixture_cs_pct', 0):.0f}%"
    print(f"  {starter_mark} {cap_mark} {p['name']:<22} {p['team']:<20} "
          f"£{p['price']/10:.1f}m  {p['projected_pts']:>5.1f}pt  {cs_mark}")

print(f"\n  Total cost: £{result['total_cost']/10:.1f}m  (remaining: £{result['budget_remaining']/10:.1f}m)")
print(f"  Projected pts (XI + captain bonus): {result['total_pts']}")

print("\n\nTop 20 players by projected pts:")
all_players = []
import sqlite3, json
from pathlib import Path as P2
from fpl.src.fpl_optimizer import (
    _load_players, _compute_player_rates, _compute_team_season_avgs,
    _project_players, _load_dc, DB_PATH
)
conn = sqlite3.connect(DB_PATH)
players = _load_players(conn)
rates   = _compute_player_rates(conn)
for p in players:
    p["_rates"] = rates.get(p["id"], {})
team_avgs = _compute_team_season_avgs(conn)
conn.close()
dc = _load_dc()
players = _project_players(players, dc, team_avgs)
players.sort(key=lambda x: x.get("projected_pts", 0), reverse=True)
for p in players[:20]:
    print(f"  {p['pos']:<4} {p['name']:<22} {p['team']:<20} "
          f"£{p['price']/10:.1f}m  proj={p['projected_pts']:.1f}  "
          f"cs={p.get('fixture_cs_pct',0):.0f}%  start={p.get('start_prob',0):.0f}")
