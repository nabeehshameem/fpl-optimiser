"""
Correctness tests for SquadOptimiser.optimise_with_transfers() that need NO
real fpl.db — a synthetic database with hand-built known-optimal answers.
Run anywhere: python scripts/test_transfers_synthetic.py

Scenarios:
  1. Optimal squad, 1 FT            -> expect 0 transfers
  2. Bad starter, upgrade +10, 1 FT -> expect exactly that swap
  3. Upgrade +3, 0 FT (hit = 4)     -> expect 0 transfers (3 < 4 not worth it)
  4. Upgrade +6, 0 FT (hit = 4)     -> expect the hit taken (6 > 4)
  5. Budget: best upgrade unaffordable -> expect cheaper alternative chosen
  6. Club cap: upgrade would make 4 from one club -> expect it blocked
  7. Selling prices: risen-price player retained, bank non-negative
  8. Horizon=6: amortised hit taken when bare hit would be refused
"""

import sqlite3
import sys
import tempfile
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.optimiser import SquadOptimiser  # noqa: E402

# ── synthetic world ──────────────────────────────────────────────────────────
# 10 teams (t1..t10). Costs in tenths of £1m.
# Squad players (ids 1-15) all cost 50. Pool players 100+.

SQUAD = list(range(1, 16))
# (player_id, name, position, team_id, cost, predicted_points)
# positions: 1 GK, 2 DEF, 3 MID, 4 FWD
BASE_PLAYERS = [
    # current squad — spread across teams 1-8, XI-calibre points
    (1,  "GK_A",  1, 1, 50, 5.0), (2,  "GK_B",  1, 2, 50, 2.0),
    (3,  "DEF_A", 2, 1, 50, 5.0), (4,  "DEF_B", 2, 2, 50, 5.0),
    (5,  "DEF_C", 2, 3, 50, 5.0), (6,  "DEF_D", 2, 4, 50, 4.0),
    (7,  "DEF_E", 2, 5, 50, 2.0),
    (8,  "MID_A", 3, 5, 50, 7.0), (9,  "MID_B", 3, 6, 50, 7.0),
    (10, "MID_C", 3, 6, 50, 6.0), (11, "MID_D", 3, 7, 50, 6.0),
    (12, "MID_E", 3, 7, 50, 2.0),
    (13, "FWD_A", 4, 8, 50, 8.0), (14, "FWD_B", 4, 8, 50, 6.0),
    (15, "FWD_C", 4, 3, 50, 5.0),
]


def build_db(extra_players, base_override=None):
    base = base_override if base_override is not None else BASE_PLAYERS
    tmp = Path(tempfile.mkdtemp()) / "fpl_test.db"
    conn = sqlite3.connect(tmp)
    conn.execute("CREATE TABLE teams (team_id INT, short_name TEXT)")
    conn.executemany("INSERT INTO teams VALUES (?,?)",
                     [(i, f"T{i}") for i in range(1, 11)])
    conn.execute("""CREATE TABLE players
        (player_id INT, web_name TEXT, position INT, team_id INT, current_cost INT)""")
    rows = [(p[0], p[1], p[2], p[3], p[4]) for p in base + extra_players]
    conn.executemany("INSERT INTO players VALUES (?,?,?,?,?)", rows)
    conn.commit()
    conn.close()
    return tmp


def preds(extra_players, base_override=None):
    base = base_override if base_override is not None else BASE_PLAYERS
    all_p = base + extra_players
    return pd.DataFrame({
        "player_id": [p[0] for p in all_p],
        "predicted_points": [p[5] for p in all_p],
        "qualifying_games_3": 3,
        "qualifying_games_5": 5,
        "chance_of_playing_next": 100,
    })


def run(extra_players, free_transfers, bank=0, max_transfers=3,
        selling_prices=None, horizon=1, base_override=None):
    db = build_db(extra_players, base_override)
    opt = SquadOptimiser(db_path=db)
    kwargs = dict(
        current_squad=SQUAD,
        free_transfers=free_transfers,
        max_transfers=max_transfers,
        bank=bank,
        horizon=horizon,
    )
    if selling_prices is not None:
        kwargs["selling_prices"] = selling_prices
    return opt.optimise_with_transfers(
        preds(extra_players, base_override), **kwargs
    )


def check(label, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {label}" + (f"  ({detail})" if detail else ""))
    return cond


def main():
    ok = True

    # S1: optimal squad, 1 FT -> no transfers
    r = run([], free_transfers=1)
    ok &= check("S1 optimal squad -> 0 transfers", r["num_transfers"] == 0,
                f"made {r['num_transfers']}")

    # S2: +10 upgrade available for weak MID_E (2.0), 1 FT, affordable
    up = [(100, "MID_STAR", 3, 9, 50, 12.0)]
    r = run(up, free_transfers=1)
    in_ids = set(r["transfers_in"]["player_id"])
    ok &= check("S2 free upgrade taken", r["num_transfers"] == 1 and 100 in in_ids,
                f"in={in_ids}")
    ok &= check("S2 no hit charged", r["hit_points"] == 0)

    # S3: DEF_D 4.0 -> 7.0 (+3), 0 FT (hit=4) -> refused
    up = [(101, "DEF_OK", 2, 9, 50, 7.0)]
    r = run(up, free_transfers=0)
    ok &= check("S3 +3 gain, 0 FT -> hit refused", r["num_transfers"] == 0,
                f"made {r['num_transfers']}")

    # S4: DEF_D 4.0 -> 10.0 (+6), 0 FT -> worth the -4
    up = [(102, "DEF_STAR", 2, 9, 50, 10.0)]
    r = run(up, free_transfers=0)
    ok &= check("S4 +6 gain, 0 FT -> hit taken", r["num_transfers"] == 1
                and r["hit_points"] == 4, f"n={r['num_transfers']} hit={r['hit_points']}")

    # S5: budget: squad value 750 + bank 0. STAR costs 200 (needs 150 extra) ->
    #     unaffordable. OK costs 55 with bank 5 -> affordable.
    up = [(103, "MID_RICH", 3, 9, 200, 15.0), (104, "MID_POOR", 3, 9, 55, 9.0)]
    r = run(up, free_transfers=1, bank=5)
    in_ids = set(r["transfers_in"]["player_id"])
    ok &= check("S5 budget respected: rich star excluded", 103 not in in_ids,
                f"in={in_ids}")
    ok &= check("S5 affordable upgrade taken instead", 104 in in_ids)

    # S6: team 8 already has FWD_A + FWD_B. Add two team-8 stars; only one more
    #     team-8 slot exists -> at most 1 can come in.
    up = [(105, "T8_STAR1", 3, 8, 50, 12.0), (106, "T8_STAR2", 3, 8, 50, 12.0)]
    r = run(up, free_transfers=2)
    squad_t8 = int((r["squad"]["team_id"] == 8).sum())
    ok &= check("S6 club cap held (<=3 from team 8)", squad_t8 <= 3,
                f"team8 count={squad_t8}")

    # S7: DEF_A rose 50->80 market; selling price = 50 + (80-50)//2 = 65.
    #     With selling prices wired in, budget = 65 + 14*50 = 765 (not 780).
    #     Squad is otherwise optimal -> 0 transfers.
    base_risen = [(pid, nm, pos, tm, 80 if pid == 3 else cost, pts_)
                  for (pid, nm, pos, tm, cost, pts_) in BASE_PLAYERS]
    sell = {pid: 50 for pid in SQUAD}
    sell[3] = 65
    r = run([], free_transfers=1, selling_prices=sell, base_override=base_risen)
    ok &= check("S7 risen-price retention -> 0 transfers", r["num_transfers"] == 0,
                f"made {r['num_transfers']}")
    ok &= check("S7 bank non-negative", r["remaining_budget"] >= 0,
                f"remaining={r['remaining_budget']}")

    # S8: horizon=6 amortises hit. DEF_D 4->7 (+3/GW * ~4.1 decay ~= 12.3 > 4).
    #     At horizon=1 this is refused (S3). At horizon=6 it should be taken.
    up = [(101, "DEF_OK", 2, 9, 50, 7.0)]
    r = run(up, free_transfers=0, horizon=6)
    ok &= check("S8 horizon=6 amortises hit -> transfer taken",
                r["num_transfers"] == 1 and r["hit_points"] == 4,
                f"n={r['num_transfers']} hit={r['hit_points']}")

    print("\n" + ("ALL PASS" if ok else "FAILURES PRESENT"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
