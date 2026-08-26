"""
eval_bench_weight.py

Compare BENCH_WEIGHT=0.1 vs BENCH_WEIGHT=0.2 on archived gameweeks.

Data source: data/fpl_2526.db — the 2025/26 full-season archive.
Prediction source: naive_v1 predictions stored for GWs 35, 37, 38.
(No DC predictions were stored for 2025/26 — these are the only GWs
with any stored predictions in the archive.)

Method
------
For each available GW:
  1. Load stored predictions + eligibility from archive snapshots.
  2. Build a fresh 15-player squad using BENCH_WEIGHT=0.1 and 0.2.
  3. Score each squad's starting XI + captain double against actual
     player_gameweek_history totals (no auto-sub simulation — isolates
     the optimiser's XI selection choice from stochastic events).
  4. Report per-GW and overall difference.

Verdict is used to decide whether to keep BENCH_WEIGHT=0.2 or revert
to the pre-5f11918 value of 0.1.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pulp

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

ARCHIVE_DB = PROJECT_ROOT / "data" / "fpl_2526.db"

SQUAD_SIZE = 15
STARTING_XI = 11
MAX_PER_CLUB = 3
MIN_QUALIFYING_GAMES_3 = 2
MIN_QUALIFYING_GAMES_5 = 2
MIN_CHANCE_OF_PLAYING = 75
BUDGET = 1000  # £100m in tenths

POSITION_RULES = [
    (1, 2, 1, 1),
    (2, 5, 3, 5),
    (3, 5, 2, 5),
    (4, 3, 1, 3),
]


def _build_squad(df: pd.DataFrame, bench_weight: float) -> dict | None:
    """Solve the squad ILP with the given bench_weight. Returns None on failure."""
    prob = pulp.LpProblem("BW_eval", pulp.LpMaximize)
    pids = df["player_id"].tolist()

    select = {pid: pulp.LpVariable(f"s_{pid}", cat="Binary") for pid in pids}
    start  = {pid: pulp.LpVariable(f"x_{pid}",  cat="Binary") for pid in pids}
    cap    = {pid: pulp.LpVariable(f"c_{pid}",  cat="Binary") for pid in pids}

    prob += pulp.lpSum(
        row.predicted_points * start[row.player_id]
        + row.predicted_points * cap[row.player_id]
        + bench_weight * row.predicted_points * (select[row.player_id] - start[row.player_id])
        for row in df.itertuples()
    )

    prob += pulp.lpSum(select.values()) == SQUAD_SIZE
    prob += pulp.lpSum(start.values())  == STARTING_XI
    prob += pulp.lpSum(cap.values())    == 1
    prob += pulp.lpSum(
        row.now_cost * select[row.player_id] for row in df.itertuples()
    ) <= BUDGET

    for pid in pids:
        prob += start[pid]  <= select[pid]
        prob += cap[pid]    <= start[pid]

    for pos_id, sq, xi_min, xi_max in POSITION_RULES:
        pos_pids = df[df["position"] == pos_id]["player_id"].tolist()
        prob += pulp.lpSum(select[p] for p in pos_pids) == sq
        prob += pulp.lpSum(start[p]  for p in pos_pids) >= xi_min
        prob += pulp.lpSum(start[p]  for p in pos_pids) <= xi_max

    for team_id in df["team_id"].unique():
        team_pids = df[df["team_id"] == team_id]["player_id"].tolist()
        prob += pulp.lpSum(select[p] for p in team_pids) <= MAX_PER_CLUB

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        return None

    df = df.copy()
    df["sel"] = [int(round(select[p].value())) for p in pids]
    df["sta"] = [int(round(start[p].value()))  for p in pids]
    df["cap"] = [int(round(cap[p].value()))    for p in pids]
    return df


def _score_squad(df: pd.DataFrame, actuals: dict[int, int]) -> tuple[int, list[str]]:
    """Score XI + captain double. Returns (actual_net_pts, captain_name)."""
    xi  = df[(df["sta"] == 1)].copy()
    cap = df[(df["cap"] == 1)].iloc[0]

    pts = sum(actuals.get(pid, 0) for pid in xi["player_id"])
    pts += actuals.get(cap["player_id"], 0)  # captain double

    bench = df[(df["sel"] == 1) & (df["sta"] == 0)].sort_values(
        "predicted_points", ascending=False
    )
    bench_names = bench["web_name"].tolist()
    return pts, bench_names


def evaluate_gw(conn: sqlite3.Connection, gw: int) -> dict | None:
    # ── predictions ─────────────────────────────────────────────────────
    preds = pd.read_sql_query(
        "SELECT player_id, predicted_points FROM predictions "
        "WHERE gameweek_id=? AND model_name='naive_v1'",
        conn, params=(gw,)
    )
    if preds.empty:
        return None

    # ── snapshot for cost + chance_of_playing ───────────────────────────
    snap = pd.read_sql_query(
        """SELECT s.player_id, s.now_cost, s.chance_of_playing_next
           FROM player_snapshots s
           WHERE s.gameweek_id=?
             AND s.snapshot_id = (
                 SELECT MAX(snapshot_id) FROM player_snapshots
                 WHERE player_id=s.player_id AND gameweek_id=?)""",
        conn, params=(gw, gw)
    )

    # ── qualifying games (last 3 and 5 GWs before current) ──────────────
    hist = pd.read_sql_query(
        "SELECT player_id, gameweek_id, minutes FROM player_gameweek_history "
        "WHERE gameweek_id BETWEEN ? AND ?",
        conn, params=(max(1, gw - 5), gw - 1)
    )
    def qualifying(n):
        recent = hist[hist["gameweek_id"] >= gw - n]
        return recent.groupby("player_id")["minutes"].apply(
            lambda m: int((m > 0).sum())
        ).rename(f"q{n}")
    q3 = qualifying(3)
    q5 = qualifying(5)

    # ── players base ────────────────────────────────────────────────────
    players = pd.read_sql_query(
        "SELECT p.player_id, p.web_name, p.team_id, p.position "
        "FROM players p", conn
    )

    df = (players
          .merge(preds, on="player_id")
          .merge(snap,  on="player_id")
          .merge(q3.reset_index(), on="player_id", how="left")
          .merge(q5.reset_index(), on="player_id", how="left"))
    df["q3"] = df["q3"].fillna(0).astype(int)
    df["q5"] = df["q5"].fillna(0).astype(int)
    df["chance_of_playing_next"] = df["chance_of_playing_next"].fillna(100)

    eligible = (
        (df["q3"] >= MIN_QUALIFYING_GAMES_3)
        & (df["q5"] >= MIN_QUALIFYING_GAMES_5)
        & (df["chance_of_playing_next"] >= MIN_CHANCE_OF_PLAYING)
    )
    df = df[eligible].reset_index(drop=True)

    # ── actuals ─────────────────────────────────────────────────────────
    act_rows = pd.read_sql_query(
        "SELECT player_id, total_points FROM player_gameweek_history WHERE gameweek_id=?",
        conn, params=(gw,)
    )
    actuals = dict(zip(act_rows["player_id"], act_rows["total_points"]))

    # ── solve both weights ───────────────────────────────────────────────
    results = {}
    for w in (0.1, 0.2):
        solved = _build_squad(df, w)
        if solved is None:
            print(f"  GW{gw}: solver failed for weight={w}")
            return None
        pts, bench = _score_squad(solved, actuals)
        cap_name = solved[solved["cap"] == 1]["web_name"].iloc[0]
        results[w] = {"pts": pts, "cap": cap_name, "bench": bench}

    return {"gw": gw, "w01": results[0.1], "w02": results[0.2], "n_eligible": len(df)}


def main():
    if not ARCHIVE_DB.exists():
        sys.exit(f"Archive DB not found: {ARCHIVE_DB}")

    conn = sqlite3.connect(ARCHIVE_DB)

    gw_rows = pd.read_sql_query(
        "SELECT DISTINCT gameweek_id FROM predictions ORDER BY gameweek_id", conn
    )
    gws = gw_rows["gameweek_id"].tolist()
    print(f"GWs with stored predictions in archive: {gws}\n")
    print(f"{'GW':>4}  {'BW=0.1':>8}  {'BW=0.2':>8}  {'Diff':>6}  {'Winner':>8}")
    print("-" * 50)

    total_01 = total_02 = 0
    gw_results = []
    for gw in gws:
        r = evaluate_gw(conn, gw)
        if r is None:
            continue
        p01 = r["w01"]["pts"]
        p02 = r["w02"]["pts"]
        diff = p02 - p01
        winner = "0.2" if diff > 0 else ("0.1" if diff < 0 else "draw")
        print(f"  {gw:>2}  {p01:>8}  {p02:>8}  {diff:>+6}  {winner:>8}")
        if r["w01"]["bench"] != r["w02"]["bench"]:
            print(f"      bench differs → 0.1: {r['w01']['bench']}")
            print(f"                     0.2: {r['w02']['bench']}")
        total_01 += p01
        total_02 += p02
        gw_results.append({"gw": gw, "w01": p01, "w02": p02})

    if not gw_results:
        print("No gameweeks evaluated.")
        return

    n = len(gw_results)
    diff = total_02 - total_01
    print("-" * 50)
    print(f"{'TOTAL':>4}  {total_01:>8}  {total_02:>8}  {diff:>+6}  "
          f"{'0.2 wins' if diff > 0 else ('0.1 wins' if diff < 0 else 'draw')}")
    print(f"\nn={n} GWs (naive_v1 predictions). Sample is too small for "
          f"statistical significance — see docs/evaluations/bench_weight.md")

    conn.close()


if __name__ == "__main__":
    main()
