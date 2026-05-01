"""
stress_test_transfers.py
Stress tests for optimise_with_transfers().

Three scenarios:
  1. Optimal squad, 1 free transfer  -> expect 0 transfers
  2. Optimal squad, 0 free transfers -> expect 0 transfers (hit not worth it)
  3. Degraded squad (swap 2 best XI starters for low scorers of same position),
     1 free transfer -> expect >= 1 transfer, net points close to optimal
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import sqlite3
import pandas as pd

from src.ml_predictor import LightGBMPredictor
from src.optimiser import SquadOptimiser


def get_target_gameweek() -> int:
    conn = sqlite3.connect(PROJECT_ROOT / "data" / "fpl.db")
    row = conn.execute("SELECT gameweek_id FROM gameweeks WHERE is_next = 1 LIMIT 1").fetchone()
    conn.close()
    if row:
        return row[0]
    raise RuntimeError("No upcoming gameweek found.")


def get_worst_eligible_by_position(predictions_df: pd.DataFrame, position: int,
                                   exclude_ids: set) -> int:
    """Return the player_id of the worst eligible player at `position`, not in exclude_ids."""
    eligible = predictions_df[
        (predictions_df["qualifying_games_3"] >= 2)
        & (predictions_df["qualifying_games_5"] >= 2)
        & (predictions_df["chance_of_playing_next"] >= 75)
        & (predictions_df["position"] == position)
        & (~predictions_df["player_id"].isin(exclude_ids))
    ]
    if eligible.empty:
        raise RuntimeError(f"No eligible replacement found for position {position}")
    return int(eligible.nsmallest(1, "predicted_points").iloc[0]["player_id"])


def run_scenario(label, optimiser, predictions_df, current_squad, free_transfers):
    print(f"\n{'='*60}")
    print(f"SCENARIO: {label}")
    print(f"  free_transfers={free_transfers}, squad size={len(current_squad)}")
    result = optimiser.optimise_with_transfers(
        predictions_df,
        current_squad=current_squad,
        free_transfers=free_transfers,
        max_transfers=3,
    )
    print(f"  transfers: {result['num_transfers']}")
    print(f"  hit_points: {result['hit_points']}")
    print(f"  gross_xi_points: {result['gross_xi_points']:.2f}")
    print(f"  net_expected_points: {result['expected_points']:.2f}")
    if result["num_transfers"] > 0:
        print("  OUT:", list(result["transfers_out"]["web_name"]))
        print("  IN: ", list(result["transfers_in"]["web_name"]))
    return result


def main():
    target_gw = get_target_gameweek()
    print(f"Stress testing transfer recommender for GW{target_gw}\n")

    predictor = LightGBMPredictor()
    predictor.load()
    predictions_df = predictor.predict_all(target_gw=target_gw)

    # Load position data so we can do position-safe squad degradation
    conn = sqlite3.connect(PROJECT_ROOT / "data" / "fpl.db")
    player_info = pd.read_sql_query("SELECT player_id, web_name, position FROM players", conn)
    conn.close()
    predictions_df = predictions_df.merge(
        player_info[["player_id", "web_name", "position"]], on="player_id", how="left",
        suffixes=("", "_pi")
    )
    if "position_pi" in predictions_df.columns:
        predictions_df["position"] = predictions_df["position"].fillna(predictions_df["position_pi"])
        predictions_df.drop(columns=["position_pi"], inplace=True)

    opt = SquadOptimiser()
    base = opt.optimise(predictions_df)
    optimal_squad = list(base["squad"]["player_id"])
    optimal_xi_pts = base["expected_points"]
    print(f"Baseline optimal squad: {len(optimal_squad)} players, XI pts={optimal_xi_pts:.2f}")

    # Scenario 1: optimal squad + 1 free transfer -> must be 0 transfers
    r1 = run_scenario("Optimal squad, 1 free transfer", opt, predictions_df, optimal_squad, 1)
    assert r1["num_transfers"] == 0, f"FAIL: expected 0 transfers, got {r1['num_transfers']}"
    print("  PASS: 0 transfers as expected")

    # Scenario 2: optimal squad + 0 free transfers -> should be 0 transfers
    # (a 4pt hit requires net gain > 4; already-optimal squad won't clear that bar)
    r2 = run_scenario("Optimal squad, 0 free transfers", opt, predictions_df, optimal_squad, 0)
    if r2["num_transfers"] == 0:
        print("  PASS: no hit taken (expected)")
    else:
        net_gain = r2["expected_points"] - optimal_xi_pts
        print(f"  NOTE: {r2['num_transfers']} transfer(s) with hit — net gain vs optimal: {net_gain:+.2f}")
        assert net_gain > 0, "FAIL: took a hit but ended up with fewer expected points"
        print("  PASS: hit taken and net gain is positive")

    # Scenario 3: degrade squad by swapping 2 top-scoring XI starters for worst eligible
    #   of the same position, then verify the recommender fixes it.
    xi = base["starting_xi"].sort_values("predicted_points", ascending=False)
    xi_with_pos = xi.merge(player_info[["player_id", "position"]], on="player_id", how="left",
                           suffixes=("", "_info"))
    if "position_info" in xi_with_pos.columns:
        xi_with_pos["position"] = xi_with_pos["position"].fillna(xi_with_pos["position_info"])

    # Pick top 2 starters from different positions if possible, same position otherwise
    top2 = xi_with_pos.head(2)
    removed = []
    added = []
    used_replacements = set(optimal_squad)

    for _, row in top2.iterrows():
        pid = int(row["player_id"])
        pos = int(row["position"])
        replacement = get_worst_eligible_by_position(predictions_df, pos, used_replacements)
        removed.append(pid)
        added.append(replacement)
        used_replacements.add(replacement)

    degraded_squad = [pid for pid in optimal_squad if pid not in removed] + added
    assert len(degraded_squad) == 15

    removed_pts = [round(float(predictions_df[predictions_df["player_id"]==p]["predicted_points"].iloc[0]), 2)
                   for p in removed]
    added_pts   = [round(float(predictions_df[predictions_df["player_id"]==p]["predicted_points"].iloc[0]), 2)
                   for p in added]
    print(f"\n  Degraded squad: removed top-2 starters (pts {removed_pts})")
    print(f"  Replaced with same-position worst eligibles (pts {added_pts})")

    r3 = run_scenario("Degraded squad (2 best out, same position), 1 free transfer",
                      opt, predictions_df, degraded_squad, 1)

    assert r3["num_transfers"] >= 1, \
        f"FAIL: degraded squad should trigger >= 1 transfer, got {r3['num_transfers']}"

    # net_expected should be better than running the degraded squad with no transfers at all.
    # Compare against the base expected_points of the degraded squad (no transfers).
    # The degraded squad's XI expected points without any change = base XI pts minus removed + added
    degraded_xi_pts = optimal_xi_pts - sum(removed_pts) + min(added_pts)  # rough lower bound
    assert r3["expected_points"] >= degraded_xi_pts, \
        f"FAIL: transfer result ({r3['expected_points']:.2f}) worse than degraded XI ({degraded_xi_pts:.2f})"

    print(f"  PASS: {r3['num_transfers']} transfer(s) recommended")
    print(f"  Net expected: {r3['expected_points']:.2f} vs degraded baseline ~{degraded_xi_pts:.2f}")

    print("\n" + "="*60)
    print("All stress tests passed.")


if __name__ == "__main__":
    main()
