"""
recommend_transfers.py
Given a current squad (hardcoded for now or loaded from a saved file),
recommend optimal transfer move(s) for the next gameweek.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import sqlite3

import pandas as pd

from src.ml_predictor import LightGBMPredictor
from src.optimiser import SquadOptimiser


# For Stage 3b v1: hardcode a "current squad" by re-running the optimiser fresh
# and pretending its output is what you currently own. Tomorrow we'll add a
# proper "load my actual FPL team from the API" feature.
def get_starting_squad_from_optimiser(predictions_df) -> list:
    optimiser = SquadOptimiser()
    result = optimiser.optimise(predictions_df)
    return list(result["squad"]["player_id"])


def get_target_gameweek() -> int:
    conn = sqlite3.connect(PROJECT_ROOT / "data" / "fpl.db")
    cursor = conn.cursor()
    cursor.execute("SELECT gameweek_id FROM gameweeks WHERE is_next = 1 LIMIT 1")
    row = cursor.fetchone()
    conn.close()
    if row:
        return row[0]
    raise RuntimeError("No upcoming gameweek found.")


def print_squad(label, squad_df, captain_id=None, vice_id=None):
    pos_names = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
    print(label)
    for _, p in squad_df.sort_values(["position", "predicted_points"], ascending=[True, False]).iterrows():
        marker = ""
        if captain_id is not None and p["player_id"] == captain_id:
            marker = " (C)"
        elif vice_id is not None and p["player_id"] == vice_id:
            marker = " (VC)"
        print(f"  {pos_names[p['position']]:3s} {p['web_name']:20s} {p['team']:4s} "
              f"£{p['current_cost']/10:.1f}m  {p['predicted_points']:5.2f}{marker}")


def main():
    target_gw = get_target_gameweek()
    print(f"Recommending transfers for GW{target_gw}\n")

    predictor = LightGBMPredictor()
    predictor.load()
    predictions_df = predictor.predict_all(target_gw=target_gw)

    print("Building hypothetical 'current squad' from a fresh optimal selection...")
    current_squad = get_starting_squad_from_optimiser(predictions_df)
    print(f"Current squad has {len(current_squad)} players.\n")

    optimiser = SquadOptimiser()

    # Try with 1 free transfer (the typical case)
    print("=== Optimal transfer move with 1 free transfer ===\n")
    result = optimiser.optimise_with_transfers(
        predictions_df,
        current_squad=current_squad,
        free_transfers=1,
        max_transfers=3,
    )

    print(f"Recommended transfers: {result['num_transfers']}")
    print(f"Hit cost: {result['hit_points']} points")
    print(f"Gross expected XI points: {result['gross_xi_points']:.2f}")
    print(f"Net expected points (after hit): {result['expected_points']:.2f}")
    print(f"Squad cost: £{result['total_cost']/10:.1f}m\n")

    if result["num_transfers"] == 0:
        print("Recommendation: NO TRANSFERS this gameweek (current squad is optimal).\n")
    else:
        print("--- Transfers OUT ---")
        for _, p in result["transfers_out"].iterrows():
            print(f"  {p['web_name']:20s} {p['team']:4s} (predicted {p['predicted_points']:.2f})")
        print("\n--- Transfers IN ---")
        for _, p in result["transfers_in"].iterrows():
            print(f"  {p['web_name']:20s} {p['team']:4s} (predicted {p['predicted_points']:.2f})")
        print()

    print_squad("--- New Starting XI ---", result["starting_xi"],
                captain_id=result["captain"]["player_id"],
                vice_id=result["vice_captain"]["player_id"])
    print()
    print_squad("--- New Bench ---", result["bench"])


if __name__ == "__main__":
    main()