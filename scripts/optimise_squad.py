"""
optimise_squad.py
Run the optimiser using the latest LightGBM predictions for the next gameweek.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import sqlite3

from src.ml_predictor import LightGBMPredictor
from src.optimiser import SquadOptimiser


def get_target_gameweek() -> int:
    conn = sqlite3.connect(PROJECT_ROOT / "data" / "fpl.db")
    cursor = conn.cursor()
    cursor.execute("SELECT gameweek_id FROM gameweeks WHERE is_next = 1 LIMIT 1")
    row = cursor.fetchone()
    conn.close()
    if row:
        return row[0]
    raise RuntimeError("No upcoming gameweek found.")


def main():
    target_gw = get_target_gameweek()
    print(f"Optimising squad for GW{target_gw}\n")

    predictor = LightGBMPredictor()
    predictor.load()
    predictions_df = predictor.predict_all(target_gw=target_gw)

    optimiser = SquadOptimiser()
    result = optimiser.optimise(predictions_df)

    pos_names = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

    print(f"=== Optimal Squad (Total cost: £{result['total_cost']/10:.1f}m) ===")
    print(f"Expected GW points: {result['expected_points']:.2f}\n")

    print("--- Starting XI ---")
    xi = result["starting_xi"]
    for _, p in xi.iterrows():
        marker = ""
        if p["player_id"] == result["captain"]["player_id"]:
            marker = " (C)"
        elif p["player_id"] == result["vice_captain"]["player_id"]:
            marker = " (VC)"
        print(f"  {pos_names[p['position']]:3s} {p['web_name']:20s} {p['team']:4s} "
              f"£{p['current_cost']/10:.1f}m  {p['predicted_points']:5.2f}{marker}")

    print("\n--- Bench ---")
    for _, p in result["bench"].iterrows():
        print(f"  {pos_names[p['position']]:3s} {p['web_name']:20s} {p['team']:4s} "
              f"£{p['current_cost']/10:.1f}m  {p['predicted_points']:5.2f}")


if __name__ == "__main__":
    main()