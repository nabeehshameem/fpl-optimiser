"""
diagnose_top_predictions.py
Show top predicted players by team and by position to understand the model's biases.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import sqlite3

import pandas as pd

from src.ml_predictor import LightGBMPredictor


def main():
    pd.set_option("display.float_format", lambda x: f"{x:.2f}")

    conn = sqlite3.connect(PROJECT_ROOT / "data" / "fpl.db")
    cursor = conn.cursor()
    cursor.execute("SELECT gameweek_id FROM gameweeks WHERE is_next = 1 LIMIT 1")
    target_gw = cursor.fetchone()[0]
    conn.close()

    predictor = LightGBMPredictor()
    predictor.load()
    preds = predictor.predict_all(target_gw=target_gw)

    conn = sqlite3.connect(PROJECT_ROOT / "data" / "fpl.db")
    players = pd.read_sql_query(
        """
        SELECT p.player_id, p.web_name, p.position, p.current_cost,
               t.short_name AS team
        FROM players p JOIN teams t ON p.team_id = t.team_id
        """,
        conn,
    )
    conn.close()

    # preds already contains position and current_cost from the feature pipeline.
    # Merge only the columns we don't already have.
    df = preds.merge(players[["player_id", "web_name", "team"]], on="player_id")

    print(f"=== Top 20 predicted players overall (GW{target_gw}) ===")
    top20 = df.nlargest(20, "predicted_points")[
        ["web_name", "team", "position", "current_cost", "predicted_points"]
    ]
    print(top20.to_string(index=False))

    print("\n=== Top 5 by each Premier League club ===")
    for team, grp in df.groupby("team"):
        top = grp.nlargest(3, "predicted_points")
        print(f"\n{team}:")
        print(top[["web_name", "position", "current_cost", "predicted_points"]].to_string(index=False))


if __name__ == "__main__":
    main()