"""
run_predictions.py
Run the current prediction model for the next gameweek and write predictions
to the database.
"""

import sqlite3
from pathlib import Path

# Make src/ importable
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.predictor import NaivePredictor


DB_PATH = PROJECT_ROOT / "data" / "fpl.db"


def get_target_gameweek() -> int:
    """Return the gameweek_id of the next upcoming gameweek."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT gameweek_id FROM gameweeks WHERE is_next = 1 LIMIT 1")
    row = cursor.fetchone()
    if row:
        conn.close()
        return row[0]
    # Fallback: first unfinished gameweek
    cursor.execute("SELECT MIN(gameweek_id) FROM gameweeks WHERE finished = 0")
    row = cursor.fetchone()
    conn.close()
    if row and row[0] is not None:
        return row[0]
    raise RuntimeError("No upcoming gameweek found.")


def main():
    target_gw = get_target_gameweek()
    print(f"Predicting for gameweek {target_gw}")

    predictor = NaivePredictor()
    predictions_df = predictor.predict_all(target_gw)

    print(f"\nGenerated {len(predictions_df)} predictions.")
    print(f"\nTop 15 predicted scorers for GW{target_gw}:")
    top = predictions_df.nlargest(15, "predicted_points")[
        ["player_id", "predicted_points", "recent_form", "fdr", "availability", "num_fixtures"]
    ]
    print(top.to_string(index=False))

    rows_written = predictor.write_predictions(target_gw, predictions_df)
    print(f"\nWrote {rows_written} predictions to the database.")


if __name__ == "__main__":
    main()