"""
diagnose_predictor.py
Run the trained LightGBM model on holdout gameweeks and inspect its outputs.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml_predictor import LightGBMPredictor

p = LightGBMPredictor()
p.load()

for gw in [30, 32, 34]:
    df = p.predict_all(target_gw=gw)
    print(f"\n=== GW{gw} ===")
    print(df["predicted_points"].describe())
    print(f"Nonzero predictions: {(df['predicted_points'] != 0).sum()} of {len(df)}")
    print("Top 5 predictions:")
    top5 = df.nlargest(5, "predicted_points")[
        ["player_id", "predicted_points", "qualifying_games_10", "num_fixtures"]
    ]
    print(top5.to_string(index=False))