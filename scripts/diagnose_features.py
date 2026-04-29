"""
diagnose_features.py
Inspect the feature pipeline for a backtest gameweek to find why
qualifying_games_10 is always zero.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.features import build_prediction_features

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

df = build_prediction_features(target_gw=30)

print(f"Total rows: {len(df)}")
print(f"\nColumns: {list(df.columns)}")

print(f"\nSample of 10 rows:")
print(df[["player_id", "team_id", "gameweek_id",
         "total_points_mean_5", "minutes_mean_5",
         "qualifying_games_5", "qualifying_games_10"]].head(10))

print(f"\nDistribution of qualifying_games_10:")
print(df["qualifying_games_10"].value_counts().sort_index())

print(f"\nDistribution of total_points_mean_5:")
print(df["total_points_mean_5"].describe())

print(f"\nNumber of rows where any rolling feature is nonzero:")
form_cols = [c for c in df.columns if "_mean_" in c]
nonzero = (df[form_cols] != 0).any(axis=1).sum()
print(f"{nonzero} of {len(df)}")