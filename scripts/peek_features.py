"""
peek_features.py
Sanity-check the feature builder by inspecting a slice of training data.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.features import build_training_data, get_feature_columns


def main():
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    df = build_training_data(min_gw=8, max_gw=34)
    feature_cols = get_feature_columns(df)

    print(f"Training rows: {len(df)}")
    print(f"Feature columns ({len(feature_cols)}): {feature_cols}")
    print(f"\nGameweek distribution:")
    print(df["gameweek_id"].value_counts().sort_index())

    print(f"\nSample of features for top scorers in GW20:")
    sample = df[df["gameweek_id"] == 20].nlargest(5, "actual_points")
    cols_to_show = [
        "player_id", "actual_points",
        "total_points_mean_5", "expected_goals_mean_5",
        "fdr", "num_fixtures", "is_home", "position",
        "chance_of_playing_next", "news_injury_flag",
    ]
    print(sample[cols_to_show].to_string(index=False))

    print(f"\nFeature null counts (should mostly be 0):")
    null_counts = df[feature_cols].isnull().sum()
    print(null_counts[null_counts > 0] if (null_counts > 0).any() else "All features fully populated.")


if __name__ == "__main__":
    main()