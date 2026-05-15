"""
train_model.py
Train a LightGBM model on WC2018 + WC2022 StatsBomb data.

Prerequisites:
  python wc/scripts/ingest_sb.py      (populates training data)

Usage:
  python wc/scripts/train_model.py
  python wc/scripts/train_model.py --tournaments WC2022
  python wc/scripts/train_model.py --cv              # print cross-val MAE
"""

import argparse
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import mean_absolute_error

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from wc.src.features import build_training_features, FEATURE_COLS

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "lgbm_wc.pkl"

LGBM_PARAMS = {
    "objective":     "regression",
    "metric":        "mae",
    "n_estimators":  400,
    "learning_rate": 0.05,
    "num_leaves":    31,
    "max_depth":     5,
    "min_child_samples": 10,
    "feature_fraction":  0.8,
    "bagging_fraction":  0.8,
    "bagging_freq":      5,
    "lambda_l1":     0.1,
    "lambda_l2":     0.1,
    "verbose":       -1,
}


def train(tournaments: list | None = None, run_cv: bool = False) -> None:
    print("Loading training features...")
    df = build_training_features(tournaments=tournaments)
    print(f"  {len(df)} training rows, {df['player_id'].nunique()} players.")

    if df.empty:
        print("[ERROR] No training data. Run ingest_sb.py first.")
        sys.exit(1)

    X = df.reindex(columns=FEATURE_COLS, fill_value=0.0)
    y = df["fantasy_pts"].fillna(0.0)

    # Naive baseline: predict mean pts by position
    pos_means = df.groupby("pos_enc")["fantasy_pts"].mean()
    naive_pred = df["pos_enc"].map(pos_means).fillna(y.mean())
    naive_mae  = mean_absolute_error(y, naive_pred)
    print(f"  Naive baseline MAE (pos mean): {naive_mae:.3f}")

    model = lgb.LGBMRegressor(**LGBM_PARAMS)

    if run_cv:
        # Group k-fold by player so we don't leak a player's future form into training
        groups = df["player_id"].values
        cv     = GroupKFold(n_splits=5)
        scores = cross_val_score(model, X, y, cv=cv, groups=groups,
                                 scoring="neg_mean_absolute_error")
        cv_mae = -scores.mean()
        print(f"  CV MAE (5-fold, grouped by player): {cv_mae:.3f}")
        print(f"  Improvement vs naive: {(naive_mae - cv_mae) / naive_mae * 100:.1f}%")

    print("Training final model on full dataset...")
    model.fit(X, y)

    train_preds = model.predict(X)
    train_mae   = mean_absolute_error(y, train_preds)
    print(f"  Train MAE: {train_mae:.3f}")

    # Feature importances (top 10)
    fi = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("\n  Top feature importances:")
    for feat, imp in fi.head(10).items():
        print(f"    {feat:30s}  {imp}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved: {MODEL_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Train WC Fantasy LightGBM model")
    parser.add_argument("--tournaments", type=str, default="",
                        help="Comma-separated tournament keys to train on (default: all)")
    parser.add_argument("--cv", action="store_true",
                        help="Print grouped cross-validation MAE")
    args = parser.parse_args()

    t_list = [t.strip() for t in args.tournaments.split(",") if t.strip()] or None
    train(tournaments=t_list, run_cv=args.cv)


if __name__ == "__main__":
    main()
