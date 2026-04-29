"""
ml_predictor.py
LightGBM-based predictor for FPL points.

Trains on (features → actual_points) historical data, with proper time-series
cross-validation. At prediction time, exposes the same predict_all() interface
as NaivePredictor so the optimiser and evaluator can swap implementations.
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.features import (
    build_training_data,
    build_prediction_features,
    get_feature_columns,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "fpl.db"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ---------- Hyperparameters ----------
# These are sensible LightGBM defaults for ~20k tabular rows. Not tuned.
LGB_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_data_in_leaf": 30,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "random_state": 42,
}

NUM_BOOST_ROUND = 1000
EARLY_STOPPING_ROUNDS = 50


class LightGBMPredictor:
    """
    Production predictor using a trained LightGBM model.

    Workflow:
      1. Train once with train(min_gw, max_gw)
      2. Save with save() / load() for persistence
      3. Predict next gameweek with predict_all(target_gw)
    """

    MODEL_NAME = "lightgbm_v1"
    MODEL_PATH = MODEL_DIR / "lightgbm_v1.txt"

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.model: Optional[lgb.Booster] = None
        self.feature_columns: Optional[list] = None

    def train(self, min_gw: int = 8, max_gw: Optional[int] = None,
              validation_fraction_by_time: float = 0.2) -> dict:
        """
        Train the model on historical data with a time-based holdout.

        The last `validation_fraction_by_time` of gameweeks are used as validation.
        This preserves time order — we never train on future data and evaluate on past.
        """
        df = build_training_data(min_gw=min_gw, max_gw=max_gw)
        self.feature_columns = get_feature_columns(df)

        # Time-based split: hold out the most recent gameweeks
        all_gws = sorted(df["gameweek_id"].unique())
        split_idx = int(len(all_gws) * (1 - validation_fraction_by_time))
        train_gws = all_gws[:split_idx]
        val_gws = all_gws[split_idx:]

        train_df = df[df["gameweek_id"].isin(train_gws)]
        val_df = df[df["gameweek_id"].isin(val_gws)]

        print(f"Training: {len(train_df)} rows from GWs {train_gws[0]}..{train_gws[-1]}")
        print(f"Validation: {len(val_df)} rows from GWs {val_gws[0]}..{val_gws[-1]}")

        X_train = train_df[self.feature_columns]
        y_train = train_df["actual_points"]
        X_val = val_df[self.feature_columns]
        y_val = val_df["actual_points"]

        train_set = lgb.Dataset(X_train, label=y_train)
        val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

        self.model = lgb.train(
            params=LGB_PARAMS,
            train_set=train_set,
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[train_set, val_set],
            valid_names=["train", "val"],
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING_ROUNDS),
                lgb.log_evaluation(period=50),
            ],
        )

        # Final validation MAE
        val_pred = self.model.predict(X_val)
        val_mae = float(np.mean(np.abs(val_pred - y_val)))
        val_rmse = float(np.sqrt(np.mean((val_pred - y_val) ** 2)))

        return {
            "best_iteration": self.model.best_iteration,
            "val_mae": val_mae,
            "val_rmse": val_rmse,
            "train_gws": (train_gws[0], train_gws[-1]),
            "val_gws": (val_gws[0], val_gws[-1]),
            "n_features": len(self.feature_columns),
        }

    def feature_importance(self, top_n: int = 15) -> pd.DataFrame:
        """Return the most important features by gain."""
        if self.model is None:
            raise RuntimeError("Model not trained yet.")
        importances = self.model.feature_importance(importance_type="gain")
        df = pd.DataFrame({
            "feature": self.feature_columns,
            "gain": importances,
        }).sort_values("gain", ascending=False).head(top_n)
        return df.reset_index(drop=True)

    def save(self, path: Optional[Path] = None) -> None:
        if self.model is None:
            raise RuntimeError("Model not trained yet.")
        path = path or self.MODEL_PATH
        self.model.save_model(str(path))
        feature_path = path.with_suffix(".features.txt")
        feature_path.write_text("\n".join(self.feature_columns))
        print(f"Saved model to {path}")
        print(f"Saved feature list to {feature_path}")

    def load(self, path: Optional[Path] = None) -> None:
        path = path or self.MODEL_PATH
        self.model = lgb.Booster(model_file=str(path))
        feature_path = path.with_suffix(".features.txt")
        self.feature_columns = feature_path.read_text().strip().split("\n")
        print(f"Loaded model from {path}")

    def predict_all(self, target_gw: int,
                    as_of_gameweek: Optional[int] = None) -> pd.DataFrame:
        """
        Predict expected points for every player for target_gw.

        Has the same signature as NaivePredictor.predict_all, so the evaluator
        and optimiser don't care which implementation is used.

        Note: the LightGBM features are built using historical data strictly
        before target_gw, regardless of as_of_gameweek (which is accepted only
        for interface compatibility).
        """
        if self.model is None:
            raise RuntimeError("Model not trained or loaded yet.")

        df = build_prediction_features(target_gw=target_gw)
        X = df[self.feature_columns]
        df["predicted_points"] = self.model.predict(X)

        # Zero out players whose teams have no fixtures this gameweek
        df.loc[df["num_fixtures"] == 0, "predicted_points"] = 0.0

        # Also zero out players with no form history at all (no qualifying games anywhere)
        no_history = df["qualifying_games_10"] == 0
        df.loc[no_history, "predicted_points"] = 0.0

        return df

    def write_predictions(self, target_gw: int,
                          predictions_df: pd.DataFrame) -> int:
        """Write predictions to the predictions table."""
        now_iso = datetime.now(timezone.utc).isoformat()
        rows = [
            (int(r.player_id), int(target_gw), self.MODEL_NAME,
             float(r.predicted_points), now_iso)
            for r in predictions_df.itertuples()
        ]
        conn = sqlite3.connect(self.db_path)
        try:
            conn.executemany(
                """
                INSERT INTO predictions (player_id, gameweek_id, model_name,
                                          predicted_points, prediction_time)
                VALUES (?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()
        finally:
            conn.close()
        return len(rows)