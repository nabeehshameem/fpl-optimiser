"""
ml_predictor.py
LightGBM wrapper for World Cup Fantasy point predictions.

Trained on WC2018 + WC2022 StatsBomb data.
At inference time, features come from qualifying stats + in-tournament form.
"""

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH   = PROJECT_ROOT / "models" / "lgbm_wc.pkl"

from wc.src.features import FEATURE_COLS, build_prediction_features


class WCPredictor:
    """
    Loads a trained LightGBM model and predicts fantasy points for
    all players in an upcoming matchday.
    """

    def __init__(self, model_path: Path = MODEL_PATH):
        self.model_path = model_path
        self.model      = None

    def load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. "
                "Run: python wc/scripts/train_model.py"
            )
        self.model = joblib.load(self.model_path)

    def predict_matchday(
        self,
        matchday: int,
        tournament: str = "WC2026",
        qualifying_tournament: str = "WCQ2026",
    ) -> pd.DataFrame:
        """
        Returns a DataFrame with one row per fantasy player who has a
        fixture in `matchday`, with a `predicted_pts` column added.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call .load() first.")

        df = build_prediction_features(matchday, tournament, qualifying_tournament)

        # Align feature columns — fill missing with 0
        X = df.reindex(columns=FEATURE_COLS, fill_value=0.0)
        df["predicted_pts"] = np.maximum(0.0, self.model.predict(X))
        return df

    def predict_all(self, matchday: int, **kwargs) -> pd.DataFrame:
        """Alias for predict_matchday (consistent with FPL predictor API)."""
        return self.predict_matchday(matchday, **kwargs)
