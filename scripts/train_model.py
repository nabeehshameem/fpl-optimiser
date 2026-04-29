"""
train_model.py
Train the LightGBMPredictor on historical data, evaluate on a time-based holdout,
print feature importance, and save the model.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml_predictor import LightGBMPredictor


def main():
    predictor = LightGBMPredictor()
    info = predictor.train(min_gw=8, max_gw=None, validation_fraction_by_time=0.2)

    print("\n=== Training complete ===")
    print(f"Best iteration: {info['best_iteration']}")
    print(f"Validation MAE:  {info['val_mae']:.4f}")
    print(f"Validation RMSE: {info['val_rmse']:.4f}")
    print(f"Train GWs: {info['train_gws'][0]}..{info['train_gws'][1]}")
    print(f"Val GWs:   {info['val_gws'][0]}..{info['val_gws'][1]}")
    print(f"# features: {info['n_features']}")

    print("\n=== Top 15 features by gain ===")
    print(predictor.feature_importance(top_n=15).to_string(index=False))

    predictor.save()


if __name__ == "__main__":
    main()