"""
run_evaluation.py
Backtest naive_v1 and lightgbm_v1 across the LightGBM holdout window.

FPL's ep_next baseline is only included for gameweeks where snapshots exist
(typically only the current gameweek). Pre-snapshot gameweeks compare
the two models against actuals only.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.evaluator import Evaluator
from src.predictor import NaivePredictor
from src.ml_predictor import LightGBMPredictor


EVAL_GAMEWEEKS = list(range(29, 35))


def _safe_mae(pred: pd.Series, actual: pd.Series) -> float:
    """Compute MAE, returning NaN if pred is entirely NaN (no snapshot for that GW)."""
    pred_numeric = pd.to_numeric(pred, errors="coerce")
    if pred_numeric.notna().sum() == 0:
        return float("nan")
    mask = pred_numeric.notna()
    return float(np.mean(np.abs(pred_numeric[mask] - actual[mask])))


def main():
    pd.set_option("display.float_format", lambda x: f"{x:.3f}" if pd.notna(x) else "n/a")

    evaluator = Evaluator()
    naive = NaivePredictor()
    ml = LightGBMPredictor()
    ml.load()

    print(f"\nEvaluation gameweeks: GW{EVAL_GAMEWEEKS[0]}..GW{EVAL_GAMEWEEKS[-1]}")
    print("Predictors: naive_v1, lightgbm_v1\n")
    print("Note: FPL ep_next baseline shown only for gameweeks with available snapshots.\n")

    naive_raw = evaluator.evaluate_many(naive, EVAL_GAMEWEEKS, restrict_to_appeared=True)
    ml_raw = evaluator.evaluate_many(ml, EVAL_GAMEWEEKS, restrict_to_appeared=True)

    rows = []
    for gw in EVAL_GAMEWEEKS:
        naive_gw = naive_raw[naive_raw["gameweek"] == gw]
        ml_gw = ml_raw[ml_raw["gameweek"] == gw]

        rows.append({
            "gameweek": gw,
            "n_players": len(naive_gw),
            "naive_mae": _safe_mae(naive_gw["model_pred"], naive_gw["actual_points"]),
            "ml_mae":    _safe_mae(ml_gw["model_pred"], ml_gw["actual_points"]),
            "fpl_mae":   _safe_mae(ml_gw["fpl_ep_next"], ml_gw["actual_points"]),
            "zero_mae":  float(ml_gw["actual_points"].abs().mean()),
        })

    overall_row = {
        "gameweek": "OVERALL",
        "n_players": len(naive_raw),
        "naive_mae": _safe_mae(naive_raw["model_pred"], naive_raw["actual_points"]),
        "ml_mae":    _safe_mae(ml_raw["model_pred"], ml_raw["actual_points"]),
        "fpl_mae":   _safe_mae(ml_raw["fpl_ep_next"], ml_raw["actual_points"]),
        "zero_mae":  float(ml_raw["actual_points"].abs().mean()),
    }

    summary = pd.DataFrame(rows + [overall_row])
    print(summary.to_string(index=False))

    # Top-N MAE: focus on each model's high-confidence picks.
    print("\n=== Top-N MAE (accuracy on each model's top picks across the holdout) ===")
    top_n_rows = []
    for n in [10, 30, 50]:
        naive_top = naive_raw.nlargest(n * len(EVAL_GAMEWEEKS), "model_pred")
        ml_top = ml_raw.nlargest(n * len(EVAL_GAMEWEEKS), "model_pred")
        top_n_rows.append({
            "top_n": n,
            "naive_mae_on_picks": _safe_mae(naive_top["model_pred"], naive_top["actual_points"]),
            "ml_mae_on_picks":    _safe_mae(ml_top["model_pred"], ml_top["actual_points"]),
        })
    print(pd.DataFrame(top_n_rows).to_string(index=False))


if __name__ == "__main__":
    main()