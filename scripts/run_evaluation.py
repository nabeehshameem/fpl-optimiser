"""
run_evaluation.py
Backtest the current predictor across historical gameweeks and print results.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluator import Evaluator
from src.predictor import NaivePredictor


# Skip very early gameweeks where there's not enough form history
EVAL_GAMEWEEKS = list(range(8, 35))  # GW8 through GW34


def main():
    predictor = NaivePredictor()
    evaluator = Evaluator()

    print(f"Evaluating {predictor.MODEL_NAME} across {len(EVAL_GAMEWEEKS)} gameweeks "
          f"(GW{EVAL_GAMEWEEKS[0]}..GW{EVAL_GAMEWEEKS[-1]})")

    raw = evaluator.evaluate_many(predictor, EVAL_GAMEWEEKS, restrict_to_appeared=True)
    summary = evaluator.summarise(raw)

    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    import pandas as pd  # local import for the float formatter
    main()