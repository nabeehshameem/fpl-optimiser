"""
train_dc.py
Train and save the Dixon-Coles score prediction model.

Prerequisites:
  python wc/scripts/ingest_sb.py   (populates matches table)

Usage:
  python wc/scripts/train_dc.py
  python wc/scripts/train_dc.py --decay 0.4   # down-weight WC2018 more aggressively
  python wc/scripts/train_dc.py --strengths   # also print team strength table
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from wc.src.score_predictor import DCPredictor


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Dixon-Coles WC score predictor")
    parser.add_argument("--decay", type=float, default=0.6,
                        help="Weight for WC2018 matches (WC2022=1.0, default 0.6)")
    parser.add_argument("--strengths", action="store_true",
                        help="Print team attack/defense ratings after training")
    args = parser.parse_args()

    print("Training Dixon-Coles model...")
    predictor = DCPredictor()

    try:
        info = predictor.fit()
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    print(f"  WC 2018/22   : {info['n_wc_matches']}")
    print(f"  WC 2026      : {info['n_wc2026']}  (weight 4.0×)")
    print(f"  Recent form  : {info['n_recent']}")
    print(f"  Total        : {info['n_total']}")
    print(f"  Teams fitted : {info['n_teams']}")
    print(f"  Home adv     : {info['home_adv']:.4f}")
    print(f"  Rho (corr.)  : {info['rho']:.4f}")
    print(f"  Converged    : {info['converged']}")

    predictor.save()

    if args.strengths:
        print("\n  Team strengths (attack / defensive weakness):")
        print(f"  {'Team':25s}  {'Attack':>8}  {'Defense':>8}")
        print("  " + "-" * 46)
        for t in predictor.team_strengths():
            print(f"  {t['name']:25s}  {t['attack']:8.3f}  {t['defense']:8.3f}")


if __name__ == "__main__":
    main()
