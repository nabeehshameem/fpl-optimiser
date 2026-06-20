"""
run_all.py
Full FPL pipeline in one command.

Steps:
  1. ingest_bootstrap  — refresh players, teams, gameweeks, snapshots (~5s)
  2. ingest_fixtures   — refresh fixture list + FDR (~2s)
  3. backfill_history  — per-player GW history (~5-8 min for 830 players)
  4. train_model       — retrain LightGBM on latest history (~15s)
  5. run_predictions   — write predictions for next GW (~2s)
  6. analyse_gw        — print captain picks, differentials, fixture ticker (~3s)

Usage:
  python scripts/run_all.py
  python scripts/run_all.py --skip-backfill   # fast refresh (skips step 3)
  python scripts/run_all.py --squad 123,456   # pass squad IDs through to analyse_gw
"""

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _header(step: int, title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  Step {step}: {title}")
    print(f"{'='*60}")


def _run_ingest_bootstrap() -> None:
    import importlib.util, runpy
    runpy.run_path(str(PROJECT_ROOT / "scripts" / "ingest_bootstrap.py"), run_name="__main__")


def _run_ingest_fixtures() -> None:
    import runpy
    runpy.run_path(str(PROJECT_ROOT / "scripts" / "ingest_fixtures.py"), run_name="__main__")


def _run_backfill() -> None:
    import runpy
    runpy.run_path(str(PROJECT_ROOT / "scripts" / "backfill_history.py"), run_name="__main__")


def _run_train() -> None:
    from src.ml_predictor import LightGBMPredictor
    predictor = LightGBMPredictor()
    info = predictor.train(min_gw=8, max_gw=None, validation_fraction_by_time=0.2)
    print(f"  Best iteration : {info['best_iteration']}")
    print(f"  Validation MAE : {info['val_mae']:.4f}")
    print(f"  Validation RMSE: {info['val_rmse']:.4f}")
    print(f"  Train GWs: {info['train_gws'][0]}..{info['train_gws'][1]}")
    print(f"  Val GWs  : {info['val_gws'][0]}..{info['val_gws'][1]}")
    print(f"  # features     : {info['n_features']}")
    top = predictor.feature_importance(top_n=10)
    print("\n  Top 10 features:")
    print(top.to_string(index=False))
    predictor.save()


def _run_predictions() -> None:
    import sqlite3
    from src.predictor import NaivePredictor

    db = PROJECT_ROOT / "data" / "fpl.db"
    conn = sqlite3.connect(db)
    row = conn.execute("SELECT gameweek_id FROM gameweeks WHERE is_next=1 LIMIT 1").fetchone()
    if not row:
        row = conn.execute("SELECT MIN(gameweek_id) FROM gameweeks WHERE finished=0").fetchone()
    conn.close()
    if not row or row[0] is None:
        print("  [ERROR] No upcoming gameweek found in DB.")
        return

    gw = row[0]
    print(f"  Predicting for GW{gw}...")
    predictor = NaivePredictor()
    df = predictor.predict_all(gw)
    written = predictor.write_predictions(gw, df)
    print(f"  {written} predictions written for GW{gw}.")

    top = df.nlargest(10, "predicted_points")[["player_id", "predicted_points"]]
    print(f"\n  Top 10 for GW{gw}:")
    print(top.to_string(index=False))


def _run_analyse(squad_ids: list[int] | None) -> None:
    import subprocess
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "analyse_gw.py")]
    if squad_ids:
        cmd += ["--squad", ",".join(str(i) for i in squad_ids)]
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print("  [WARN] analyse_gw.py exited with errors.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Full FPL pipeline")
    parser.add_argument(
        "--skip-backfill", action="store_true",
        help="Skip per-player history backfill (faster, uses cached history)"
    )
    parser.add_argument(
        "--squad", type=str, default="",
        help="Comma-separated player IDs to mark in analysis (optional)"
    )
    args = parser.parse_args()

    squad_ids = []
    if args.squad:
        try:
            squad_ids = [int(x.strip()) for x in args.squad.split(",") if x.strip()]
        except ValueError:
            print("[ERROR] --squad must be comma-separated integers")
            sys.exit(1)

    total_start = time.time()

    _header(1, "Bootstrap (players, teams, gameweeks, snapshots)")
    _run_ingest_bootstrap()

    _header(2, "Fixtures + FDR")
    _run_ingest_fixtures()

    if not args.skip_backfill:
        _header(3, "Backfill per-player GW history  [~5-8 min]")
        print("  Tip: use --skip-backfill to skip this step if history is recent.")
        _run_backfill()
    else:
        print("\n[Step 3 skipped: --skip-backfill]")

    _header(4, "Train LightGBM model")
    _run_train()

    _header(5, "Run predictions")
    _run_predictions()

    _header(6, "GW analysis")
    _run_analyse(squad_ids or None)

    elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  Done in {elapsed/60:.1f} min.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
