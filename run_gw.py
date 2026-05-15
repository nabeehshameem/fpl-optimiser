"""
run_gw.py
One-command gameweek runner. Fetches fresh data, optionally retrains the model,
and prints the optimal squad (and transfer recommendations if --squad is given).

Usage:
  python run_gw.py                         # Full pipeline (fetch + retrain + optimise)
  python run_gw.py --quick                 # Skip backfill + retrain (use existing model)
  python run_gw.py --squad 123,456,...     # Also recommend transfers for your current squad
  python run_gw.py --free-transfers 2      # Set available free transfers (default 1)
  python run_gw.py --bank 1.4             # Cash in the bank in £m (crucial for correct budget)

Flags:
  --quick           Skip the slow history backfill (~5-10 min) and model retrain.
                    Use when prices/fixtures changed but you don't need updated training data.
  --skip-backfill   Skip only backfill (still retrains on existing data).
  --skip-train      Skip only model retraining (still backfills history).
  --squad IDS       Comma-separated player_ids of your current 15-player squad.
                    Triggers transfer recommendation on top of squad optimisation.
  --free-transfers N  Number of free transfers available (default 1).
  --max-transfers N   Maximum transfers to consider (default 3).
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS = PROJECT_ROOT / "scripts"


def run_step(label: str, script: str) -> bool:
    """Run a script step, print timing, return True on success."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run([sys.executable, str(SCRIPTS / script)])
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n[FAILED] {label} (exit code {result.returncode}) after {elapsed:.1f}s")
        return False
    print(f"\n[OK] {label} completed in {elapsed:.1f}s")
    return True


def run_optimise(squad_ids: list[int], free_transfers: int, max_transfers: int, bank: int = 0):
    """Run squad optimisation inline (avoids a second subprocess startup + model load)."""
    import sqlite3
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.ml_predictor import LightGBMPredictor
    from src.optimiser import SquadOptimiser

    conn = sqlite3.connect(PROJECT_ROOT / "data" / "fpl.db")
    row = conn.execute("SELECT gameweek_id FROM gameweeks WHERE is_next = 1 LIMIT 1").fetchone()
    conn.close()
    if not row:
        print("[ERROR] No upcoming gameweek found in DB. Run ingest_bootstrap first.")
        return
    target_gw = row[0]

    print(f"\nOptimising squad for GW{target_gw}\n")
    predictor = LightGBMPredictor()
    predictor.load()
    predictions_df = predictor.predict_all(target_gw=target_gw)

    pos_names = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
    opt = SquadOptimiser()

    # ── Fresh optimal squad ──────────────────────────────────────────────────
    result = opt.optimise(predictions_df)

    print(f"{'='*60}")
    print(f"  OPTIMAL SQUAD  (GW{target_gw})")
    print(f"{'='*60}")
    print(f"Total cost:       £{result['total_cost']/10:.1f}m  (budget £100.0m, "
          f"remaining £{(1000 - result['total_cost'])/10:.1f}m)")
    print(f"Expected XI pts:  {result['expected_points']:.2f}  "
          f"(incl. captain bonus)\n")

    print("--- Starting XI ---")
    for _, p in result["starting_xi"].iterrows():
        marker = ""
        if p["player_id"] == result["captain"]["player_id"]:
            marker = " (C)"
        elif p["player_id"] == result["vice_captain"]["player_id"]:
            marker = " (VC)"
        print(f"  {pos_names[p['position']]:3s}  {p['web_name']:20s}  "
              f"{p['team']:4s}  £{p['current_cost']/10:.1f}m  "
              f"{p['predicted_points']:5.2f}{marker}")

    print("\n--- Bench ---")
    for _, p in result["bench"].iterrows():
        print(f"  {pos_names[p['position']]:3s}  {p['web_name']:20s}  "
              f"{p['team']:4s}  £{p['current_cost']/10:.1f}m  "
              f"{p['predicted_points']:5.2f}")

    # ── Transfer recommendations (if squad provided) ──────────────────────
    if squad_ids:
        if len(squad_ids) != 15:
            print(f"\n[WARN] --squad must have exactly 15 player IDs, got {len(squad_ids)}. "
                  "Skipping transfer recommendations.")
            return

        print(f"\n{'='*60}")
        print(f"  TRANSFER RECOMMENDATIONS  (GW{target_gw})")
        print(f"  free_transfers={free_transfers}, max_transfers={max_transfers}, bank=£{bank/10:.1f}m")
        print(f"{'='*60}")

        tr = opt.optimise_with_transfers(
            predictions_df,
            current_squad=squad_ids,
            free_transfers=free_transfers,
            max_transfers=max_transfers,
            bank=bank,
        )

        print(f"Recommended transfers: {tr['num_transfers']}")
        print(f"Hit cost:              {tr['hit_points']} pts")
        print(f"Gross XI expected:     {tr['gross_xi_points']:.2f}")
        print(f"Net expected (after hit): {tr['expected_points']:.2f}")
        print(f"Squad cost:            £{tr['total_cost']/10:.1f}m  "
              f"(remaining £{tr['remaining_budget']/10:.1f}m)\n")

        if tr["num_transfers"] == 0:
            print("Recommendation: NO TRANSFERS this gameweek.")
        else:
            print("--- OUT ---")
            for _, p in tr["transfers_out"].iterrows():
                print(f"  {p['web_name']:20s}  (predicted {p['predicted_points']:.2f})")
            print("--- IN  ---")
            for _, p in tr["transfers_in"].iterrows():
                print(f"  {p['web_name']:20s}  (predicted {p['predicted_points']:.2f})")

        print("\n--- New Starting XI ---")
        for _, p in tr["starting_xi"].iterrows():
            marker = ""
            if p["player_id"] == tr["captain"]["player_id"]:
                marker = " (C)"
            elif p["player_id"] == tr["vice_captain"]["player_id"]:
                marker = " (VC)"
            print(f"  {pos_names[p['position']]:3s}  {p['web_name']:20s}  "
                  f"{p['team']:4s}  £{p['current_cost']/10:.1f}m  "
                  f"{p['predicted_points']:5.2f}{marker}")

        print("\n--- New Bench ---")
        for _, p in tr["bench"].iterrows():
            print(f"  {pos_names[p['position']]:3s}  {p['web_name']:20s}  "
                  f"{p['team']:4s}  £{p['current_cost']/10:.1f}m  "
                  f"{p['predicted_points']:5.2f}")


def main():
    parser = argparse.ArgumentParser(description="FPL gameweek runner")
    parser.add_argument("--quick", action="store_true",
                        help="Skip backfill and retrain; use existing model")
    parser.add_argument("--skip-backfill", action="store_true",
                        help="Skip history backfill only")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip model retraining only")
    parser.add_argument("--squad", type=str, default="",
                        help="Comma-separated player_ids of your current 15-player squad")
    parser.add_argument("--free-transfers", type=int, default=1,
                        help="Free transfers available (default 1)")
    parser.add_argument("--max-transfers", type=int, default=3,
                        help="Max transfers to consider (default 3)")
    parser.add_argument("--bank", type=float, default=0.0,
                        help="Cash in the bank in £m, e.g. --bank 1.4 (default 0.0). "
                             "Required for accurate transfer budget. Find it on your FPL squad page.")
    args = parser.parse_args()

    squad_ids = []
    if args.squad:
        try:
            squad_ids = [int(x.strip()) for x in args.squad.split(",") if x.strip()]
        except ValueError:
            print("[ERROR] --squad must be comma-separated integers, e.g. --squad 123,456,789")
            sys.exit(1)

    bank_tenths = round(args.bank * 10)  # convert £m to tenths for the optimiser

    skip_backfill = args.quick or args.skip_backfill
    skip_train    = args.quick or args.skip_train

    wall_start = time.time()

    # Step 1: always refresh prices and snapshots first — fixes stale-price budget errors
    if not run_step("Step 1/4: Fetching latest player prices & snapshots", "ingest_bootstrap.py"):
        sys.exit(1)

    # Step 2: always refresh fixture list (DGW detection depends on this)
    if not run_step("Step 2/4: Fetching fixtures", "ingest_fixtures.py"):
        sys.exit(1)

    # Step 3: backfill match history (slow but needed for accurate form features)
    if skip_backfill:
        print("\n[SKIP] Step 3/4: History backfill (--quick / --skip-backfill)")
    else:
        if not run_step("Step 3/4: Backfilling match history (this takes ~5-10 min)", "backfill_history.py"):
            sys.exit(1)

    # Step 4: retrain model on refreshed data
    if skip_train:
        print("\n[SKIP] Step 4/4: Model retrain (--quick / --skip-train)")
    else:
        if not run_step("Step 4/4: Retraining LightGBM model", "train_model.py"):
            sys.exit(1)

    # Optimise (+ optional transfers)
    run_optimise(squad_ids, args.free_transfers, args.max_transfers, bank_tenths)

    print(f"\n{'='*60}")
    print(f"  All done in {time.time() - wall_start:.0f}s")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
