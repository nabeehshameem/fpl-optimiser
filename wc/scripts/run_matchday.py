"""
run_matchday.py
One-command World Cup Fantasy matchday runner.

Loads predictions and prints score predictions, captain picks, differentials.
Optionally shows squad optimiser if --optimise flag is set.

Usage:
  python wc/scripts/run_matchday.py                       # auto-detect next matchday
  python wc/scripts/run_matchday.py --matchday 2
  python wc/scripts/run_matchday.py --matchday 1 --top 10
  python wc/scripts/run_matchday.py --matchday 1 --squad 123,456,...  # mark squad players *
  python wc/scripts/run_matchday.py --matchday 1 --optimise           # squad optimiser
  python wc/scripts/run_matchday.py --matchday 1 --max-ownership 20   # wider differentials
  python wc/scripts/run_matchday.py --matchday 1 --no-scores          # skip score predictions
"""

import argparse
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from wc.src.ml_predictor import WCPredictor
from wc.src.analytics import captain_picks, find_differentials, squad_optimiser
from wc.src.score_predictor import DCPredictor

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "wc.db"


def _next_matchday() -> int:
    conn = sqlite3.connect(DB_PATH)
    # Find the lowest matchday that hasn't been played yet
    row = conn.execute(
        """
        SELECT MIN(f.matchday) FROM fixtures f
        WHERE NOT EXISTS (
            SELECT 1 FROM matches m
            WHERE m.tournament = 'WC2026'
            AND m.matchday = f.matchday
        )
        """
    ).fetchone()
    conn.close()
    if row and row[0]:
        return row[0]
    # Fall back to max matchday + 1 if no unplayed games found
    conn = sqlite3.connect(DB_PATH)
    row2 = conn.execute("SELECT MAX(matchday) FROM fixtures").fetchone()
    conn.close()
    return (row2[0] or 0) + 1


def _header(title: str) -> None:
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)


def _fmt(df, max_col_width: int = 22) -> str:
    return df.to_string(index=False, max_colwidth=max_col_width, justify="left")


def main():
    parser = argparse.ArgumentParser(description="WC Fantasy matchday runner")
    parser.add_argument("--matchday", type=int, default=0,
                        help="Matchday to predict (default: auto-detect)")
    parser.add_argument("--top", type=int, default=7,
                        help="Top-N captain picks to show (default 7)")
    parser.add_argument("--diff-top", type=int, default=10,
                        help="Top-N differentials to show (default 10)")
    parser.add_argument("--max-ownership", type=float, default=15.0,
                        help="Max ownership %% for differentials (default 15)")
    parser.add_argument("--squad", type=str, default="",
                        help="Comma-separated fantasy_ids of your squad (marks with *)")
    parser.add_argument("--optimise", action="store_true",
                        help="Run group stage squad optimiser")
    parser.add_argument("--budget", type=float, default=100.0,
                        help="Budget in millions for squad optimiser (default 100)")
    parser.add_argument("--no-scores", action="store_true",
                        help="Skip score predictions section")
    args = parser.parse_args()

    matchday = args.matchday if args.matchday > 0 else _next_matchday()
    squad_ids = []
    if args.squad:
        try:
            squad_ids = [int(x.strip()) for x in args.squad.split(",") if x.strip()]
        except ValueError:
            print("[ERROR] --squad must be comma-separated integers")
            sys.exit(1)

    print(f"\nWorld Cup Fantasy — Matchday {matchday}")
    print("Loading model...")

    predictor = WCPredictor()
    try:
        predictor.load()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    predictions = predictor.predict_matchday(matchday)
    print(f"  {len(predictions)} players with fixtures in matchday {matchday}.\n")

    if not args.no_scores:
        _header(f"SCORE PREDICTIONS  — Matchday {matchday}")
        try:
            dc = DCPredictor()
            dc.load()
            score_results = dc.predict_matchday(matchday)
            print(f"\n  {'Home':20s}  {'xG':>5}  {'Win%':>5}  {'Draw%':>5}  {'Loss%':>5}  {'Away':20s}  {'xG':>5}")
            print("  " + "-" * 76)
            for r in score_results:
                top = r["most_likely"][0]
                predicted = f"{top[0]}-{top[1]}"
                print(
                    f"  {r['home_name']:20s}  {r['home_xg']:5.2f}  "
                    f"{r['win_pct']:5.1f}  {r['draw_pct']:5.1f}  {r['loss_pct']:5.1f}  "
                    f"{r['away_name']:20s}  {r['away_xg']:5.2f}  (pred {predicted})"
                )
        except FileNotFoundError:
            print("  [skip] DC model not trained — run: python wc/scripts/train_dc.py")
        except RuntimeError as e:
            print(f"  [skip] {e}")

    _header(f"CAPTAIN PICKS  — Matchday {matchday}")
    print("  * = in your squad\n")
    try:
        cap_df = captain_picks(predictions, top_n=args.top, squad_ids=squad_ids or None)
        print(_fmt(cap_df))
    except Exception as e:
        print(f"  [error] {e}")

    _header(f"DIFFERENTIALS  (own <= {args.max_ownership:.0f}%, top {args.diff_top})")
    try:
        diff_df = find_differentials(predictions, top_n=args.diff_top,
                                     max_ownership=args.max_ownership)
        if diff_df.empty:
            print("  No differentials found with current filters.")
        else:
            print(_fmt(diff_df))
    except Exception as e:
        print(f"  [error] {e}")

    if args.optimise:
        _header(f"GROUP STAGE XI  (budget £{args.budget:.1f}m)")
        budget_tenths = round(args.budget * 10)
        try:
            xi = squad_optimiser(predictions, budget=budget_tenths)
            if xi is not None:
                pos_map = {0: "GK", 1: "DEF", 2: "MID", 3: "FWD"}
                for _, p in xi.sort_values(["pos_enc", "predicted_pts"],
                                           ascending=[True, False]).iterrows():
                    marker = " (C)" if p.get("is_captain") else ""
                    print(f"  {pos_map.get(p['pos_enc'], '?'):3s}  {p['name']:22s}  "
                          f"{p['predicted_pts']:5.2f}{marker}")
        except Exception as e:
            print(f"  [error] {e}")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
