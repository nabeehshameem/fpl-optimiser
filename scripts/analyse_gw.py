"""
analyse_gw.py
Full analysis suite for the upcoming gameweek.

Sections printed:
  1. Captain picks
  2. Differentials (low-ownership, high-upside)
  3. Fixture ticker (next 5 GWs)
  4. Bonus point candidates
  5. Price change signals
  6. Player comparison        [opt: --compare ID1,ID2,...]
  7. Chip planner             [opt: --squad ..., --chips ...]

Usage:
  python scripts/analyse_gw.py
  python scripts/analyse_gw.py --squad 123,456,...
  python scripts/analyse_gw.py --compare 233,302
  python scripts/analyse_gw.py --squad 123,... --chips TC,BB,FH
  python scripts/analyse_gw.py --squad 123,... --chip-horizon 6
  python scripts/analyse_gw.py --top 15              # show top-N differentials / bonus
"""

import argparse
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml_predictor import LightGBMPredictor
from src.analytics import (
    captain_picks,
    find_differentials,
    fixture_ticker,
    bonus_point_predictions,
    price_change_predictions,
    compare_players,
)


# ---------- helpers ----------

def _header(title: str) -> None:
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)


def _get_target_gw() -> int:
    conn = sqlite3.connect(PROJECT_ROOT / "data" / "fpl.db")
    row = conn.execute(
        "SELECT gameweek_id FROM gameweeks WHERE is_next = 1 LIMIT 1"
    ).fetchone()
    conn.close()
    if row:
        return row[0]
    raise RuntimeError("No upcoming gameweek found in DB.")


def _fmt_df(df, max_col_width: int = 22) -> str:
    """Pretty-print a DataFrame with truncated column values."""
    return df.to_string(
        index=True,
        max_colwidth=max_col_width,
        justify="left",
    )


# ---------- main ----------

def main() -> None:
    parser = argparse.ArgumentParser(description="FPL gameweek analysis suite")
    parser.add_argument("--squad", type=str, default="",
                        help="Comma-separated player_ids of your 15-player squad")
    parser.add_argument("--compare", type=str, default="",
                        help="Comma-separated player_ids to compare head-to-head")
    parser.add_argument("--chips", type=str, default="TC,BB,FH,WC",
                        help="Which chips to plan: comma-separated from TC,BB,FH,WC")
    parser.add_argument("--chip-horizon", type=int, default=5,
                        help="How many GWs ahead to score chips (default 5)")
    parser.add_argument("--top", type=int, default=10,
                        help="Top-N rows for differentials and bonus tables (default 10)")
    parser.add_argument("--ticker-gws", type=int, default=5,
                        help="Number of GWs in fixture ticker (default 5)")
    args = parser.parse_args()

    squad_ids: list[int] = []
    if args.squad:
        try:
            squad_ids = [int(x.strip()) for x in args.squad.split(",") if x.strip()]
        except ValueError:
            print("[ERROR] --squad must be comma-separated integers")
            sys.exit(1)

    compare_ids: list[int] = []
    if args.compare:
        try:
            compare_ids = [int(x.strip()) for x in args.compare.split(",") if x.strip()]
        except ValueError:
            print("[ERROR] --compare must be comma-separated integers")
            sys.exit(1)

    chips = [c.strip().lower() for c in args.chips.split(",") if c.strip()]

    target_gw = _get_target_gw()
    print(f"\nFPL Analysis — GW{target_gw}")

    print("\nLoading model and predictions...")
    predictor = LightGBMPredictor()
    predictor.load()
    predictions_df = predictor.predict_all(target_gw=target_gw)
    print(f"  {len(predictions_df)} players loaded.\n")

    # ── 1. Captain Picks ─────────────────────────────────────────────────
    _header("CAPTAIN PICKS")
    try:
        cap_df = captain_picks(predictions_df, top_n=7)
        print(_fmt_df(cap_df))
    except Exception as e:
        print(f"  [error] {e}")

    # ── 2. Differentials ─────────────────────────────────────────────────
    _header(f"DIFFERENTIALS  (ownership <= 15%, top {args.top})")
    try:
        diff_df = find_differentials(predictions_df, top_n=args.top)
        if diff_df.empty:
            print("  No differentials found with current filters.")
        else:
            print(_fmt_df(diff_df))
    except Exception as e:
        print(f"  [error] {e}")

    # ── 3. Fixture Ticker ─────────────────────────────────────────────────
    _header(f"FIXTURE TICKER  (GW{target_gw}-GW{target_gw + args.ticker_gws - 1})")
    print("  Format: H(fdr) = home, A(fdr) = away. DGW shows both fixtures.")
    print("  FDR: 1=easy ... 5=hard\n")
    try:
        ticker_df = fixture_ticker(target_gw, num_gws=args.ticker_gws)
        print(_fmt_df(ticker_df, max_col_width=20))
    except Exception as e:
        print(f"  [error] {e}")

    # ── 4. Bonus Point Candidates ─────────────────────────────────────────
    _header(f"BONUS POINT CANDIDATES  (top {args.top} by avg BPS)")
    try:
        bonus_df = bonus_point_predictions(predictions_df, top_n=args.top)
        if bonus_df.empty:
            print("  No bonus data available.")
        else:
            print(_fmt_df(bonus_df))
    except Exception as e:
        print(f"  [error] {e}")

    # ── 5. Price Change Signals ───────────────────────────────────────────
    _header("PRICE CHANGE SIGNALS")
    print("  Based on ownership delta between last two snapshots.")
    print("  RISE ^ = buying pressure.  FALL v = selling pressure.\n")
    try:
        price_df = price_change_predictions(top_n=20)
        if price_df.empty:
            print("  Need at least 2 snapshot runs to compute deltas.")
        else:
            print(_fmt_df(price_df))
    except Exception as e:
        print(f"  [error] {e}")

    # ── 6. Player Comparison ──────────────────────────────────────────────
    if compare_ids:
        _header(f"PLAYER COMPARISON  ({', '.join(map(str, compare_ids))})")
        try:
            cmp_df = compare_players(compare_ids, predictions_df)
            print(_fmt_df(cmp_df))
        except Exception as e:
            print(f"  [error] {e}")

    # ── 7. Chip Planner ───────────────────────────────────────────────────
    chip_needs_squad = bool(set(chips) & {"bb", "fh", "wc"})
    if chips and (not chip_needs_squad or squad_ids):
        _header(f"CHIP PLANNER  (horizon: {args.chip_horizon} GWs)")
        if chip_needs_squad and not squad_ids:
            print("  Pass --squad to enable BB / FH / WC planning.")
        else:
            from src.chip_planner import plan_chips, print_chip_plan
            try:
                plan = plan_chips(
                    current_squad=squad_ids if squad_ids else None,
                    chips_available=chips,
                    horizon=args.chip_horizon,
                    current_gw=target_gw,
                )
                if not plan:
                    print("  No chip recommendations generated.")
                else:
                    print_chip_plan(plan)
            except Exception as e:
                print(f"  [error] {e}")
    elif chips and chip_needs_squad and not squad_ids:
        _header("CHIP PLANNER")
        print("  Pass --squad ID1,ID2,... to enable chip planning.")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
