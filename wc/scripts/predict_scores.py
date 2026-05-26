"""
predict_scores.py
Predict WC2026 match scores using the Dixon-Coles model.

Prerequisites:
  python wc/scripts/ingest_sb.py       (training data)
  python wc/scripts/train_dc.py        (trains + saves model)
  python wc/scripts/ingest_fixtures.py (WC2026 fixtures)

Usage:
  python wc/scripts/predict_scores.py --matchday 1
  python wc/scripts/predict_scores.py --all          # all group stage matchdays
  python wc/scripts/predict_scores.py --strengths    # team attack/defense table
  python wc/scripts/predict_scores.py --matchday 1 --home-advantage
"""

import argparse
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from wc.src.score_predictor import DCPredictor

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "wc.db"


def _bar(pct: float, width: int = 20) -> str:
    filled = int(round(pct / 100 * width))
    return "█" * filled + "░" * (width - filled)


def _print_match(result: dict) -> None:
    h = result["home_name"]
    a = result["away_name"]
    title = f"{h}  vs  {a}"
    print(f"\n  {title}")
    print(f"  {'─' * len(title)}")

    top = result["most_likely"][0]
    print(f"  Predicted score : {h} {top[0]}-{top[1]} {a}  ({top[2]:.1f}%)")
    print(f"  Expected goals  : {h} {result['home_xg']:.2f}   {a} {result['away_xg']:.2f}")

    w, d, l = result["win_pct"], result["draw_pct"], result["loss_pct"]
    print(f"\n  {'Win':>4}  {'Draw':>5}  {'Loss':>5}")
    print(f"  {w:4.1f}%  {d:5.1f}%  {l:5.1f}%")
    print(f"  {_bar(w)}  {_bar(d)}  {_bar(l)}")

    print(f"\n  Top scorelines:")
    for hg, ag, pct in result["most_likely"][:6]:
        bar = "█" * int(round(pct / 2))
        print(f"    {hg}-{ag}  {pct:5.1f}%  {bar}")


def _print_strengths(predictor: DCPredictor) -> None:
    strengths = predictor.team_strengths()
    if not strengths:
        print("  No team strength data — run train_dc.py first.")
        return
    print(f"\n  {'#':>3}  {'Team':25s}  {'Attack':>8}  {'Defense':>8}")
    print("  " + "─" * 50)
    for i, t in enumerate(strengths, 1):
        print(f"  {i:3d}  {t['name']:25s}  {t['attack']:8.3f}  {t['defense']:8.3f}")


def _all_matchdays() -> list[int]:
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT DISTINCT matchday FROM fixtures ORDER BY matchday"
    ).fetchall()
    conn.close()
    return [r[0] for r in rows]


def _header(title: str) -> None:
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)


def main() -> None:
    parser = argparse.ArgumentParser(description="WC2026 score predictor")
    parser.add_argument("--matchday", type=int, default=0,
                        help="Predict all fixtures in this matchday")
    parser.add_argument("--all", action="store_true",
                        help="Predict all group stage matchdays")
    parser.add_argument("--strengths", action="store_true",
                        help="Print team attack/defense ratings")
    parser.add_argument("--home-advantage", action="store_true",
                        help="Apply home advantage for host nations (USA/CAN/MEX)")
    args = parser.parse_args()

    predictor = DCPredictor()
    try:
        predictor.load()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    if args.strengths:
        _header("TEAM STRENGTHS  (Dixon-Coles attack / defensive weakness)")
        _print_strengths(predictor)
        print()
        return

    matchdays: list[int] = []
    if args.all:
        matchdays = _all_matchdays()
        if not matchdays:
            print("[ERROR] No fixtures in DB. Run ingest_fixtures.py first.")
            sys.exit(1)
    elif args.matchday > 0:
        matchdays = [args.matchday]
    else:
        parser.print_help()
        sys.exit(0)

    for md in matchdays:
        _header(f"SCORE PREDICTIONS  — Matchday {md}")
        try:
            results = predictor.predict_matchday(md)
        except RuntimeError as e:
            print(f"  [ERROR] {e}")
            continue

        for r in results:
            _print_match(r)

        # Matchday summary table
        print(f"\n  {'Home':20s}  {'xG':>5}  {'Win%':>5}  {'Draw%':>5}  {'Loss%':>5}  {'Away':20s}  {'xG':>5}")
        print("  " + "─" * 76)
        for r in results:
            print(
                f"  {r['home_name']:20s}  {r['home_xg']:5.2f}  "
                f"{r['win_pct']:5.1f}  {r['draw_pct']:5.1f}  {r['loss_pct']:5.1f}  "
                f"{r['away_name']:20s}  {r['away_xg']:5.2f}"
            )

    print(f"\n{'=' * 60}\n")


if __name__ == "__main__":
    main()
