"""
run_pipeline.py
One command to refresh all WC2026 data, retrain, and push to Railway.

Usage:
  python wc/scripts/run_pipeline.py           # ingest + retrain + push
  python wc/scripts/run_pipeline.py --no-push # skip git push (dry run)
  python wc/scripts/run_pipeline.py --md 2    # show recommendations for matchday
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def header(title: str) -> None:
    print(f"\n{'='*55}\n  {title}\n{'='*55}")


def run_ingest() -> int:
    header("Step 1/3 — Ingest match stats")
    import runpy
    runpy.run_path(
        str(PROJECT_ROOT / "wc" / "scripts" / "ingest_match_stats.py"),
        run_name="__main__",
    )


def run_compute() -> None:
    header("Step 2/3 — Compute fantasy points")
    import runpy
    runpy.run_path(
        str(PROJECT_ROOT / "wc" / "scripts" / "compute_wc2026_points.py"),
        run_name="__main__",
    )


def run_retrain() -> None:
    header("Step 3/3 — Retrain DC model")
    import runpy
    runpy.run_path(
        str(PROJECT_ROOT / "wc" / "scripts" / "train_dc.py"),
        run_name="__main__",
    )


def git_push() -> None:
    header("Committing + pushing to Railway")
    db   = "wc/data/wc.db"
    json = "wc/models/dc_params.json"

    result = subprocess.run(
        ["git", "diff", "--name-only", db, json],
        capture_output=True, text=True, cwd=PROJECT_ROOT,
    )
    changed = [f for f in result.stdout.strip().splitlines() if f]
    if not changed:
        print("  Nothing changed — skipping commit.")
        return

    ts = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    msg = f"Data: pipeline update {ts}"
    subprocess.run(["git", "add", db, json], cwd=PROJECT_ROOT, check=True)
    subprocess.run(["git", "commit", "-m", msg], cwd=PROJECT_ROOT, check=True)
    subprocess.run(["git", "push", "origin", "main"], cwd=PROJECT_ROOT, check=True)
    print("  Pushed. Railway will redeploy in ~60s.")


def show_recommendations(matchday: int) -> None:
    header(f"MD{matchday} Recommendations")
    from wc.src.fantasy_optimizer import captain_picks, optimise

    print(f"\nCaptain picks (MD{matchday}):")
    print(f"  {'#':>2}  {'Player':25s}  {'Team':20s}  {'Pos':3s}  {'£':>4}  {'Pred':>5}  {'Own%':>5}")
    print("  " + "-"*70)
    caps = captain_picks(top_n=10, matchday=matchday)
    for i, p in enumerate(caps, 1):
        own = p.get("ownership") or 0
        print(
            f"  {i:>2}  {p['name']:25s}  {p.get('team','?'):20s}  "
            f"{p['pos']:3s}  {p['price']/10:>4.1f}  "
            f"{p['projected_pts']:>5.2f}  {own:>4.1f}%"
        )

    print(f"\nOptimal squad (MD{matchday}, budget 100m):")
    result = optimise(budget=1000, matchdays=[matchday])
    if not result:
        print("  Optimiser unavailable — pip install pulp")
        return
    captain = result.get("captain", {})
    vc      = result.get("vice_captain", {})
    print(f"  Captain: {captain.get('name','?')}  |  VC: {vc.get('name','?')}")
    print(f"  Expected pts (incl. C bonus): {result.get('expected_pts', 0):.1f}")
    print(f"  Total cost: {result.get('total_cost', 0)/10:.1f}m\n")
    print(f"  {'Player':25s}  {'Pos':3s}  {'Team':20s}  {'£':>4}  {'Pred':>5}  {'':>2}")
    print("  " + "-"*68)
    for p in sorted(result.get("squad", []), key=lambda x: (x.get("pos","MID"), -x.get("projected_pts", 0))):
        flag = "C" if p["name"] == captain.get("name") else ("V" if p["name"] == vc.get("name") else "")
        print(
            f"  {p['name']:25s}  {p['pos']:3s}  {p.get('team','?'):20s}  "
            f"{p['price']/10:>4.1f}  {p['projected_pts']:>5.2f}  {flag:>2}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="WC2026 full pipeline")
    parser.add_argument("--no-push", action="store_true", help="Skip git push")
    parser.add_argument("--md",      type=int, default=None, help="Show recommendations for matchday N after pipeline")
    parser.add_argument("--recs-only", action="store_true", help="Skip pipeline, just show recommendations")
    args = parser.parse_args()

    t0 = time.time()

    if not args.recs_only:
        run_ingest()
        run_compute()
        run_retrain()
        if not args.no_push:
            git_push()
        else:
            print("\n[--no-push] Skipped git push.")

    if args.md:
        show_recommendations(args.md)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s.")


if __name__ == "__main__":
    main()
