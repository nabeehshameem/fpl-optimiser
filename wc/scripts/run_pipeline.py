"""
run_pipeline.py
One command to refresh all WC2026 data after each batch of match results.

Pipeline (runs automatically in order):
  1. Ingest match stats from API      (ingest_match_stats.py)
  2. Compute fantasy points           (compute_wc2026_points.py)
  3. Retrain Dixon-Coles model        (train_dc.py)
  4. Show batch report                (standings, bracket %, fantasy pts)
  5. Commit wc.db + dc_params.json and push to Railway

Usage:
  python wc/scripts/run_pipeline.py           # full pipeline + report + push
  python wc/scripts/run_pipeline.py --no-push # skip git push
  python wc/scripts/run_pipeline.py --md 3    # also show MD3 recommendations
  python wc/scripts/run_pipeline.py --since 2026-06-20  # filter report to recent matches
  python wc/scripts/run_pipeline.py --recs-only --md 3  # skip pipeline, just recommendations
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


def _run_script(path: str) -> None:
    """Run a script via runpy with a clean sys.argv so its argparse doesn't choke."""
    import runpy
    saved_argv = sys.argv[:]
    sys.argv = [path]
    try:
        runpy.run_path(path, run_name="__main__")
    finally:
        sys.argv = saved_argv


def check_form_coverage() -> None:
    header("Pre-flight — Form coverage check")
    import sqlite3
    from datetime import date, timedelta

    db = PROJECT_ROOT / "wc" / "data" / "wc.db"
    conn = sqlite3.connect(str(db))
    cutoff = str(date.today() - timedelta(days=90))

    wc_teams = conn.execute("""
        SELECT DISTINCT t.name FROM teams t
        JOIN fixtures f ON t.team_id = f.home_team_id OR t.team_id = f.away_team_id
        ORDER BY t.name
    """).fetchall()

    flagged = []
    for (name,) in wc_teams:
        count = conn.execute(
            "SELECT COUNT(*) FROM recent_results "
            "WHERE (home_team LIKE ? OR away_team LIKE ?) AND match_date >= ?",
            (f"%{name}%", f"%{name}%", cutoff),
        ).fetchone()[0]
        if count < 3:
            flagged.append((name, count))
    conn.close()

    if flagged:
        print(f"  WARNING: {len(flagged)} WC teams have < 3 matches in the last 90 days:")
        for name, count in flagged:
            label = "NO DATA" if count == 0 else f"{count} match{'es' if count != 1 else ''}"
            print(f"    {name}: {label}")
        print("  Consider running ingest_recent_form.py to fill gaps.")
    else:
        print(f"  All {len(wc_teams)} WC teams have >= 3 matches in last 90 days. OK.")


def run_ingest() -> None:
    header("Step 1/4 — Ingest match stats")
    _run_script(str(PROJECT_ROOT / "wc" / "scripts" / "ingest_match_stats.py"))


def run_compute() -> None:
    header("Step 2/4 — Compute fantasy points")
    _run_script(str(PROJECT_ROOT / "wc" / "scripts" / "compute_wc2026_points.py"))


def run_retrain() -> None:
    header("Step 3/4 — Retrain DC model")
    _run_script(str(PROJECT_ROOT / "wc" / "scripts" / "train_dc.py"))


def run_report(since: str | None = None, matchday: int | None = None) -> None:
    header("Step 4/4 — Batch report")
    from wc.scripts.report_batch import run_report as _report
    _report(since=since, matchday=matchday)


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
    parser = argparse.ArgumentParser(description="WC2026 full pipeline — ingest → compute → retrain → report → push")
    parser.add_argument("--no-push",   action="store_true", help="Skip git push")
    parser.add_argument("--no-report", action="store_true", help="Skip batch report")
    parser.add_argument("--md",        type=int, default=None, help="Also show fantasy recommendations for matchday N")
    parser.add_argument("--recs-only", action="store_true", help="Skip pipeline, just show recommendations (requires --md)")
    parser.add_argument("--since",     default=None, help="Filter report to results since YYYY-MM-DD")
    args = parser.parse_args()

    t0 = time.time()

    if not args.recs_only:
        check_form_coverage()
        run_ingest()
        run_compute()
        run_retrain()
        if not args.no_report:
            run_report(since=args.since, matchday=args.md)
        if not args.no_push:
            git_push()
        else:
            print("\n[--no-push] Skipped git push.")

    if args.recs_only and args.md:
        show_recommendations(args.md)
    elif args.md and not args.recs_only:
        show_recommendations(args.md)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s.")


if __name__ == "__main__":
    main()
