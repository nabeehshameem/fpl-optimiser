"""
report_batch.py  —  Post-match batch report for WC2026

Shows:
  1. Recent results vs model prediction
  2. Live group standings (actual results)
  3. Bracket advancement % (Monte Carlo)
  4. Fantasy points leaderboard (actual)
  5. Projected vs actual player points
"""

import io
import sqlite3
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DB_PATH = str(PROJECT_ROOT / "wc" / "data" / "wc.db")

def _canon(name: str) -> str:
    """Delegate to score_predictor's canonical normaliser so all name lookups are consistent."""
    from wc.src.score_predictor import _canonical
    return _canonical(name)


def _div(title: str) -> None:
    print(f"\n{'-'*60}")
    print(f"  {title}")
    print(f"{'-'*60}")


# ── 1. Results vs prediction ──────────────────────────────────────────────────

def report_results(predictor, since: str | None = None) -> None:
    _div("RESULTS vs MODEL PREDICTION")

    conn = sqlite3.connect(DB_PATH)

    # Build a canonical-name → team_id map so we handle all API name variants
    team_id_map: dict[str, int] = {
        _canon(name): tid
        for tid, name in conn.execute("SELECT team_id, name FROM teams").fetchall()
    }

    # Build fixture lookup: (home_id, away_id) → (group_id, matchday)
    fixture_info: dict[tuple[int, int], tuple[str, int]] = {
        (h, a): (g, md)
        for h, a, g, md in conn.execute(
            "SELECT home_team_id, away_team_id, group_id, matchday FROM fixtures"
        ).fetchall()
    }
    fixture_group = {k: v[0] for k, v in fixture_info.items()}

    where = f"WHERE match_date >= '{since}'" if since else ""
    rows = conn.execute(f"""
        SELECT match_date, home_team, away_team, home_score, away_score
        FROM match_stats {where}
        ORDER BY match_date, home_team
    """).fetchall()
    conn.close()

    print(f"\n  {'Date':>10}  {'Match':^35}  {'Actual':^7}  {'xG':^9}  {'Correct':^8}  {'Home%':>5}  {'Draw%':>5}  {'Away%':>5}")
    print("  " + "-" * 95)

    results: list[str] = []
    for match_date, home, away, hs, as_ in rows:
        home_id = team_id_map.get(_canon(home))
        away_id = team_id_map.get(_canon(away))
        group_id = fixture_group.get((home_id, away_id)) if home_id and away_id else None

        try:
            info = fixture_info.get((home_id, away_id)) if home_id and away_id else None
            md_num = info[1] if info else 0
            draw_boost = 0.10 if md_num == 1 else 0.0
            pred = predictor.predict(home_id, away_id, home_advantage=False, draw_boost=draw_boost)
            pred_score = f"{pred['home_xg']:.1f}-{pred['away_xg']:.1f}"
            win_pct  = pred["win_pct"]
            draw_pct = pred["draw_pct"]
            loss_pct = pred["loss_pct"]

            if hs > as_:
                actual_outcome = "H"
            elif hs == as_:
                actual_outcome = "D"
            else:
                actual_outcome = "A"

            if win_pct > draw_pct and win_pct > loss_pct:
                pred_outcome = "H"
            elif draw_pct >= win_pct and draw_pct >= loss_pct:
                pred_outcome = "D"
            else:
                pred_outcome = "A"

            correct = "Y" if pred_outcome == actual_outcome else "N"
            match_str = f"{home} {hs}-{as_} {away}"
        except Exception:
            pred_score, win_pct, draw_pct, loss_pct = "?-?", 0, 0, 0
            correct, match_str = "?", f"{home} {hs}-{as_} {away}"

        gstr = f"[Grp {group_id}]" if group_id else "       "
        print(
            f"  {match_date}  {gstr} {match_str:<28}  {hs}-{as_}     {pred_score}    "
            f"  {correct}        {win_pct:>4.1f}%  {draw_pct:>4.1f}%  {loss_pct:>4.1f}%"
        )
        results.append(correct)

    correct_count = results.count("Y")
    total = len(results)
    if total:
        print(f"\n  Accuracy: {correct_count}/{total} ({100*correct_count/total:.1f}%)  "
              f"[outcome W/D/L — correct = most likely outcome matched result]")


# ── 2. Live group standings ───────────────────────────────────────────────────

def report_group_standings() -> None:
    _div("LIVE GROUP STANDINGS")

    from wc.src.score_predictor import DCPredictor, _canonical
    WC2026_GROUPS = DCPredictor.WC2026_GROUPS

    conn = sqlite3.connect(DB_PATH)
    played = conn.execute(
        "SELECT home_team, away_team, home_score, away_score FROM match_stats"
    ).fetchall()
    conn.close()

    played_map: dict[tuple[str, str], tuple[int, int]] = {}
    for h, a, hs, as_ in played:
        played_map[(_canon(h), _canon(a))] = (int(hs), int(as_))

    for group_name, teams in WC2026_GROUPS.items():
        keys = [_canonical(t) for t in teams]
        display = {_canonical(t): t for t in teams}
        stats: dict[str, list[int]] = {k: [0, 0, 0, 0, 0, 0] for k in keys}
        # [pts, gd, gf, w, d, l]

        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                h, a = keys[i], keys[j]
                result = played_map.get((h, a)) or played_map.get((a, h))
                if result is None:
                    continue
                hg, ag = result if (h, a) in played_map else (result[1], result[0])
                diff = hg - ag
                stats[h][2] += hg; stats[h][1] += diff; stats[a][2] += ag; stats[a][1] -= diff
                if diff > 0:
                    stats[h][0] += 3; stats[h][3] += 1; stats[a][5] += 1
                elif diff < 0:
                    stats[a][0] += 3; stats[a][3] += 1; stats[h][5] += 1
                else:
                    stats[h][0] += 1; stats[h][4] += 1
                    stats[a][0] += 1; stats[a][4] += 1

        ranked = sorted(keys, key=lambda k: (stats[k][0], stats[k][1], stats[k][2]), reverse=True)

        print(f"\n  Group {group_name}")
        print(f"  {'Team':22s}  Pts  W  D  L  GD  GF  Played")
        for k in ranked:
            s = stats[k]
            played_count = s[3] + s[4] + s[5]
            print(
                f"  {display[k]:22s}  {s[0]:>3}  {s[3]}  {s[4]}  {s[5]}  "
                f"{s[1]:>+3}  {s[2]:>2}  {played_count}/3"
            )


# ── 3. Bracket % ─────────────────────────────────────────────────────────────

def report_bracket(predictor) -> None:
    _div("BRACKET ADVANCEMENT %  (50k simulations)")

    results = predictor.simulate_tournament(n_sim=50_000)

    WC2026_GROUPS = predictor.WC2026_GROUPS
    group_of = {
        t.lower(): g
        for g, teams in WC2026_GROUPS.items()
        for t in teams
    }

    by_group: dict[str, list] = {}
    for r in results:
        g = r.get("group", "?")
        by_group.setdefault(g, []).append(r)

    print(f"\n  {'Team':22s}  Grp  R32%   QF%   SF%  Final%   Win%")
    print("  " + "-" * 65)

    for g in sorted(by_group.keys()):
        for r in by_group[g]:
            print(
                f"  {r['team']:22s}  {r['group']:>3}  "
                f"{r['r32_pct']:>5.1f}  {r['qf_pct']:>5.1f}  "
                f"{r['sf_pct']:>4.1f}  {r['final_pct']:>5.1f}  {r['win_pct']:>5.1f}"
            )
        print()


# ── 4. Fantasy points leaderboard ────────────────────────────────────────────

def report_fantasy_points(top_n: int = 30) -> None:
    _div(f"FANTASY POINTS LEADERBOARD  (top {top_n})")

    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(f"""
        SELECT w.player_name, w.team_name, w.position,
               SUM(w.fantasy_pts) AS total_pts,
               SUM(w.goals) AS goals, SUM(w.assists) AS assists,
               SUM(w.minutes) AS mins,
               fp.price, fp.ownership
        FROM wc2026_player_points w
        LEFT JOIN fantasy_players fp
               ON LOWER(fp.name) = LOWER(w.player_name)
        GROUP BY w.player_name, w.team_name
        ORDER BY total_pts DESC
        LIMIT {top_n}
    """).fetchall()
    conn.close()

    print(f"\n  {'#':>2}  {'Player':25s}  {'Team':20s}  {'Pos':3s}  {'Pts':>5}  "
          f"{'G':>2}  {'A':>2}  {'Mins':>4}  {'£':>4}  {'Own%':>5}")
    print("  " + "-" * 85)

    for i, (name, team, pos, pts, g, a, mins, price, own) in enumerate(rows, 1):
        price_str = f"{price/10:.1f}" if price else " -  "
        own_str   = f"{own:.1f}%" if own is not None else "  -  "
        print(
            f"  {i:>2}  {name:25s}  {team:20s}  {pos:3s}  {pts:>5.1f}  "
            f"{g:>2}  {a:>2}  {mins:>4}  {price_str:>4}  {own_str:>5}"
        )


# ── 5. Projected vs actual ───────────────────────────────────────────────────

def report_projected_vs_actual(predictor, matchday: int | None = None) -> None:
    from wc.src.fantasy_optimizer import get_projected_players

    # Determine which matchdays have actual data so projection scope matches.
    conn = sqlite3.connect(DB_PATH)
    played_fixture_ids = [r[0] for r in conn.execute(
        "SELECT DISTINCT api_fixture_id FROM wc2026_player_points"
    ).fetchall()]
    # Map played fixtures → matchday numbers via the fixtures table (name-match join).
    md_rows = conn.execute("""
        SELECT DISTINCT f.matchday
        FROM fixtures f
        JOIN teams t1 ON f.home_team_id = t1.team_id
        JOIN teams t2 ON f.away_team_id = t2.team_id
        JOIN match_stats ms ON LOWER(t1.name) = LOWER(ms.home_team)
                           AND LOWER(t2.name) = LOWER(ms.away_team)
        ORDER BY f.matchday
    """).fetchall()
    played_mds = [r[0] for r in md_rows] or [1, 2]

    _div(f"PROJECTED vs ACTUAL  (MD{','.join(str(m) for m in played_mds)} scope)")

    from wc.src.fantasy_optimizer import get_projected_players

    projected_map: dict[str, float] = {}
    try:
        for p in get_projected_players(predictor, matchdays=played_mds):
            projected_map[p["name"].lower()] = p["projected_pts"]
    except Exception as e:
        print(f"  Could not load projections: {e}")
        conn.close()
        return

    query = """
        SELECT w.player_name, w.team_name, w.position,
               SUM(w.fantasy_pts) AS actual_pts,
               fp.price, fp.ownership
        FROM wc2026_player_points w
        JOIN fantasy_players fp ON LOWER(fp.name) = LOWER(w.player_name)
        GROUP BY w.player_name, w.team_name
        HAVING actual_pts > 0
        ORDER BY actual_pts DESC
    """
    rows = conn.execute(query).fetchall()
    conn.close()

    matched = []
    for name, team, pos, actual, price, own in rows:
        proj = projected_map.get(name.lower())
        if proj is not None:
            matched.append((name, team, pos, actual, proj, actual - proj, price, own))

    if not matched:
        print("  No matched players found (name matching may need improvement).")
        return

    print(f"\n  {'Player':25s}  {'Team':18s}  {'Pos':3s}  {'Actual':>6}  {'Proj':>6}  {'Diff':>6}  {'£':>4}")
    print("  " + "-" * 80)

    for name, team, pos, actual, proj, diff, price, own in sorted(matched, key=lambda x: -x[3]):
        diff_str = f"{diff:>+.1f}"
        print(
            f"  {name:25s}  {team:18s}  {pos:3s}  {actual:>6.1f}  {proj:>6.2f}  "
            f"{diff_str:>6}  {price/10 if price else 0:>4.1f}"
        )


# ── Main ─────────────────────────────────────────────────────────────────────

def run_report(since: str | None = None, matchday: int | None = None) -> None:
    from wc.src.score_predictor import DCPredictor
    predictor = DCPredictor()
    predictor.load()

    since_label = f" since {since}" if since else ""
    print(f"\n{'='*60}")
    print(f"  WC2026 BATCH REPORT{since_label}")
    print(f"{'='*60}")

    report_results(predictor, since=since)
    report_group_standings()
    report_bracket(predictor)
    report_fantasy_points()
    report_projected_vs_actual(predictor, matchday=matchday)

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="WC2026 batch report")
    parser.add_argument("--since", default=None, help="Only show results from this date (YYYY-MM-DD)")
    parser.add_argument("--md", type=int, default=None, help="Matchday context for projections")
    args = parser.parse_args()
    run_report(since=args.since, matchday=args.md)
