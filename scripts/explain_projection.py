"""
explain_projection.py

Break a player's projected points into the components that produced it.

Use this whenever a projection looks wrong. A squad locked for a gameweek is
hash-committed and published permanently — an inflated projection that reaches
the lock becomes part of the public record, so "keep an eye on it" is not a
strategy. Decompose it before the deadline.

    python scripts/explain_projection.py --top 10
    python scripts/explain_projection.py --name Kelleher --name Jacob
    python scripts/explain_projection.py --gw 1 --position GK --top 5

Components are recorded by the projection itself (DCPredictor._project's
`explain` hook), not recomputed here — a second implementation would drift
from the real one and could agree with a bug.

Reading the output:
  from_prior=True  the player had no history in the rates source, so every
                   rate is a (position, price-bucket) prior. Expect these to
                   be flat and mid-range; a prior-driven player near the top
                   of the board is a red flag.
  PTS_goals        for a GK this should be ~0.00. Anything else means either
                   a nonzero xG is reaching a keeper or PT_GOAL["GK"] is
                   wrong. Both are bugs.
  PTS_bonus        inherited straight from historical bonus rates; under the
                   2026/27 BPS changes these are known to be stale.
  adjusted_xga     expected goals against. Drives clean sheet, saves and the
                   concede penalty together — if it looks extreme, the DC
                   team parameters or the fixture adjustment are the cause,
                   not the scoring constants.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dc_predictor import DCPredictor, POSITION_MAP  # noqa: E402

DB = PROJECT_ROOT / "data" / "fpl.db"

COMPONENTS = ["PTS_appearance", "PTS_clean_sheet", "PTS_concede", "PTS_goals",
              "PTS_assists", "PTS_saves", "PTS_bonus"]


def next_gw(conn) -> int:
    row = conn.execute(
        "SELECT gameweek_id FROM gameweeks WHERE is_next = 1 LIMIT 1"
    ).fetchone()
    if row:
        return int(row[0])
    row = conn.execute(
        "SELECT MIN(gameweek_id) FROM gameweeks WHERE finished = 0").fetchone()
    return int(row[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gw", type=int, default=None)
    ap.add_argument("--name", action="append", default=[],
                    help="player web_name (repeatable, case-insensitive)")
    ap.add_argument("--position", choices=["GK", "DEF", "MID", "FWD"])
    ap.add_argument("--top", type=int, default=0,
                    help="explain the N highest projections")
    args = ap.parse_args()

    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    gw = args.gw or next_gw(conn)

    pred = DCPredictor()
    df = pred.predict_all(target_gw=gw)

    meta = {int(r["player_id"]): (r["web_name"], POSITION_MAP.get(r["position"], "?"),
                                  r["short_name"], r["current_cost"])
            for r in conn.execute(
                "SELECT p.player_id, p.web_name, p.position, p.current_cost, "
                "t.short_name FROM players p LEFT JOIN teams t "
                "ON t.team_id = p.team_id")}
    conn.close()

    ranked = df.sort_values("predicted_points", ascending=False)
    targets: list[int] = []

    if args.name:
        wanted = {n.lower() for n in args.name}
        targets += [pid for pid, m in meta.items() if m[0].lower() in wanted]
    if args.position:
        pos_ids = [pid for pid in ranked["player_id"]
                   if meta.get(int(pid), ("", "?"))[1] == args.position]
        targets += [int(p) for p in pos_ids[:max(args.top, 5)]]
    if args.top and not args.position:
        targets += [int(p) for p in ranked["player_id"].head(args.top)]
    if not targets:
        targets = [int(p) for p in ranked["player_id"].head(10)]

    targets = list(dict.fromkeys(targets))
    pred._project(gw, explain=set(targets))

    print(f"\nProjection breakdown — GW{gw}\n" + "=" * 78)
    for pid in sorted(targets,
                      key=lambda p: pred.explained.get(p, {}).get("TOTAL", 0),
                      reverse=True):
        e = pred.explained.get(pid)
        name, pos, team, cost = meta.get(pid, (f"#{pid}", "?", "?", 0))
        if not e:
            print(f"\n{name} ({pos}, {team}) — no projection produced")
            continue
        print(f"\n{name}  ({pos}, {team}, £{cost / 10:.1f}m)"
              f"   TOTAL {e['TOTAL']:.2f}"
              + ("   [PRIOR-DRIVEN]" if e["from_prior"] else ""))
        for k in COMPONENTS:
            v = e[k]
            bar = "#" * int(abs(v) * 4)
            flag = ""
            if k == "PTS_goals" and pos == "GK" and v > 0.05:
                flag = "  <-- GK scoring goal points: check PT_GOAL['GK'] and prior xG"
            if k == "PTS_bonus" and v > 2.0:
                flag = "  <-- large bonus term, fitted on last season's BPS"
            print(f"   {k:<16} {v:>7.2f}  {bar}{flag}")
        print(f"   {'—' * 30}")
        print(f"   inputs: start_prob {e['start_prob']}  p60 {e['p60']}  "
              f"xg_adj {e['xg_adj']}  xga_adj {e['xga_adj']}")
        print(f"           adjusted_xga {e['adjusted_xga']}  "
              f"cs_prob {e['cs_prob']}  eff_xg {e['eff_xg']}  eff_xa {e['eff_xa']}")

    print("\nIf a component looks wrong, fix it BEFORE the lock — the squad "
          "published at the deadline is permanent.\n")


if __name__ == "__main__":
    main()
