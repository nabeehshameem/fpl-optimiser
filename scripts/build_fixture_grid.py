"""build_fixture_grid.py

Generate the GW1-N fixture grid for the FPL preview page from the database and
the model's own Dixon-Coles parameters, and splice it into the page in place.

    python scripts/build_fixture_grid.py \\
        --page ../themodelsays-web/public/fpl-preview.html --from-gw 1 --to-gw 6

WHY THIS IS GENERATED

Two reasons, both learned the hard way on this project.

1. Hand-typed data drifts. Fixtures get rescheduled, promoted sides get
   misremembered (this project has produced three different lists of the 2026/27
   promoted clubs in as many conversations), and nothing catches it. Every other
   number on the site now comes from a committed artifact; this was the last
   hand-typed table.

2. The page described its colour coding as "the model's relative difficulty
   assessment" while the colours were assigned by reputation. That made a claim
   the artifacts did not support. Now the colour IS the model's assessment: it
   is the expected goals the Dixon-Coles parameters give that team in that
   fixture, from the same file the player projections use. The sentence becomes
   true by construction rather than by good intentions.

The generator edits only between the markers, so the surrounding editorial prose
is untouched:

    <!-- FIXTURE_GRID:START -->  ... generated ...  <!-- FIXTURE_GRID:END -->

If the markers are absent it prints the fragment to stdout instead of guessing
where it belongs.
"""

from __future__ import annotations

import argparse
import html
import json
import sqlite3
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "fpl.db"
DC_PARAMS = PROJECT_ROOT / "models" / "fpl_dc_params.json"

START = "<!-- FIXTURE_GRID:START -->"
END = "<!-- FIXTURE_GRID:END -->"


def load_dc() -> tuple[dict, float, float, float]:
    if not DC_PARAMS.exists():
        raise SystemExit(
            f"missing {DC_PARAMS} - run train_dc.py first. Refusing to publish "
            "a grid whose colours claim to be the model's assessment while "
            "being invented here.")
    d = json.loads(DC_PARAMS.read_text(encoding="utf-8"))
    tp = d.get("team_params", {})
    if not tp:
        raise SystemExit("DC parameters contain no team_params")
    mean_atk = sum(v["attack"] for v in tp.values()) / len(tp)
    mean_def = sum(v["defense"] for v in tp.values()) / len(tp)
    return tp, mean_atk, mean_def, float(d.get("home_adv", 1.20))


def expected_goals(tp: dict, mean_atk: float, mean_def: float, home_adv: float,
                   team_id: int, opp_id: int, at_home: bool) -> float:
    """Goals the model expects `team_id` to score against `opp_id`."""
    t = tp.get(str(team_id), {"attack": mean_atk, "defense": mean_def})
    o = tp.get(str(opp_id), {"attack": mean_atk, "defense": mean_def})
    mu = t["attack"] * o["defense"]
    return mu * home_adv if at_home else mu


def fetch(conn, from_gw: int, to_gw: int):
    teams = {int(r[0]): r[1] for r in
             conn.execute("SELECT team_id, short_name FROM teams")}
    if not teams:
        raise SystemExit("no teams in the database - run ingest_bootstrap.py")
    rows = conn.execute(
        "SELECT gameweek_id, home_team_id, away_team_id FROM fixtures "
        "WHERE gameweek_id BETWEEN ? AND ? ORDER BY gameweek_id",
        (from_gw, to_gw)).fetchall()
    if not rows:
        raise SystemExit(
            f"no fixtures for GW{from_gw}-{to_gw} - run ingest_fixtures.py")
    return teams, rows


def build(from_gw: int, to_gw: int, db_path: Path = DB_PATH) -> str:
    tp, mean_atk, mean_def, home_adv = load_dc()
    conn = sqlite3.connect(db_path)
    try:
        teams, rows = fetch(conn, from_gw, to_gw)
    finally:
        conn.close()

    # team_id -> {gw: (opponent_short, at_home, expected_goals_for)}
    grid: dict[int, dict[int, tuple[str, bool, float]]] = {t: {} for t in teams}
    for gw, h, a in rows:
        gw, h, a = int(gw), int(h), int(a)
        if h in grid:
            grid[h][gw] = (teams.get(a, "?"), True,
                           expected_goals(tp, mean_atk, mean_def, home_adv, h, a, True))
        if a in grid:
            grid[a][gw] = (teams.get(h, "?"), False,
                           expected_goals(tp, mean_atk, mean_def, home_adv, a, h, False))

    gws = list(range(from_gw, to_gw + 1))
    # Rank teams by total expected goals over the window — the model's own view
    # of who has the best run, rather than a hand-picked shortlist.
    ranked = sorted(
        grid.items(),
        key=lambda kv: sum(v[2] for v in kv[1].values()), reverse=True)

    all_xg = [v[2] for cells in grid.values() for v in cells.values()]
    lo, hi = (min(all_xg), max(all_xg)) if all_xg else (0.0, 1.0)
    span = (hi - lo) or 1.0

    def cell(entry) -> str:
        if entry is None:
            return '<td class="fx-none">-</td>'
        opp, at_home, xg = entry
        # 0 = worst run in the window, 1 = best. Continuous, so no arbitrary
        # "easy/hard" cut-off has to be defended.
        t = (xg - lo) / span
        return (f'<td class="fx" style="--t:{t:.3f}">'
                f'<span class="fx-opp">{html.escape(opp)}</span>'
                f'<span class="fx-ha">{"H" if at_home else "A"}</span>'
                f'<span class="fx-xg">{xg:.2f}</span></td>')

    head = "".join(f"<th>GW{g}</th>" for g in gws)
    body = "\n".join(
        f'<tr><th class="fx-team">{html.escape(teams[tid])}</th>'
        + "".join(cell(cells.get(g)) for g in gws)
        + f'<td class="fx-total">{sum(v[2] for v in cells.values()):.1f}</td></tr>'
        for tid, cells in ranked)

    return f"""{START}
<!-- generated by scripts/build_fixture_grid.py on {date.today():%Y-%m-%d} - do not hand-edit -->
<style>
.fx-wrap{{overflow-x:auto}}
table.fx-grid{{width:100%;border-collapse:collapse;font-size:13px;margin:18px 0}}
table.fx-grid th,table.fx-grid td{{padding:7px 8px;text-align:center;
border-bottom:1px solid rgba(255,255,255,.08)}}
table.fx-grid th{{color:#796a93;font-size:11px;text-transform:uppercase;
letter-spacing:.06em}}
th.fx-team{{text-align:left;color:#fff;font-size:13px;text-transform:none;
letter-spacing:0}}
td.fx{{background:color-mix(in srgb,#00FF87 calc(var(--t)*26%),transparent)}}
.fx-opp{{color:#fff;font-weight:600}}
.fx-ha{{color:#796a93;font-size:10px;margin-left:3px}}
.fx-xg{{display:block;color:#b9aed0;font-size:10px;font-variant-numeric:tabular-nums}}
td.fx-total{{color:#00FF87;font-weight:700;font-variant-numeric:tabular-nums}}
td.fx-none{{color:#796a93}}
</style>
<div class="fx-wrap">
<table class="fx-grid">
<thead><tr><th class="fx-team">Team</th>{head}<th>xG</th></tr></thead>
<tbody>
{body}
</tbody>
</table>
</div>
<p class="fx-note">Each cell shows the opponent, home or away, and the goals the
model expects that team to score in that fixture. Shading is relative to the
best and worst runs in this window; the final column totals expected goals
across GW{from_gw}-{to_gw}. These come from the same Dixon-Coles parameters as
the player projections, so the ordering is the model's view of the fixture run,
not a hand-assigned difficulty rating. It reflects attacking output only - a
good run for a striker is not automatically a good run for a defender.</p>
{END}"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--page", type=Path, default=None,
                    help="HTML file containing the FIXTURE_GRID markers")
    ap.add_argument("--from-gw", type=int, default=1)
    ap.add_argument("--to-gw", type=int, default=6)
    ap.add_argument("--db", type=Path, default=DB_PATH)
    args = ap.parse_args()

    fragment = build(args.from_gw, args.to_gw, args.db)

    if args.page is None:
        print(fragment)
        return
    if not args.page.exists():
        raise SystemExit(f"{args.page} does not exist")

    doc = args.page.read_text(encoding="utf-8")
    if START not in doc or END not in doc:
        print(fragment)
        print(f"\n--- {args.page.name} has no FIXTURE_GRID markers. Paste the "
              "block above where the grid belongs, keeping both marker "
              "comments, and this script will maintain it from then on.",
              file=sys.stderr)
        raise SystemExit(1)

    before = doc[:doc.index(START)]
    after = doc[doc.index(END) + len(END):]
    args.page.write_text(before + fragment + after, encoding="utf-8")
    print(f"updated fixture grid in {args.page} (GW{args.from_gw}-{args.to_gw})")


if __name__ == "__main__":
    main()
