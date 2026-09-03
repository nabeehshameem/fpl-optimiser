"""
gw_postmortem.py
Full-league post-GW analysis: compares every player's actual result to the
model's prediction, flags DNPs, rotation suspects, squad misses, and surfaces
notable non-squad performers.

Run right after grade_model_gw.py finishes:

    python scripts/gw_postmortem.py          # latest graded GW
    python scripts/gw_postmortem.py --gw 2   # specific GW
    python scripts/gw_postmortem.py --gw 2 --top 20
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DB_PATH = PROJECT_ROOT / "data" / "fpl.db"

POS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

# Minutes thresholds
FULL_GAME = 60   # below this = rotation suspect
DNP_MINS  = 0    # exactly 0 = did not play


def _header(title: str) -> None:
    bar = "-" * 62
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)


def run(gw: int | None = None, top_n: int = 15) -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # ── resolve GW ───────────────────────────────────────────────────────────
    if gw is None:
        row = conn.execute(
            "SELECT gameweek_id FROM model_gw_results ORDER BY gameweek_id DESC LIMIT 1"
        ).fetchone()
        if not row:
            print("No graded gameweek found. Run grade_model_gw.py first.")
            return
        gw = int(row[0])

    gw_meta = conn.execute(
        "SELECT gross_points, net_points, hit_points, effective_captain "
        "FROM model_gw_results WHERE gameweek_id = ?", (gw,)
    ).fetchone()
    if not gw_meta:
        print(f"GW{gw} not graded yet. Run grade_model_gw.py --gw {gw} first.")
        return

    avg_row = conn.execute(
        "SELECT average_score FROM gameweeks WHERE gameweek_id = ?", (gw,)
    ).fetchone()
    avg_score = int(avg_row[0]) if avg_row and avg_row[0] is not None else None

    print(f"\n{'='*62}")
    print(f"  GW{gw} POSTMORTEM")
    if avg_score:
        print(f"  Model: {gw_meta['net_points']}pts net  |  GW avg: {avg_score}pts")
    else:
        print(f"  Model: {gw_meta['net_points']}pts net")
    print(f"{'='*62}")

    # ── load squad ───────────────────────────────────────────────────────────
    lock_row = conn.execute(
        "SELECT squad_json, transfers_json FROM model_squad_log WHERE gameweek_id = ?", (gw,)
    ).fetchone()
    squad_meta: dict[int, dict] = {}
    transfer_in: set[int] = set()
    if lock_row:
        for p in json.loads(lock_row["squad_json"]):
            squad_meta[p["player_id"]] = p
        transfers = json.loads(lock_row["transfers_json"])
        transfer_in = set(transfers.get("in", []))
    squad_ids = set(squad_meta.keys())

    # ── latest prediction per player ─────────────────────────────────────────
    pred_rows = conn.execute("""
        SELECT player_id, predicted_points
        FROM predictions
        WHERE gameweek_id = ?
          AND prediction_time = (
              SELECT MAX(prediction_time) FROM predictions p2
              WHERE p2.player_id = predictions.player_id
                AND p2.gameweek_id = predictions.gameweek_id
          )
    """, (gw,)).fetchall()
    preds: dict[int, float] = {int(r["player_id"]): float(r["predicted_points"])
                                for r in pred_rows}

    # ── actuals ───────────────────────────────────────────────────────────────
    hist_rows = conn.execute(
        "SELECT player_id, minutes, total_points "
        "FROM player_gameweek_history WHERE gameweek_id = ?", (gw,)
    ).fetchall()
    actuals: dict[int, tuple[int, int]] = {
        int(r["player_id"]): (int(r["minutes"] or 0), int(r["total_points"] or 0))
        for r in hist_rows
    }

    # ── player info ───────────────────────────────────────────────────────────
    all_ids = list(set(preds) | set(actuals) | squad_ids)
    ph = ",".join("?" * len(all_ids))
    info_rows = conn.execute(
        f"SELECT p.player_id, p.web_name, p.position, t.short_name "
        f"FROM players p LEFT JOIN teams t ON t.team_id = p.team_id "
        f"WHERE p.player_id IN ({ph})", all_ids
    ).fetchall()
    info: dict[int, dict] = {
        int(r["player_id"]): {"name": r["web_name"], "pos": r["position"],
                               "team": r["short_name"] or "???"}
        for r in info_rows
    }
    conn.close()

    def nm(pid: int) -> str:
        i = info.get(pid, {})
        return i.get("name") or f"#{pid}"

    def pos(pid: int) -> str:
        return POS.get(info.get(pid, {}).get("pos"), "?")

    def team(pid: int) -> str:
        return info.get(pid, {}).get("team", "???")

    def mins(pid: int) -> int:
        return actuals.get(pid, (0, 0))[0]

    def pts(pid: int) -> int:
        return actuals.get(pid, (0, 0))[1]

    def xp(pid: int) -> float | None:
        return preds.get(pid)

    def delta(pid: int) -> float | None:
        x = xp(pid)
        if x is None:
            return None
        return pts(pid) - x

    cap_pid = gw_meta["effective_captain"]

    # ── 1. SQUAD SUMMARY ────────────────────────────────────────────────────
    _header(f"SQUAD RESULT  (GW{gw})")
    xi_players = [p for p in squad_meta.values() if p["is_xi"]]
    bench_players = [p for p in squad_meta.values() if not p["is_xi"]]

    def _squad_row(p: dict) -> None:
        pid = p["player_id"]
        role = ("C" if p["is_captain"] else
                "V" if p["is_vice"] else
                " ")
        badge = "(new)" if pid in transfer_in else "     "
        effective_pts = pts(pid) * 2 if pid == cap_pid else pts(pid)
        x = xp(pid)
        xp_str = f"xp={x:.1f}" if x is not None else "xp=  ?"
        d = delta(pid)
        delta_str = (f"Δ{d:+.0f}" if d is not None else "  ??")
        flag = ""
        if mins(pid) == DNP_MINS:
            flag = " [DNP]"
        elif mins(pid) < FULL_GAME:
            flag = f" [{mins(pid)}']"
        name_a = nm(pid).encode("ascii", "replace").decode()
        print(f"  {role} {pos(pid):3} {name_a:20s} {team(pid):3}  "
              f"{pts(pid):3}pts ({xp_str}, {delta_str})  "
              f"{effective_pts:3}eff{badge}{flag}")

    print(f"  {'':2} {'POS':3} {'Name':20s} {'Tm':3}  "
          f"{'Actual':6} {'xp / delta':18} {'Eff':5}")
    print(f"  {'-'*60}")
    print("  -- XI --")
    for p in sorted(xi_players, key=lambda p: -pts(p["player_id"])):
        _squad_row(p)
    print("  -- Bench --")
    for p in sorted(bench_players, key=lambda p: -pts(p["player_id"])):
        _squad_row(p)

    # ── 2. DNPs ──────────────────────────────────────────────────────────────
    # Players who had a prediction (model considered them available) but played 0 mins
    dnps = [
        pid for pid in preds
        if mins(pid) == DNP_MINS and pid in actuals
    ]
    dnps.sort(key=lambda pid: -(xp(pid) or 0))
    if dnps:
        _header("DID NOT PLAY  (had prediction, 0 minutes)")
        squad_dnps = [p for p in dnps if p in squad_ids]
        # Exclude non-squad GKs — backup GKs almost never play and add noise
        other_dnps = [p for p in dnps if p not in squad_ids and pos(p) != "GK"][:20]
        if squad_dnps:
            print("  [SQUAD]")
            for pid in squad_dnps:
                x = xp(pid) or 0
                name_a = nm(pid).encode("ascii", "replace").decode()
                print(f"    {pos(pid):3} {name_a:20s} {team(pid):3}  xp={x:.1f}  0pts  0'")
            print()
        print("  [Notable non-squad DNPs]")
        for pid in other_dnps[:15]:
            x = xp(pid) or 0
            name_a = nm(pid).encode("ascii", "replace").decode()
            print(f"    {pos(pid):3} {name_a:20s} {team(pid):3}  xp={x:.1f}  0pts")

    # ── 3. Rotation suspects ─────────────────────────────────────────────────
    rotation = [
        pid for pid in preds
        if 0 < mins(pid) < FULL_GAME and pos(pid) != "GK"
    ]
    rotation.sort(key=lambda pid: -(xp(pid) or 0))
    if rotation:
        _header(f"ROTATION SUSPECTS  (played < {FULL_GAME} mins)")
        squad_rot = [p for p in rotation if p in squad_ids]
        if squad_rot:
            print("  [SQUAD]")
            for pid in squad_rot:
                x = xp(pid) or 0
                name_a = nm(pid).encode("ascii", "replace").decode()
                print(f"    {pos(pid):3} {name_a:20s} {team(pid):3}  "
                      f"xp={x:.1f}  {pts(pid)}pts  {mins(pid)}'")
            print()
        notable_rot = [p for p in rotation if p not in squad_ids][:10]
        if notable_rot:
            print("  [Notable — monitor for next GW]")
            for pid in notable_rot:
                x = xp(pid) or 0
                name_a = nm(pid).encode("ascii", "replace").decode()
                print(f"    {pos(pid):3} {name_a:20s} {team(pid):3}  "
                      f"xp={x:.1f}  {pts(pid)}pts  {mins(pid)}'")

    # ── 4. Top non-squad scorers ──────────────────────────────────────────────
    non_squad = [pid for pid in actuals if pid not in squad_ids and pts(pid) > 0]
    non_squad.sort(key=lambda pid: -pts(pid))
    _header(f"TOP NON-SQUAD SCORERS  (top {top_n})")
    print(f"  {'POS':3} {'Name':20s} {'Tm':3}  {'Pts':5} {'xp':6}  {'Δ':6}")
    print(f"  {'-'*52}")
    for pid in non_squad[:top_n]:
        x = xp(pid)
        xp_str = f"{x:.1f}" if x is not None else "  ?"
        d = delta(pid)
        d_str = f"{d:+.0f}" if d is not None else "  ?"
        name_a = nm(pid).encode("ascii", "replace").decode()
        print(f"  {pos(pid):3} {name_a:20s} {team(pid):3}  {pts(pid):5}  {xp_str:5}  {d_str:6}")

    # ── 5. Biggest overperformers (vs model) ──────────────────────────────────
    all_with_delta = [
        (pid, pts(pid), xp(pid), delta(pid))
        for pid in actuals
        if pid in preds and pts(pid) >= 3
    ]
    overperformers = sorted(all_with_delta, key=lambda t: -(t[3] or 0))
    _header(f"MODEL MISSES — OVERPERFORMED  (actual vs xp, top {top_n})")
    print(f"  {'SQ':2} {'POS':3} {'Name':20s} {'Tm':3}  {'Pts':5} {'xp':6}  {'Δ':6}")
    print(f"  {'-'*55}")
    for pid, actual, x, d in overperformers[:top_n]:
        sq = "✓" if pid in squad_ids else " "
        xp_str = f"{x:.1f}" if x is not None else "  ?"
        d_str = f"{d:+.0f}" if d is not None else "  ?"
        name_a = nm(pid).encode("ascii", "replace").decode()
        sq_a = "S" if pid in squad_ids else " "
        print(f"  {sq_a}  {pos(pid):3} {name_a:20s} {team(pid):3}  {actual:5}  {xp_str:5}  {d_str:6}")

    # ── 6. Biggest underperformers (model expected big, got small) ────────────
    underperformers = sorted(all_with_delta, key=lambda t: (t[3] or 0))
    squad_underperf = [(pid, a, x, d) for pid, a, x, d in underperformers if pid in squad_ids]
    if squad_underperf:
        _header("SQUAD UNDERPERFORMERS  (expected most, returned least)")
        print(f"  {'POS':3} {'Name':20s} {'Tm':3}  {'Pts':5} {'xp':6}  {'Δ':6}")
        print(f"  {'-'*52}")
        for pid, actual, x, d in squad_underperf[:8]:
            xp_str = f"{x:.1f}" if x is not None else "  ?"
            d_str = f"{d:+.0f}" if d is not None else "  ?"
            name_a = nm(pid).encode("ascii", "replace").decode()
            print(f"  {pos(pid):3} {name_a:20s} {team(pid):3}  {actual:5}  {xp_str:5}  {d_str:6}")

    print(f"\n{'='*62}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gw", type=int, default=None, help="Gameweek to analyse (default: latest graded)")
    ap.add_argument("--top", type=int, default=15, help="Top-N rows for non-squad/overperformer tables")
    args = ap.parse_args()
    run(gw=args.gw, top_n=args.top)
