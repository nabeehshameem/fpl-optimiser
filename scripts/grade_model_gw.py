"""
Grade the model's locked squad after a gameweek finishes. This produces the
number the receipt card shows, so the rules here are real FPL rules:

  * Captain scores double. If the captain played 0 minutes, the armband passes
    to the vice; if both blanked, nobody doubles.
  * Auto-subs: a starter with 0 minutes is replaced from the bench.
      - GK: only the bench GK can replace the starting GK.
      - Outfield: bench players tried in bench_order (slots 2-4); a bench
        player comes in only if they played >0 minutes AND the resulting
        formation stays legal (>=3 DEF, >=2 MID, >=1 FWD).
  * Transfer hits recorded at lock time are subtracted.

Results are written append-only to model_gw_results and exported to
predictions/fpl/gwNN_result.json (commit it — the graded receipt is part of
the public record, same as the lock).

Usage:
  python scripts/grade_model_gw.py            # grade latest finished, locked GW
  python scripts/grade_model_gw.py --gw 3
  python scripts/grade_model_gw.py --gw 3 --dry-run
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DB_PATH = PROJECT_ROOT / "data" / "fpl.db"
EXPORT_DIR = PROJECT_ROOT / "predictions" / "fpl"

# formation minimums for a legal XI (position_id: min count)
XI_MINIMUMS = {1: 1, 2: 3, 3: 2, 4: 1}

RESULTS_SQL = """
CREATE TABLE IF NOT EXISTS model_gw_results (
    gameweek_id     INTEGER PRIMARY KEY,
    graded_at_utc   TEXT NOT NULL,
    gross_points    INTEGER NOT NULL,   -- XI after auto-subs, captain doubled
    hit_points      INTEGER NOT NULL,
    net_points      INTEGER NOT NULL,
    effective_captain INTEGER,          -- player_id whose score doubled (NULL if none)
    autosubs_json   TEXT NOT NULL,      -- [{out, in}]
    detail_json     TEXT NOT NULL       -- per-player {player_id, points, minutes, role}
)
"""


def grade(gw: int | None = None, dry_run: bool = False,
          db_path: Path = DB_PATH) -> dict:
    conn = sqlite3.connect(db_path)
    conn.execute(RESULTS_SQL)

    # ── pick the gameweek ────────────────────────────────────────────────────
    if gw is None:
        row = conn.execute("""
            SELECT l.gameweek_id FROM model_squad_log l
            JOIN gameweeks g ON g.gameweek_id = l.gameweek_id
            WHERE g.finished = 1
              AND l.gameweek_id NOT IN (SELECT gameweek_id FROM model_gw_results)
            ORDER BY l.gameweek_id DESC LIMIT 1
        """).fetchone()
        if not row:
            raise RuntimeError("No finished, locked, ungraded gameweek found.")
        gw = int(row[0])

    finished = conn.execute(
        "SELECT finished FROM gameweeks WHERE gameweek_id = ?", (gw,)
    ).fetchone()
    if not finished or not finished[0]:
        raise RuntimeError(f"GW{gw} is not marked finished — refusing to grade early.")

    already = conn.execute(
        "SELECT graded_at_utc FROM model_gw_results WHERE gameweek_id = ?", (gw,)
    ).fetchone()
    if already:
        raise RuntimeError(f"GW{gw} already graded at {already[0]} (append-only).")

    lock_row = conn.execute(
        "SELECT squad_json, transfers_json FROM model_squad_log WHERE gameweek_id = ?",
        (gw,),
    ).fetchone()
    if not lock_row:
        raise RuntimeError(f"GW{gw} was never locked — nothing to grade.")
    squad = json.loads(lock_row[0])
    hits = int(json.loads(lock_row[1]).get("hits", 0))

    # ── pull results + positions ─────────────────────────────────────────────
    ids = [p["player_id"] for p in squad]
    ph = ",".join("?" * len(ids))
    stats = {int(pid): (int(mins or 0), int(pts or 0)) for pid, mins, pts in
             conn.execute(f"SELECT player_id, minutes, total_points "
                          f"FROM player_gameweek_history "
                          f"WHERE gameweek_id = ? AND player_id IN ({ph})",
                          [gw] + ids)}
    positions = {int(pid): int(pos) for pid, pos in
                 conn.execute(f"SELECT player_id, position FROM players "
                              f"WHERE player_id IN ({ph})", ids)}

    def minutes(pid): return stats.get(pid, (0, 0))[0]
    def points(pid): return stats.get(pid, (0, 0))[1]

    xi = [p["player_id"] for p in squad if p["is_xi"]]
    bench = sorted((p for p in squad if not p["is_xi"]),
                   key=lambda p: p.get("bench_order", 99))
    captain = next((p["player_id"] for p in squad if p["is_captain"]), None)
    vice = next((p["player_id"] for p in squad if p["is_vice"]), None)

    # ── auto-subs ────────────────────────────────────────────────────────────
    autosubs: list[dict] = []
    used_bench: set[int] = set()
    final_xi = list(xi)

    def formation_ok(players: list[int]) -> bool:
        counts = {1: 0, 2: 0, 3: 0, 4: 0}
        for pid in players:
            counts[positions[pid]] += 1
        return all(counts[pos] >= mn for pos, mn in XI_MINIMUMS.items())

    for starter in xi:
        if minutes(starter) > 0:
            continue
        starter_pos = positions[starter]
        if starter_pos == 1:
            candidates = [b for b in bench
                          if positions[b["player_id"]] == 1
                          and b["player_id"] not in used_bench]
        else:
            candidates = [b for b in bench
                          if positions[b["player_id"]] != 1
                          and b["player_id"] not in used_bench]
        for cand in candidates:
            cid = cand["player_id"]
            if minutes(cid) == 0:
                continue
            trial = [p for p in final_xi if p != starter] + [cid]
            if formation_ok(trial):
                final_xi = trial
                used_bench.add(cid)
                autosubs.append({"out": starter, "in": cid})
                break

    # ── captaincy ────────────────────────────────────────────────────────────
    effective_captain = None
    if captain is not None and minutes(captain) > 0:
        effective_captain = captain
    elif vice is not None and minutes(vice) > 0:
        effective_captain = vice

    gross = sum(points(pid) for pid in final_xi)
    if effective_captain is not None and effective_captain in final_xi:
        gross += points(effective_captain)
    net = gross - hits

    detail = [{
        "player_id": pid,
        "points": points(pid),
        "minutes": minutes(pid),
        "role": ("captain" if pid == effective_captain else
                 "xi" if pid in final_xi else "bench"),
    } for pid in ids]

    result = {
        "gameweek": gw,
        "graded_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "gross_points": int(gross),
        "hit_points": hits,
        "net_points": int(net),
        "effective_captain": effective_captain,
        "autosubs": autosubs,
        "detail": detail,
    }

    if dry_run:
        print(json.dumps(result, indent=2))
        conn.close()
        return result

    conn.execute(
        "INSERT INTO model_gw_results VALUES (?,?,?,?,?,?,?,?)",
        (gw, result["graded_at_utc"], result["gross_points"], hits,
         result["net_points"], effective_captain,
         json.dumps(autosubs), json.dumps(detail)),
    )
    conn.commit()
    conn.close()

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = EXPORT_DIR / f"gw{gw:02d}_result.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"GW{gw} graded: {result['net_points']} net "
          f"({result['gross_points']} gross - {hits} hits), "
          f"{len(autosubs)} auto-sub(s).\nExported {out} — commit it; "
          f"the graded receipt is part of the public record.")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gw", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    grade(gw=args.gw, dry_run=args.dry_run)
