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
import os
import sqlite3
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.squad_commit import compute_squad_hash  # noqa: E402

# Overridable so a DRY RUN can point at copies. Rehearsing against the real
# database and export directory would write a row into the append-only ledger
# and publish a commitment hash — the real lock would then be refused, and the
# public record would carry a squad built at rehearsal time.
#   FPL_DB_PATH=data/fpl_dryrun.db FPL_EXPORT_DIR=/tmp/dryrun GIT_BRANCH=dryrun \
#       python scripts/weekly_ops.py lock
DB_PATH = Path(os.getenv("FPL_DB_PATH", PROJECT_ROOT / "data" / "fpl.db"))
EXPORT_DIR = Path(os.getenv("FPL_EXPORT_DIR", PROJECT_ROOT / "predictions" / "fpl"))
# The FPL entry being graded. Used to fetch the actual bench order at grade time
# so auto-subs match FPL's result exactly, regardless of the model's intended order.
FPL_ENTRY_ID = int(os.getenv("FPL_ENTRY_ID", "690670"))

POS_NAMES = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

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


def _fetch_fpl_bench_order(entry_id: int, gw: int) -> dict[int, int]:
    """Return {player_id: bench_slot} from FPL's actual picks for the entry.

    Bench slot is the pick position minus 11 (so 1-4 for bench players).
    Returns {} on any error so callers fall back to the commitment bench_order.
    """
    try:
        url = f"https://fantasy.premierleague.com/api/entry/{entry_id}/event/{gw}/picks/"
        req = urllib.request.Request(url, headers={"User-Agent": "fpl-grader/1.0"})
        data = json.loads(urllib.request.urlopen(req, timeout=15).read())
        return {p["element"]: p["position"] - 11
                for p in data.get("picks", [])
                if p["position"] > 11}
    except Exception as exc:
        print(f"[WARN] could not fetch FPL bench order (falling back to commitment): {exc}",
              file=sys.stderr)
        return {}


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
    if already and not dry_run:
        out = EXPORT_DIR / f"gw{gw:02d}_result.json"
        if out.exists() and out.stat().st_size > 0:
            raise RuntimeError(f"GW{gw} already graded at {already[0]} (append-only).")
        # JSON is missing/empty but DB row exists — fall through to re-export only
        print(f"[INFO] GW{gw} DB row exists but JSON is missing/empty — re-exporting.")
    elif already and dry_run:
        raise RuntimeError(f"GW{gw} already graded at {already[0]} (append-only).")

    lock_row = conn.execute(
        "SELECT squad_json, transfers_json, squad_hash, expected_points, excluded_json "
        "FROM model_squad_log WHERE gameweek_id = ?",
        (gw,),
    ).fetchone()
    if not lock_row:
        raise RuntimeError(f"GW{gw} was never locked — nothing to grade.")
    squad_json, transfers_json, stored_hash, lock_expected_pts, excluded_json = lock_row
    excluded_from_pool = json.loads(excluded_json) if excluded_json else None

    # Verify the pre-deadline commitment before revealing the graded result.
    # src.squad_commit is the single source of truth for canonicalisation —
    # a mismatch means the ledger record was altered after the pre-deadline push.
    recomputed = compute_squad_hash(json.loads(squad_json))
    if recomputed != stored_hash:
        raise RuntimeError(
            f"GW{gw} squad hash mismatch: stored={stored_hash!r} "
            f"recomputed={recomputed!r}. Ledger may have been tampered with — "
            "refusing to grade."
        )

    squad = json.loads(squad_json)
    transfers_data = json.loads(transfers_json)
    hits = int(transfers_data.get("hits", 0))

    # ── pull results + positions + display names ─────────────────────────────
    ids = [p["player_id"] for p in squad]
    out_ids = transfers_data.get("out", [])
    all_ids = list({*ids, *out_ids})
    ph = ",".join("?" * len(ids))
    ph_all = ",".join("?" * len(all_ids)) if all_ids else "?"

    stats = {int(pid): (int(mins or 0), int(pts or 0)) for pid, mins, pts in
             conn.execute(f"SELECT player_id, minutes, total_points "
                          f"FROM player_gameweek_history "
                          f"WHERE gameweek_id = ? AND player_id IN ({ph})",
                          [gw] + ids)}
    player_info: dict[int, dict] = {}
    for pid, pos, name, team in conn.execute(
        f"SELECT p.player_id, p.position, p.web_name, t.short_name "
        f"FROM players p LEFT JOIN teams t ON t.team_id = p.team_id "
        f"WHERE p.player_id IN ({ph_all})",
        all_ids or [0]
    ).fetchall():
        player_info[int(pid)] = {"position": int(pos or 0), "name": name, "team": team}
    positions = {pid: info["position"] for pid, info in player_info.items()}

    avg_row = conn.execute(
        "SELECT average_score FROM gameweeks WHERE gameweek_id = ?", (gw,)
    ).fetchone()
    average_score = int(avg_row[0]) if avg_row and avg_row[0] is not None else None

    def minutes(pid): return stats.get(pid, (0, 0))[0]
    def points(pid): return stats.get(pid, (0, 0))[1]

    xi = [p["player_id"] for p in squad if p["is_xi"]]
    # Fetch actual bench order from FPL so auto-subs match FPL's result exactly.
    # The commitment file records the model's intended order (by predicted pts),
    # which may differ from the order the user set in FPL. Fall back to the
    # commitment order if the API is unavailable.
    actual_bench_order = _fetch_fpl_bench_order(FPL_ENTRY_ID, gw)
    bench = sorted(
        (p for p in squad if not p["is_xi"]),
        key=lambda p: actual_bench_order.get(p["player_id"],
                                             p.get("bench_order", 99)),
    )
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

    # Load the lock-time projections so we can compare predicted vs actual
    # per player in the graded record. None if file was not committed.
    proj_file = EXPORT_DIR / f"gw{gw:02d}_projections.json"
    if proj_file.exists():
        proj_data = json.loads(proj_file.read_text(encoding="utf-8"))
        _proj_map: dict[int, float] = {}
        for entry in proj_data.get("by_position", {}).values():
            for p in entry:
                _proj_map[int(p["player_id"])] = float(p["projected_points"])
        for p in proj_data.get("captain_candidates", []):
            _proj_map.setdefault(int(p["player_id"]), float(p["projected_points"]))
    else:
        _proj_map = {}

    detail = [{
        "player_id": pid,
        "points": points(pid),
        "minutes": minutes(pid),
        "role": ("captain" if pid == effective_captain else
                 "xi" if pid in final_xi else "bench"),
        "projected_points": _proj_map.get(pid),
    } for pid in ids]

    def _pinfo(pid: int) -> dict:
        info = player_info.get(pid, {})
        return {
            "name": info.get("name") or f"#{pid}",
            "position": POS_NAMES.get(info.get("position"), "?"),
            "team": info.get("team"),
        }

    result = {
        "gameweek": gw,
        # THE REVEAL: the exact rows the pre-deadline hash committed to.
        # External verification: sha256 of this list serialised with
        # sort_keys=True, separators=(",",":"), ensure_ascii=True must equal
        # squad_hash in the pre-deadline gwNN.json. See src/squad_commit.py.
        "squad": squad,
        "squad_display": [{"player_id": pid, **_pinfo(pid)} for pid in ids],
        "squad_hash": stored_hash,
        # Preserve original graded_at_utc on re-export so the public record timestamp
        # matches the DB row — don't generate a new timestamp for a file rebuild.
        "graded_at_utc": (already[0] if already else
                          datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")),
        "gross_points": int(gross),
        "hit_points": hits,
        "net_points": int(net),
        "effective_captain": effective_captain,
        "effective_captain_display": (
            {"player_id": effective_captain, **_pinfo(effective_captain)}
            if effective_captain is not None else None
        ),
        "autosubs": autosubs,
        "autosubs_display": [
            {"out": {"player_id": s["out"], **_pinfo(s["out"])},
             "in": {"player_id": s["in"], **_pinfo(s["in"])}}
            for s in autosubs
        ],
        "transfers_display": {
            "in": [{"player_id": p, **_pinfo(p)} for p in transfers_data.get("in", [])],
            "out": [{"player_id": p, **_pinfo(p)} for p in transfers_data.get("out", [])],
        },
        "average_score": average_score,
        "expected_points": float(lock_expected_pts) if lock_expected_pts is not None else None,
        "detail": detail,
        # Players removed from the optimiser's pool at lock time.
        # manual: hand-maintained config/player_exclusions.txt (validated
        #   against the DB at lock time; the file's state is pinned by the
        #   same git commit that pushed this result).
        # unavailable: FPL's chance_of_playing_next < 50 at lock time.
        # None on squads locked before this field was introduced.
        "excluded_from_pool": excluded_from_pool,
        # Corrections appended when a published result is later amended.
        # Rule 11: published artifacts are append-only — corrected in public
        # with a reason and commit, never silently replaced.
        "corrections": [],
    }

    if dry_run:
        print(json.dumps(result, indent=2))
        conn.close()
        return result

    reexport_only = bool(already)
    if not reexport_only:
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
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if reexport_only:
        print(f"GW{gw} re-exported (DB row preserved): {out}")
    else:
        print(f"GW{gw} graded: {result['net_points']} net "
              f"({result['gross_points']} gross - {hits} hits), "
              f"{len(autosubs)} auto-sub(s).\nExported {out} — commit it; "
              f"the graded receipt is part of the public record.")
    print("\nRun full-league postmortem:")
    print(f"  python scripts/gw_postmortem.py --gw {gw}")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gw", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    grade(gw=args.gw, dry_run=args.dry_run)
