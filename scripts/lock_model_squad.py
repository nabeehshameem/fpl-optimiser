"""
Lock the model's FPL squad for the upcoming gameweek BEFORE the deadline and
record it in an append-only ledger. This is the credibility backbone of the
"Beat the Model" product: every squad is timestamped pre-deadline, and a JSON
export is written for git commit (the public, third-party-verifiable record —
same mechanism as the WC2026 retrospective).

The model plays by real FPL rules — it is a legitimate manager, not an oracle:
  * GW1: fresh GBP 100.0m squad via SquadOptimiser.optimise()
  * Later GWs: optimise_with_transfers() using the model's OWN tracked state:
      - previous squad (from the ledger)
      - bank (tracked to the tenth)
      - free transfers (banked up to 5, FPL rules)
      - selling prices (purchase price + floor(rise/2), from the ledger)
  * Hits are recorded and charged at grading time.

Append-only: a gameweek that is already locked can NEVER be overwritten.

Usage:
  python scripts/lock_model_squad.py            # lock next GW
  python scripts/lock_model_squad.py --dry-run  # print without writing
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.optimiser import SquadOptimiser  # noqa: E402

DB_PATH = PROJECT_ROOT / "data" / "fpl.db"
EXPORT_DIR = PROJECT_ROOT / "predictions" / "fpl"

MAX_BANKED_FT = 5
MODEL_NAME = "lightgbm_v1"   # predictions row filter; falls back to any model


# ── ledger schema ────────────────────────────────────────────────────────────

LEDGER_SQL = """
CREATE TABLE IF NOT EXISTS model_squad_log (
    gameweek_id     INTEGER PRIMARY KEY,
    locked_at_utc   TEXT NOT NULL,
    deadline_utc    TEXT NOT NULL,
    squad_json      TEXT NOT NULL,   -- 15 x {player_id, purchase_price, is_xi, is_captain, is_vice, bench_order}
    transfers_json  TEXT NOT NULL,   -- {in: [...], out: [...], hits: int}
    free_transfers  INTEGER NOT NULL,
    bank            INTEGER NOT NULL, -- tenths of GBP 1m, AFTER transfers
    expected_points REAL,
    squad_hash      TEXT NOT NULL
)
"""


def ensure_ledger(conn: sqlite3.Connection) -> None:
    conn.execute(LEDGER_SQL)
    conn.commit()


# ── helpers ──────────────────────────────────────────────────────────────────

def next_gameweek(conn) -> tuple[int, str]:
    row = conn.execute(
        "SELECT gameweek_id, deadline_time FROM gameweeks WHERE is_next = 1 LIMIT 1"
    ).fetchone()
    if not row:
        row = conn.execute(
            "SELECT gameweek_id, deadline_time FROM gameweeks "
            "WHERE finished = 0 ORDER BY gameweek_id LIMIT 1"
        ).fetchone()
    if not row:
        raise RuntimeError("No upcoming gameweek in DB — run ingest_bootstrap first.")
    return int(row[0]), str(row[1])


def load_predictions(conn, gw: int):
    """Latest prediction per player for `gw`, joined with eligibility features."""
    import pandas as pd
    df = pd.read_sql_query(
        """
        SELECT p.player_id, p.predicted_points
        FROM predictions p
        JOIN (
            SELECT player_id, MAX(prediction_time) AS mt
            FROM predictions WHERE gameweek_id = ?
            GROUP BY player_id
        ) latest ON latest.player_id = p.player_id AND latest.mt = p.prediction_time
        WHERE p.gameweek_id = ?
        """,
        conn, params=(gw, gw),
    )
    if df.empty:
        raise RuntimeError(
            f"No predictions for GW{gw}. Run scripts/run_predictions.py first."
        )
    hist = pd.read_sql_query(
        """
        SELECT player_id,
               SUM(CASE WHEN minutes >= 60 THEN 1 ELSE 0 END) AS q5,
               SUM(CASE WHEN minutes >= 60 AND recent3 = 1 THEN 1 ELSE 0 END) AS q3
        FROM (
            SELECT h.player_id, h.minutes,
                   CASE WHEN h.gameweek_id > ? - 3 THEN 1 ELSE 0 END AS recent3
            FROM player_gameweek_history h
            WHERE h.gameweek_id > ? - 5
        )
        GROUP BY player_id
        """,
        conn, params=(gw, gw),
    )
    df = df.merge(hist, on="player_id", how="left")
    df["qualifying_games_3"] = df["q3"].fillna(0)
    df["qualifying_games_5"] = df["q5"].fillna(0)
    df["chance_of_playing_next"] = 100

    # GW1-3 edge case: no PL history yet -> everyone fails qualifying gates.
    # Open the gates rather than pick from an empty pool.
    max_hist_gw = conn.execute(
        "SELECT COALESCE(MAX(gameweek_id), 0) FROM player_gameweek_history"
    ).fetchone()[0]
    if max_hist_gw < 3:
        df["qualifying_games_3"] = 3
        df["qualifying_games_5"] = 5
    return df


def current_prices(conn) -> dict[int, int]:
    return dict(conn.execute("SELECT player_id, current_cost FROM players"))


def selling_price(purchase: int, current: int) -> int:
    """FPL rule: sell at purchase + floor(rise / 2); fallen players sell at current."""
    if current <= purchase:
        return current
    return purchase + (current - purchase) // 2


def previous_state(conn) -> dict | None:
    row = conn.execute(
        "SELECT gameweek_id, squad_json, free_transfers, bank "
        "FROM model_squad_log ORDER BY gameweek_id DESC LIMIT 1"
    ).fetchone()
    if not row:
        return None
    return {
        "gw": int(row[0]),
        "squad": json.loads(row[1]),
        "free_transfers": int(row[2]),
        "bank": int(row[3]),
    }


# ── main ─────────────────────────────────────────────────────────────────────

def lock(dry_run: bool = False) -> dict:
    conn = sqlite3.connect(DB_PATH)
    ensure_ledger(conn)

    gw, deadline = next_gameweek(conn)
    now = datetime.now(timezone.utc)
    deadline_dt = datetime.fromisoformat(deadline.replace("Z", "+00:00"))
    if now >= deadline_dt:
        raise RuntimeError(
            f"GW{gw} deadline ({deadline}) has passed — refusing to lock. "
            "A post-deadline lock would be worthless as evidence."
        )

    already = conn.execute(
        "SELECT locked_at_utc FROM model_squad_log WHERE gameweek_id = ?", (gw,)
    ).fetchone()
    if already:
        raise RuntimeError(
            f"GW{gw} already locked at {already[0]}. The ledger is append-only; "
            "locked gameweeks are never overwritten."
        )

    preds = load_predictions(conn, gw)
    prices = current_prices(conn)
    prev = previous_state(conn)
    opt = SquadOptimiser(db_path=DB_PATH)

    if prev is None:
        # GW1 (or first ever lock): fresh GBP 100.0m squad
        result = opt.optimise(preds)
        squad_ids = list(result["squad"]["player_id"])
        transfers = {"in": [], "out": [], "hits": 0}
        free_transfers_after = 1
        bank_after = 1000 - int(result["total_cost"])
        purchase = {pid: prices[pid] for pid in squad_ids}
    else:
        prev_ids = [p["player_id"] for p in prev["squad"]]
        prev_purchase = {p["player_id"]: p["purchase_price"] for p in prev["squad"]}
        selling = {pid: selling_price(prev_purchase[pid], prices.get(pid, prev_purchase[pid]))
                   for pid in prev_ids}
        ft_available = min(MAX_BANKED_FT, prev["free_transfers"])

        result = opt.optimise_with_transfers(
            preds,
            current_squad=prev_ids,
            free_transfers=ft_available,
            max_transfers=MAX_BANKED_FT,
            bank=prev["bank"],
            selling_prices=selling,
        )
        squad_ids = list(result["squad"]["player_id"])
        in_ids = list(result["transfers_in"]["player_id"])
        out_ids = list(result["transfers_out"]["player_id"])
        n_tr = len(in_ids)
        transfers = {"in": in_ids, "out": out_ids, "hits": int(result["hit_points"])}
        # FPL FT rule: next week's FT = min(5, unused + 1); hits never go below 0 remaining
        free_transfers_after = min(MAX_BANKED_FT, max(ft_available - n_tr, 0) + 1)
        # bank: sold value in, purchases out
        funds = prev["bank"] + sum(selling[p] for p in out_ids)
        spend = sum(prices[p] for p in in_ids)
        bank_after = funds - spend
        if bank_after < 0:
            raise RuntimeError(
                f"Negative bank after transfers ({bank_after}) — selling-price "
                "wiring is broken; refusing to lock an illegal squad."
            )
        purchase = {}
        for pid in squad_ids:
            purchase[pid] = prices[pid] if pid in in_ids else prev_purchase[pid]

    xi_ids = set(result["starting_xi"]["player_id"])
    cap = int(result["captain"]["player_id"])
    vice = int(result["vice_captain"]["player_id"])

    # Bench order drives auto-subs at grading time. FPL convention:
    # slot 1 = backup GK (fixed), slots 2-4 = outfield bench, here ordered by
    # lock-time predicted points descending (best first). XI players get 0.
    bench = result["bench"]
    bench_gk = [int(r.player_id) for r in bench.itertuples() if r.position == 1]
    bench_out = [int(r.player_id) for r in
                 bench.sort_values("predicted_points", ascending=False).itertuples()
                 if r.position != 1]
    bench_order = {}
    if bench_gk:
        bench_order[bench_gk[0]] = 1
    for i, pid in enumerate(bench_out, start=2):
        bench_order[pid] = i

    squad_rows = [{
        "player_id": int(pid),
        "purchase_price": int(purchase[pid]),
        "is_xi": int(pid in xi_ids),
        "is_captain": int(pid == cap),
        "is_vice": int(pid == vice),
        "bench_order": bench_order.get(int(pid), 0),
    } for pid in squad_ids]

    payload = {
        "gameweek": gw,
        "locked_at_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "deadline_utc": deadline,
        "squad": squad_rows,
        "transfers": transfers,
        "free_transfers_after": free_transfers_after,
        "bank_after": bank_after,
        "expected_points": round(float(result["expected_points"]), 2),
    }
    payload["squad_hash"] = hashlib.sha256(
        json.dumps(squad_rows, sort_keys=True).encode()
    ).hexdigest()[:16]

    if dry_run:
        print(json.dumps(payload, indent=2))
        conn.close()
        return payload

    conn.execute(
        "INSERT INTO model_squad_log VALUES (?,?,?,?,?,?,?,?,?)",
        (gw, payload["locked_at_utc"], deadline,
         json.dumps(squad_rows), json.dumps(transfers),
         free_transfers_after, bank_after,
         payload["expected_points"], payload["squad_hash"]),
    )
    conn.commit()
    conn.close()

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = EXPORT_DIR / f"gw{gw:02d}.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"GW{gw} locked at {payload['locked_at_utc']} "
          f"(deadline {deadline}).\nExported {out} — COMMIT AND PUSH THIS "
          f"BEFORE THE DEADLINE; the push is the public timestamp.")
    return payload


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    lock(dry_run=args.dry_run)
