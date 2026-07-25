"""
End-to-end test of scripts/lock_model_squad.py over a synthetic two-GW season.
No real fpl.db required. Run: python scripts/test_lock_model_squad.py

Verifies:
  L1  GW1 lock: fresh squad, bank = 1000 - cost, FT rolls to 1, purchase
      prices recorded, JSON export written
  L2  Append-only: second lock of GW1 refused
  L3  GW2 lock after a price rise: selling prices honoured (purchase + rise/2),
      bank math exact, transfers recorded, FT banking rule applied
  L4  Deadline guard: locking a GW whose deadline has passed is refused
"""

import json
import sqlite3
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import scripts.lock_model_squad as lock_mod  # noqa: E402


def iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def build_world(tmpdir: Path) -> Path:
    """20 players (2 GK / 6 DEF / 7 MID / 5 FWD across 10 teams), 2 gameweeks."""
    db = tmpdir / "fpl.db"
    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE TABLE teams (team_id INT, short_name TEXT);
        CREATE TABLE players (player_id INT PRIMARY KEY, web_name TEXT,
            position INT, team_id INT, current_cost INT);
        CREATE TABLE gameweeks (gameweek_id INT PRIMARY KEY, deadline_time TEXT,
            is_current INT DEFAULT 0, is_next INT DEFAULT 0, finished INT DEFAULT 0,
            average_score INT);
        CREATE TABLE predictions (prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id INT, gameweek_id INT, model_name TEXT,
            predicted_points REAL, prediction_time TEXT);
        CREATE TABLE player_gameweek_history (player_id INT, gameweek_id INT,
            fixture_id INT, minutes INT, goals_scored INT, assists INT,
            clean_sheets INT, total_points INT, bonus INT, bps INT,
            expected_goals REAL, expected_assists REAL,
            defensive_contribution INT, value INT, selected INT,
            PRIMARY KEY (player_id, gameweek_id));
    """)
    conn.executemany("INSERT INTO teams VALUES (?,?)",
                     [(i, f"T{i}") for i in range(1, 11)])
    # (id, name, pos, team, cost) â€” enough depth in every position
    players = [
        (1, "GK1", 1, 1, 45), (2, "GK2", 1, 2, 40),
        (3, "D1", 2, 1, 55), (4, "D2", 2, 2, 55), (5, "D3", 2, 3, 50),
        (6, "D4", 2, 4, 45), (7, "D5", 2, 5, 40), (8, "D6", 2, 6, 40),
        (9, "M1", 3, 5, 105), (10, "M2", 3, 6, 90), (11, "M3", 3, 7, 75),
        (12, "M4", 3, 7, 60), (13, "M5", 3, 8, 50), (14, "M6", 3, 9, 45),
        (15, "F1", 4, 8, 125), (16, "F2", 4, 3, 80), (17, "F3", 4, 9, 60),
        (18, "F4", 4, 10, 45), (19, "M7", 3, 10, 45), (20, "D7", 2, 10, 40),
    ]
    conn.executemany("INSERT INTO players VALUES (?,?,?,?,?)", players)
    now = datetime.now(timezone.utc)
    conn.execute("INSERT INTO gameweeks VALUES (1, ?, 0, 1, 0, NULL)",
                 (iso(now + timedelta(hours=6)),))
    conn.execute("INSERT INTO gameweeks VALUES (2, ?, 0, 0, 0, NULL)",
                 (iso(now + timedelta(days=7)),))
    # GW1 predictions: points roughly tracking price
    pts = {1: 4, 2: 3, 3: 5, 4: 5, 5: 4.5, 6: 4, 7: 3, 8: 3, 9: 8, 10: 7,
           11: 6, 12: 5, 13: 4, 14: 3, 15: 9, 16: 6, 17: 5, 18: 3, 19: 3, 20: 2.5}
    conn.executemany(
        "INSERT INTO predictions (player_id, gameweek_id, model_name, "
        "predicted_points, prediction_time) VALUES (?,?,?,?,?)",
        [(pid, 1, "dc_projection_v1", p, iso(now)) for pid, p in pts.items()])
    conn.commit()
    conn.close()
    return db


def check(label, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {label}" + (f"  ({detail})" if detail else ""))
    return cond


def main():
    import src.availability as avail

    tmp = Path(tempfile.mkdtemp())
    db = build_world(tmp)
    lock_mod.DB_PATH = db
    lock_mod.EXPORT_DIR = tmp / "predictions"

    # Isolate from real config/player_exclusions.txt â€” the production file
    # contains IDs that collide with the synthetic world's player numbering.
    empty_excl = tmp / "exclusions_empty.txt"
    empty_excl.write_text("")
    avail.EXCLUSIONS_FILE = empty_excl

    ok = True

    # â”€â”€ L1: GW1 fresh lock â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    p1 = lock_mod.lock()
    conn = sqlite3.connect(db)
    squad_cost = sum(
        conn.execute("SELECT current_cost FROM players WHERE player_id=?",
                     (r["player_id"],)).fetchone()[0]
        for r in p1["squad"])
    ok &= check("L1 squad of 15 locked", len(p1["squad"]) == 15)
    ok &= check("L1 bank = 1000 - cost", p1["bank_after"] == 1000 - squad_cost,
                f"bank={p1['bank_after']} cost={squad_cost}")
    ok &= check("L1 FT rolls to 1", p1["free_transfers_after"] == 1)
    ok &= check("L1 purchase prices recorded",
                all(r["purchase_price"] > 0 for r in p1["squad"]))
    ok &= check("L1 export written", (lock_mod.EXPORT_DIR / "gw01.json").exists())
    _COMMITMENT_KEYS = {"gameweek", "locked_at_utc", "deadline_utc", "squad_hash"}
    exported = json.loads((lock_mod.EXPORT_DIR / "gw01.json").read_text())
    ok &= check("L1 commitment export has no squad details",
                set(exported.keys()) == _COMMITMENT_KEYS,
                f"keys={set(exported.keys())}")
    ok &= check("L1 exactly one captain",
                sum(r["is_captain"] for r in p1["squad"]) == 1)

    # â”€â”€ L2: append-only â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    try:
        lock_mod.lock()
        ok &= check("L2 re-lock refused", False)
    except RuntimeError as e:
        ok &= check("L2 re-lock refused", "append-only" in str(e))

    # â”€â”€ L3: GW2 with a price rise â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # advance world: GW1 done, GW2 next; every squad player rises 3 (0.3m)
    squad_ids = [r["player_id"] for r in p1["squad"]]
    conn.execute("UPDATE gameweeks SET is_next=0, finished=1 WHERE gameweek_id=1")
    conn.execute("UPDATE gameweeks SET is_next=1 WHERE gameweek_id=2")
    conn.executemany("UPDATE players SET current_cost = current_cost + 3 "
                     "WHERE player_id=?", [(i,) for i in squad_ids])
    # GW2 predictions: make one squad player collapse and one outsider surge,
    # so at least one transfer is clearly optimal
    now = datetime.now(timezone.utc)
    pts2 = {pid: 4.0 for pid in range(1, 21)}
    worst_xi = min((r for r in p1["squad"] if r["is_xi"]),
                   key=lambda r: r["purchase_price"])
    pts2[worst_xi["player_id"]] = 0.5
    outsider = next(i for i in range(1, 21) if i not in squad_ids)
    pts2[outsider] = 12.0
    conn.executemany(
        "INSERT INTO predictions (player_id, gameweek_id, model_name, "
        "predicted_points, prediction_time) VALUES (?,?,?,?,?)",
        [(pid, 2, "dc_projection_v1", p, iso(now)) for pid, p in pts2.items()])
    conn.commit(); conn.close()

    p2 = lock_mod.lock()
    tr = p2["transfers"]
    ok &= check("L3 transfers made", len(tr["in"]) >= 1, f"in={tr['in']} out={tr['out']}")
    # bank math: prev bank + sum(selling of outs) - sum(market of ins)
    conn = sqlite3.connect(db)
    prev_purchase = {r["player_id"]: r["purchase_price"] for r in p1["squad"]}
    sell = 0
    for pid in tr["out"]:
        cur = conn.execute("SELECT current_cost FROM players WHERE player_id=?",
                           (pid,)).fetchone()[0]
        sell += lock_mod.selling_price(prev_purchase[pid], cur)
    buy = sum(conn.execute("SELECT current_cost FROM players WHERE player_id=?",
                           (pid,)).fetchone()[0] for pid in tr["in"])
    conn.close()
    expected_bank = p1["bank_after"] + sell - buy
    ok &= check("L3 bank math exact (selling = purchase + rise//2)",
                p2["bank_after"] == expected_bank,
                f"bank={p2['bank_after']} expected={expected_bank}")
    n_tr = len(tr["in"])
    expected_ft = min(5, max(1 - n_tr, 0) + 1)
    ok &= check("L3 FT banking rule", p2["free_transfers_after"] == expected_ft,
                f"ft={p2['free_transfers_after']} expected={expected_ft}")
    ok &= check("L3 incoming purchase = market price",
                all(r["purchase_price"] == dict_cost(db, r["player_id"])
                    for r in p2["squad"] if r["player_id"] in tr["in"]))
    ok &= check("L3 retained purchase price preserved",
                all(r["purchase_price"] == prev_purchase[r["player_id"]]
                    for r in p2["squad"] if r["player_id"] in prev_purchase))

    # â”€â”€ L4: deadline guard â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    conn = sqlite3.connect(db)
    conn.execute("DELETE FROM model_squad_log WHERE gameweek_id=2")
    conn.execute("UPDATE gameweeks SET deadline_time=? WHERE gameweek_id=2",
                 (iso(datetime.now(timezone.utc) - timedelta(hours=1)),))
    conn.commit(); conn.close()
    try:
        lock_mod.lock()
        ok &= check("L4 post-deadline lock refused", False)
    except RuntimeError as e:
        ok &= check("L4 post-deadline lock refused", "deadline" in str(e).lower())

    # â”€â”€ L5: excluded player cannot appear in optimised squad â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    import src.availability as avail

    tmp5 = Path(tempfile.mkdtemp())
    db5 = build_world(tmp5)
    lock_mod.DB_PATH = db5
    lock_mod.EXPORT_DIR = tmp5 / "predictions"

    # Exclude F1 (player_id=15), the highest-scored forward â€” it would be in
    # any unconstrained squad given pts=9. After exclusion it must not appear.
    excl_file = tmp5 / "exclusions.txt"
    # Labelled form is mandatory: the loader validates the comment against the
    # database so a stale or reassigned id fails loudly instead of removing
    # the wrong player.
    excl_file.write_text("15  # F1 (T8) -- deliberately excluded for L5\n", encoding="utf-8")
    avail.EXCLUSIONS_FILE = excl_file

    p5 = lock_mod.lock()
    squad5 = {r["player_id"] for r in p5["squad"]}
    ok &= check("L5 excluded player absent from squad", 15 not in squad5,
                f"squad={sorted(squad5)}")
    ok &= check("L5 squad still has 15 players", len(p5["squad"]) == 15,
                f"len={len(p5['squad'])}")

    avail.EXCLUSIONS_FILE = PROJECT_ROOT / "config" / "player_exclusions.txt"

    print("\n" + ("ALL PASS" if ok else "FAILURES PRESENT"))
    sys.exit(0 if ok else 1)


def dict_cost(db, pid):
    conn = sqlite3.connect(db)
    c = conn.execute("SELECT current_cost FROM players WHERE player_id=?",
                     (pid,)).fetchone()[0]
    conn.close()
    return c


if __name__ == "__main__":
    main()

