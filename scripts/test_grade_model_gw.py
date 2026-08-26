"""
Synthetic tests for scripts/grade_model_gw.py. No real fpl.db needed.
Run: python scripts/test_grade_model_gw.py

G1  All XI play: gross = sum + captain again; hits subtracted
G2  Captain blanks (0 min): vice doubles instead
G3  Captain and vice both blank: nobody doubles
G4  Blanked DEF in a 3-DEF formation: bench MID (slot 2) is ILLEGAL to bring
    on, bench DEF (slot 3) comes in instead — formation rule enforced over
    bench order
G5  Blanked starter, all eligible bench also blanked: no sub, player scores 0
G6  GK blank: only bench GK can replace
G7  Ungraded guards: unfinished GW refused; double-grading refused
"""

import json
import sqlite3
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import scripts.grade_model_gw as g  # noqa: E402
from src.squad_commit import CANONICAL, compute_squad_hash  # noqa: E402

# Squad layout (player_id: position): 3-5-2 XI + bench
# XI: GK 1 | DEF 2,3,4 | MID 5,6,7,8,9 | FWD 10,11
# Bench: GK 12 (slot1), MID 13 (slot2), DEF 14 (slot3), FWD 15 (slot4)
POS = {1: 1, 2: 2, 3: 2, 4: 2, 5: 3, 6: 3, 7: 3, 8: 3, 9: 3,
       10: 4, 11: 4, 12: 1, 13: 3, 14: 2, 15: 4}
XI = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
BENCH_ORDER = {12: 1, 13: 2, 14: 3, 15: 4}


def build_db(tmp: Path, stats: dict, hits: int = 0,
             captain: int = 10, vice: int = 5, finished: int = 1,
             excluded_json: str | None = None) -> Path:
    """stats: {player_id: (minutes, points)}"""
    db = tmp / "fpl.db"
    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE TABLE teams (team_id INT PRIMARY KEY, short_name TEXT);
        CREATE TABLE players (player_id INT PRIMARY KEY, position INT,
            web_name TEXT, team_id INT);
        CREATE TABLE gameweeks (gameweek_id INT PRIMARY KEY, deadline_time TEXT,
            is_current INT, is_next INT, finished INT, average_score INT);
        CREATE TABLE player_gameweek_history (player_id INT, gameweek_id INT,
            minutes INT, total_points INT, PRIMARY KEY (player_id, gameweek_id));
        CREATE TABLE model_squad_log (gameweek_id INT PRIMARY KEY,
            locked_at_utc TEXT, deadline_utc TEXT, squad_json TEXT,
            transfers_json TEXT, free_transfers INT, bank INT,
            expected_points REAL, squad_hash TEXT, excluded_json TEXT);
    """)
    conn.execute("INSERT INTO teams VALUES (1, 'TST')")
    conn.executemany("INSERT INTO players VALUES (?,?,?,1)",
                     [(pid, pos, f"P{pid}") for pid, pos in POS.items()])
    conn.execute("INSERT INTO gameweeks VALUES (1,'x',0,0,?,NULL)", (finished,))
    conn.executemany(
        "INSERT INTO player_gameweek_history VALUES (?,1,?,?)",
        [(pid, m, p) for pid, (m, p) in stats.items()])
    squad = [{"player_id": pid, "purchase_price": 50,
              "is_xi": int(pid in XI),
              "is_captain": int(pid == captain),
              "is_vice": int(pid == vice),
              "bench_order": BENCH_ORDER.get(pid, 0)} for pid in POS]
    squad_hash = compute_squad_hash(squad)
    conn.execute("INSERT INTO model_squad_log VALUES (1,'t','t',?,?,1,0,62.0,?,?)",
                 (json.dumps(squad, **CANONICAL),
                  json.dumps({"in": [], "out": [], "hits": hits}),
                  squad_hash, excluded_json))
    conn.commit()
    conn.close()
    return db


def check(label, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {label}" + (f"  ({detail})" if detail else ""))
    return cond


def base_stats(overrides=None):
    """Everyone plays 90 and scores: XI 4 pts each, bench 2 pts each."""
    s = {pid: (90, 4 if pid in XI else 2) for pid in POS}
    s.update(overrides or {})
    return s


def main():
    import tempfile as _tf
    g.EXPORT_DIR = Path(_tf.mkdtemp())
    ok = True

    # G1: clean week, captain 10 (4 pts) doubles, 4-pt hit
    db = build_db(Path(tempfile.mkdtemp()), base_stats(), hits=4)
    r = g.grade(gw=1, dry_run=True, db_path=db)
    ok &= check("G1 gross = 11*4 + captain 4", r["gross_points"] == 48,
                f"gross={r['gross_points']}")
    ok &= check("G1 net = gross - 4", r["net_points"] == 44)

    # G2: captain blanks -> vice (pid 5) doubles
    db = build_db(Path(tempfile.mkdtemp()), base_stats({10: (0, 0)}))
    r = g.grade(gw=1, dry_run=True, db_path=db)
    ok &= check("G2 vice doubles when captain blanks",
                r["effective_captain"] == 5, f"eff={r['effective_captain']}")
    # captain's 0 replaced by bench FWD 15 (2 pts): 10*4 + 2 + vice 4 = 46
    ok &= check("G2 gross correct with sub + vice double",
                r["gross_points"] == 46, f"gross={r['gross_points']}")

    # G3: captain and vice both blank -> nobody doubles
    db = build_db(Path(tempfile.mkdtemp()),
                  base_stats({10: (0, 0), 5: (0, 0)}))
    r = g.grade(gw=1, dry_run=True, db_path=db)
    ok &= check("G3 no effective captain", r["effective_captain"] is None)

    # G4: DEF 2 blanks in a 3-DEF XI. Bench slot 2 is MID 13 — bringing a MID
    # for a DEF leaves 2 DEF = illegal. Slot 3 DEF 14 must come in.
    db = build_db(Path(tempfile.mkdtemp()), base_stats({2: (0, 0)}))
    r = g.grade(gw=1, dry_run=True, db_path=db)
    subs = {(s["out"], s["in"]) for s in r["autosubs"]}
    ok &= check("G4 formation forces DEF sub over bench order",
                (2, 14) in subs and not any(i == 13 for _, i in subs),
                f"subs={subs}")

    # G5: MID 5 blanks; eligible bench (13 MID, 14 DEF, 15 FWD) all blank too
    db = build_db(Path(tempfile.mkdtemp()),
                  base_stats({5: (0, 0), 13: (0, 0), 14: (0, 0), 15: (0, 0)}))
    r = g.grade(gw=1, dry_run=True, db_path=db)
    ok &= check("G5 no sub when bench blanked", r["autosubs"] == [],
                f"subs={r['autosubs']}")
    ok &= check("G5 gross drops by blanked starter",
                r["gross_points"] == 44, f"gross={r['gross_points']}")

    # G6: GK 1 blanks -> only bench GK 12 comes in (never outfield)
    db = build_db(Path(tempfile.mkdtemp()), base_stats({1: (0, 0)}))
    r = g.grade(gw=1, dry_run=True, db_path=db)
    subs = {(s["out"], s["in"]) for s in r["autosubs"]}
    ok &= check("G6 GK replaced by bench GK only", subs == {(1, 12)},
                f"subs={subs}")

    # G7: guards
    db = build_db(Path(tempfile.mkdtemp()), base_stats(), finished=0)
    try:
        g.grade(gw=1, dry_run=True, db_path=db)
        ok &= check("G7 unfinished GW refused", False)
    except RuntimeError as e:
        ok &= check("G7 unfinished GW refused", "finished" in str(e))
    db = build_db(Path(tempfile.mkdtemp()), base_stats())
    g.grade(gw=1, dry_run=False, db_path=db)
    try:
        g.grade(gw=1, dry_run=False, db_path=db)
        ok &= check("G7 double-grading refused", False)
    except RuntimeError as e:
        ok &= check("G7 double-grading refused", "already graded" in str(e))

    # G8: tamper check — mutate bench_order in stored squad_json AFTER the
    # commitment is written; grader must refuse, not produce a silently wrong score.
    db = build_db(Path(tempfile.mkdtemp()), base_stats())
    conn = sqlite3.connect(db)
    row = conn.execute(
        "SELECT squad_json FROM model_squad_log WHERE gameweek_id = 1"
    ).fetchone()
    tampered = json.loads(row[0])
    tampered[0]["bench_order"] = 99   # mutate one field post-commitment
    conn.execute(
        "UPDATE model_squad_log SET squad_json = ? WHERE gameweek_id = 1",
        (json.dumps(tampered, **CANONICAL),),
    )
    conn.commit()
    conn.close()
    try:
        g.grade(gw=1, dry_run=True, db_path=db)
        ok &= check("G8 tampered squad_json rejected", False)
    except RuntimeError as e:
        ok &= check("G8 tampered squad_json rejected",
                    "mismatch" in str(e).lower(), str(e))

    # G9: external verification — the published result must contain enough to
    # verify the commitment using ONLY stdlib (what a skeptic on the internet has).
    db = build_db(Path(tempfile.mkdtemp()), base_stats())
    r = g.grade(gw=1, dry_run=True, db_path=db)
    import hashlib as _h
    independent = _h.sha256(json.dumps(
        r["squad"], sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()).hexdigest()
    ok &= check("G9 published reveal verifies against commitment",
                independent == r["squad_hash"],
                f"independent={independent[:16]}... stored={r['squad_hash'][:16]}...")

    # G10: the excluded pool is part of the published record. Both sources must
    # round-trip labelled, and a squad locked before the field existed must
    # reveal null -- NOT [], which would falsely assert nothing was filtered.
    payload = json.dumps({"manual": [12, 17], "unavailable": [13]})
    db = build_db(Path(tempfile.mkdtemp()), base_stats(), excluded_json=payload)
    r = g.grade(gw=1, dry_run=True, db_path=db)
    efp = r.get("excluded_from_pool")
    ok &= check("G10 excluded pool revealed with both sources labelled",
                efp == {"manual": [12, 17], "unavailable": [13]}, str(efp))

    db = build_db(Path(tempfile.mkdtemp()), base_stats(), excluded_json=None)
    r = g.grade(gw=1, dry_run=True, db_path=db)
    ok &= check("G10 pre-change squad reveals null, not empty list",
                r.get("excluded_from_pool", "MISSING") is None,
                repr(r.get("excluded_from_pool", "MISSING")))

    # G11: dry_run=False also refuses an unfinished GW (the guard fires before
    # any write, so no filesystem side-effect even though EXPORT_DIR is live).
    db = build_db(Path(tempfile.mkdtemp()), base_stats(), finished=0)
    try:
        g.grade(gw=1, dry_run=False, db_path=db)
        ok &= check("G11 dry_run=False refuses unfinished GW", False)
    except RuntimeError as e:
        ok &= check("G11 dry_run=False refuses unfinished GW",
                    "finished" in str(e), str(e)[:60])

    # G12: correct_model_gw records corrections and enforces limits.
    import scripts.correct_model_gw as corr
    tmp_export = Path(tempfile.mkdtemp())
    corr.EXPORT_DIR = tmp_export
    db = build_db(Path(tempfile.mkdtemp()), base_stats())
    r = g.grade(gw=1, dry_run=False, db_path=db)
    # Write the graded result to the tmp export dir for correction tests
    result_path = tmp_export / "gw01_result.json"
    result_path.write_text(json.dumps(r, indent=2))

    # G12 graded net_points = 48 (no hits). Correction: 48 -> 47.
    corr.correct(gw=1, field="net_points", from_val=48, to_val=47,
                 reason="test correction", commit="abc1234")
    amended = json.loads(result_path.read_text())
    ok &= check("G12 correction updates field value",
                amended["net_points"] == 47, f"net={amended['net_points']}")
    ok &= check("G12 correction recorded in corrections array",
                len(amended.get("corrections", [])) == 1,
                f"corrections={amended.get('corrections')}")
    ok &= check("G12 correction entry has required keys",
                all(k in amended["corrections"][0]
                    for k in ("revised_at_utc", "field", "from", "to", "reason", "commit")))

    # No-op refused (already at 47, not 48)
    try:
        corr.correct(gw=1, field="net_points", from_val=48, to_val=47,
                     reason="no-op", commit="abc")
        ok &= check("G12 no-op correction refused", False)
    except SystemExit:
        ok &= check("G12 no-op correction refused (value already corrected)", True)

    # Max corrections enforced
    for i in range(corr.MAX_CORRECTIONS - 1):
        amended["corrections"].append({"dummy": i})
    result_path.write_text(json.dumps(amended, indent=2))
    try:
        corr.correct(gw=1, field="net_points", from_val=47, to_val=46,
                     reason="overflow", commit="abc")
        ok &= check("G12 corrections limit enforced", False)
    except SystemExit:
        ok &= check("G12 corrections limit enforced", True)

    print("\n" + ("ALL PASS" if ok else "FAILURES PRESENT"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
