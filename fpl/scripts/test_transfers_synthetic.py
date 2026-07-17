"""
fpl/scripts/test_transfers_synthetic.py

Six-scenario transfer-logic test suite for fpl_optimizer.optimise().
Uses a synthetic in-memory SQLite database — no dependency on data/fpl.db.
Runs anywhere, including CI.

Scenarios:
  S1  Refuses transfers when squad is already globally optimal
  S2  Takes a free upgrade (within free transfer allowance)
  S3  Refuses a +small upgrade when it requires a hit (net negative)
  S4  Takes a +large upgrade despite the hit (net positive)
  S5  Effective budget: selling_prices < current_cost → expensive player excluded
  S6  Per-team cap: only 3 players selected from a dominant club

Run:
  python -m pytest fpl/scripts/test_transfers_synthetic.py -v
  python fpl/scripts/test_transfers_synthetic.py       # runs without pytest
"""
from __future__ import annotations

import sqlite3
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import fpl.src.fpl_optimizer as opt
from fpl.src.fpl_optimizer import optimise

# ── Minimal schema matching what the optimizer queries ───────────────────────
_DDL = """
CREATE TABLE teams (
    team_id    INTEGER PRIMARY KEY,
    name       TEXT,
    short_name TEXT
);
CREATE TABLE players (
    player_id       INTEGER PRIMARY KEY,
    web_name        TEXT,
    first_name      TEXT,
    second_name     TEXT,
    team_id         INTEGER,
    position        INTEGER,   -- 1=GK 2=DEF 3=MID 4=FWD
    current_cost    INTEGER,
    corners_order   INTEGER,
    freekicks_order INTEGER,
    penalties_order INTEGER
);
CREATE TABLE player_gameweek_history (
    player_id       INTEGER,
    minutes         REAL,
    goals_scored    REAL,
    assists         REAL,
    clean_sheets    REAL,
    bonus           REAL,
    total_points    REAL,
    expected_goals  REAL,
    expected_assists REAL
);
CREATE TABLE fixtures (
    fixture_id   INTEGER PRIMARY KEY,
    gameweek_id  INTEGER,
    home_team_id INTEGER,
    away_team_id INTEGER,
    finished     INTEGER DEFAULT 0,
    home_score   REAL,
    away_score   REAL
);
"""

# ── Player spec helpers ───────────────────────────────────────────────────────
# Each player is (pid, pos_code, team_id, cost_tenths, goals_per_gw)
# pos_code: 1=GK 2=DEF 3=MID 4=FWD
# goals_per_gw controls projected_pts via the history rows

_N_HISTORY = 6  # gameweeks of history per player (enough for min_gws=3 threshold)


def _make_db(players: list[tuple]) -> Path:
    """
    Build a temp SQLite DB from a list of player specs.
    Returns the path; caller is responsible for unlinking.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    path = Path(tmp.name)
    conn = sqlite3.connect(path)
    conn.executescript(_DDL)

    team_ids = sorted({p[2] for p in players})
    conn.executemany(
        "INSERT OR IGNORE INTO teams VALUES (?,?,?)",
        [(tid, f"Club{tid}", f"C{tid}") for tid in team_ids],
    )

    for pid, pos, tid, cost, goals in players:
        conn.execute(
            "INSERT INTO players VALUES (?,?,?,?,?,?,?,NULL,NULL,NULL)",
            (pid, f"P{pid}", "X", f"P{pid}", tid, pos, cost),
        )
        for _ in range(_N_HISTORY):
            conn.execute(
                "INSERT INTO player_gameweek_history VALUES (?,90,?,0.2,0,0.5,?,?,0.1)",
                (pid, goals, goals * 5 + 2, goals),
            )

    conn.commit()
    conn.close()
    return path


def _run(players: list[tuple], **kwargs) -> dict:
    """Run optimise() against a synthetic DB, restoring module state afterwards."""
    db = _make_db(players)
    orig_db    = opt.DB_PATH
    orig_model = opt.MODEL_PATH
    try:
        opt.DB_PATH    = db
        opt.MODEL_PATH = Path("/nonexistent_model")  # skip DC; falls back gracefully
        return optimise(**kwargs)
    finally:
        opt.DB_PATH    = orig_db
        opt.MODEL_PATH = orig_model
        db.unlink(missing_ok=True)


# ── Canonical 15-player base squad ───────────────────────────────────────────
# Satisfies: 2 GK, 5 DEF, 5 MID, 3 FWD; all distinct teams; cost=50 each.
# goals_per_gw=0 → low (but stable) projected pts.

def _base_squad() -> list[tuple]:
    return [
        # (pid, pos, team, cost, goals/gw)
        (101, 1, 1, 50, 0.0),   # GK1
        (102, 1, 2, 50, 0.0),   # GK2
        (201, 2, 3, 50, 0.0),   # DEF1
        (202, 2, 4, 50, 0.0),   # DEF2
        (203, 2, 5, 50, 0.0),   # DEF3
        (204, 2, 6, 50, 0.0),   # DEF4
        (205, 2, 7, 50, 0.0),   # DEF5
        (301, 3, 1, 50, 0.2),   # MID1
        (302, 3, 2, 50, 0.2),   # MID2
        (303, 3, 3, 50, 0.2),   # MID3
        (304, 3, 4, 50, 0.2),   # MID4
        (305, 3, 5, 50, 0.2),   # MID5
        (401, 4, 1, 50, 0.4),   # FWD1
        (402, 4, 2, 50, 0.4),   # FWD2
        (403, 4, 3, 50, 0.4),   # FWD3
    ]


# ── Scenario S1 ──────────────────────────────────────────────────────────────

def test_s1_no_transfers_when_optimal():
    """
    When the existing squad is already globally optimal, the optimizer keeps it.
    """
    squad = _base_squad()
    existing_ids = [p[0] for p in squad]
    result = _run(squad, existing_squad_ids=existing_ids, free_transfers=1)
    assert result["n_transfers"] == 0, (
        f"Expected 0 transfers for optimal squad, got {result['n_transfers']}"
    )


# ── Scenario S2 ──────────────────────────────────────────────────────────────

def test_s2_takes_free_upgrade():
    """
    A clearly better FWD (goals=2.0 vs 0.4) is available at same cost.
    With 1 free transfer, the optimizer should bring them in at no hit.
    """
    squad = _base_squad()
    # Add a star FWD from team 8 (avoids team-cap conflict)
    star = (404, 4, 8, 50, 2.0)
    pool = squad + [star]
    existing_ids = [p[0] for p in squad]

    result = _run(pool, existing_squad_ids=existing_ids, free_transfers=1)

    squad_ids = {p["id"] for p in result["squad"]}
    assert star[0] in squad_ids, "Star FWD should have been transferred in"
    assert result["n_transfers"] == 1
    assert result["transfer_hit"] == 0, "1 free transfer: no hit expected"


# ── Scenario S3 ──────────────────────────────────────────────────────────────

def test_s3_refuses_marginal_upgrade_with_hit():
    """
    An upgrade worth ~+1.5 pts/GW is available, but costs a 4-pt hit
    (free_transfers=0). Net gain is negative: optimizer keeps existing player.
    """
    squad = _base_squad()
    # Marginal upgrade: goals=0.8 vs FWD1's goals=0.4  (diff ≈ +1.5 pts projected)
    marginal = (405, 4, 8, 50, 0.8)
    pool = squad + [marginal]
    existing_ids = [p[0] for p in squad]

    result = _run(pool, existing_squad_ids=existing_ids, free_transfers=0, horizon=1)

    squad_ids = {p["id"] for p in result["squad"]}
    assert marginal[0] not in squad_ids, (
        "Marginal upgrade should be refused when hit exceeds gain"
    )


# ── Scenario S4 ──────────────────────────────────────────────────────────────

def test_s4_takes_strong_upgrade_despite_hit():
    """
    A star player worth ~+6 pts/GW more than the weakest FWD in the squad
    is available; a 4-pt hit still leaves a net gain. Should transfer in.
    """
    squad = _base_squad()
    # Strong upgrade: goals=2.0 vs FWD weakest (goals=0.4) → diff ≈ +6 pts projected
    star = (406, 4, 8, 50, 2.0)
    pool = squad + [star]
    existing_ids = [p[0] for p in squad]

    result = _run(pool, existing_squad_ids=existing_ids, free_transfers=0, horizon=1)

    squad_ids = {p["id"] for p in result["squad"]}
    assert star[0] in squad_ids, (
        "Strong upgrade should be taken even when it costs a hit"
    )


# ── Scenario S5 ──────────────────────────────────────────────────────────────

def test_s5_selling_price_budget():
    """
    Players in the existing squad have risen in price since purchase.
    Selling prices (purchase + half rise) are lower than current market prices.
    An expensive player that looks affordable using current prices is actually
    out of reach once selling prices are applied.

    Setup:
      bank = 0; 15 players at cost=55 (current), selling_price=50 each.
      Budget with current prices: 0 + 15*55 = 825
      Budget with selling prices: 0 + 15*50 = 750
      Expensive FWD costs 76 — fits in 825 but not in 750.
    """
    # All 15 existing players: cost=55 (rose from 50), team spread across 10 clubs
    def _tid(idx): return (idx % 8) + 1   # 8 teams, ≤2 per team → within cap

    squad = [
        (101, 1, 1, 55, 0.0),  # GK
        (102, 1, 2, 55, 0.0),  # GK
        (201, 2, 1, 55, 0.0),  # DEF
        (202, 2, 2, 55, 0.0),  # DEF
        (203, 2, 3, 55, 0.0),  # DEF
        (204, 2, 4, 55, 0.0),  # DEF
        (205, 2, 5, 55, 0.0),  # DEF
        (301, 3, 3, 55, 0.2),  # MID
        (302, 3, 4, 55, 0.2),  # MID
        (303, 3, 5, 55, 0.2),  # MID
        (304, 3, 6, 55, 0.2),  # MID
        (305, 3, 7, 55, 0.2),  # MID
        (401, 4, 6, 55, 0.4),  # FWD
        (402, 4, 7, 55, 0.4),  # FWD
        (403, 4, 8, 55, 0.4),  # FWD
    ]
    # The star FWD who costs 76 — better than anything in the existing squad
    expensive_star = (404, 4, 9, 76, 3.0)
    # Affordable alt FWD at cost=55
    cheap_alt = (405, 4, 9, 55, 0.5)
    pool = squad + [expensive_star, cheap_alt]

    existing_ids = [p[0] for p in squad]
    # selling_price = 50 for each existing player (bought at 50, now worth 55)
    selling_prices = {p[0]: 50 for p in squad}

    # budget = bank(0) + sum(current_cost) = 0 + 15*55 = 825
    budget_naive = 15 * 55

    result = _run(
        pool,
        budget=budget_naive,
        existing_squad_ids=existing_ids,
        free_transfers=5,
        selling_prices=selling_prices,
    )

    squad_ids = {p["id"] for p in result["squad"]}
    assert expensive_star[0] not in squad_ids, (
        "Expensive star should be excluded: selling prices make them unaffordable"
    )


# ── Scenario S6 ──────────────────────────────────────────────────────────────

def test_s6_per_team_cap():
    """
    Four elite players all belong to team 9, each better than all alternatives.
    Per-team cap of 3 must prevent the 4th from being selected.
    """
    # Base squad minus one FWD slot (14 players, 4 teams, all with goalscorers)
    pool = [
        (101, 1, 1, 50, 0.0),  # GK
        (102, 1, 2, 50, 0.0),  # GK
        (201, 2, 3, 50, 0.0),  # DEF
        (202, 2, 4, 50, 0.0),  # DEF
        (203, 2, 5, 50, 0.0),  # DEF
        (204, 2, 6, 50, 0.0),  # DEF
        (205, 2, 7, 50, 0.0),  # DEF
        (301, 3, 1, 50, 0.1),  # MID
        (302, 3, 2, 50, 0.1),  # MID
        (303, 3, 3, 50, 0.1),  # MID
        (304, 3, 4, 50, 0.1),  # MID
        (305, 3, 5, 50, 0.1),  # MID
        # 3 ordinary FWDs from various clubs
        (401, 4, 6, 50, 0.2),
        (402, 4, 7, 50, 0.2),
        (403, 4, 8, 50, 0.2),
        # 4 elite FWDs all from team 9 — easily best in the pool
        (501, 4, 9, 50, 3.0),
        (502, 4, 9, 50, 3.0),
        (503, 4, 9, 50, 3.0),
        (504, 4, 9, 50, 3.0),
    ]

    result = _run(pool, per_team_cap=3)

    team9_in_squad = [p for p in result["squad"] if p["team_id"] == 9]
    assert len(team9_in_squad) <= 3, (
        f"Per-team cap violated: {len(team9_in_squad)} players from team 9"
    )
    assert len(team9_in_squad) == 3, (
        "Exactly 3 players from team 9 should be selected (cap is binding)"
    )


# ── Scenario S7 ──────────────────────────────────────────────────────────────

def test_s7_risen_player_retention():
    """
    A player in the existing squad has risen since purchase:
      current_cost=55, selling_price=50.
    The squad is otherwise optimal. The correct answer is 0 transfers.

    This tests the Gap 1 cost-row fix: retained players must be costed
    at selling_price in the budget constraint (not current_cost), otherwise
    keeping a risen player appears unaffordable even though no cash changes
    hands for a retention.

    Without the fix: effective_budget drops by 5 (selling–market delta)
    AND the risen player still costs 55 in the constraint → net 10 tighter
    than reality → solver is forced into an unwanted sale.
    """
    # One GK has risen: current_cost=55 but selling_price=50.
    # All others: current_cost=50, no price rise (selling_price=50).
    squad = [
        (101, 1, 1, 55, 0.0),   # GK1 — risen player
        (102, 1, 2, 50, 0.0),   # GK2
        (201, 2, 3, 50, 0.0),   # DEF1
        (202, 2, 4, 50, 0.0),   # DEF2
        (203, 2, 5, 50, 0.0),   # DEF3
        (204, 2, 6, 50, 0.0),   # DEF4
        (205, 2, 7, 50, 0.0),   # DEF5
        (301, 3, 1, 50, 0.2),   # MID1
        (302, 3, 2, 50, 0.2),   # MID2
        (303, 3, 3, 50, 0.2),   # MID3
        (304, 3, 4, 50, 0.2),   # MID4
        (305, 3, 5, 50, 0.2),   # MID5
        (401, 4, 1, 50, 0.4),   # FWD1
        (402, 4, 2, 50, 0.4),   # FWD2
        (403, 4, 3, 50, 0.4),   # FWD3
    ]
    existing_ids = [p[0] for p in squad]
    # GK1 selling price = 50 (bought at 45, now worth 55 → floor((55-45)/2)=5 → 45+5=50)
    selling_prices = {101: 50}
    for p in squad[1:]:
        selling_prices[p[0]] = p[3]   # no price rise: selling = current

    # budget = bank(0) + sum(current_cost) = 55 + 14*50 = 755
    budget = 55 + 14 * 50

    result = _run(
        squad,                        # no better alternatives in the pool
        budget=budget,
        existing_squad_ids=existing_ids,
        free_transfers=1,
        selling_prices=selling_prices,
    )

    assert result["n_transfers"] == 0, (
        f"Risen-price player should be retained; got {result['n_transfers']} transfer(s)"
    )


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_s1_no_transfers_when_optimal,
        test_s2_takes_free_upgrade,
        test_s3_refuses_marginal_upgrade_with_hit,
        test_s4_takes_strong_upgrade_despite_hit,
        test_s5_selling_price_budget,
        test_s6_per_team_cap,
        test_s7_risen_player_retention,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed}/{passed+failed} passed")
    sys.exit(0 if failed == 0 else 1)
