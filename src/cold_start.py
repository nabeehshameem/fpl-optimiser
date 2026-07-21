"""
cold_start.py

Early-season projection support. After season_rollover.py the live DB has
zero player history by design — its own verification asserts this — so the
DC predictor has nothing to compute player rates from and would project 0.0
for every player through GW1-GW3. An all-zeros frame hands the optimiser a
degenerate problem: it returns an arbitrary constraint-satisfying squad,
which would then be hash-committed and published as the model's opinion.

Two mechanisms, per the A+C decision:

A. ARCHIVE RATES. While the live DB holds fewer than COLD_START_MIN_GWS
   distinct gameweeks of history, player rates are read from the archived
   previous season (data/fpl_<label>.db). Fixtures, prices, positions and
   availability always come from the LIVE DB — only the per-player rates are
   historical.

   The switch is a row count, not a date: the moment GW3 finishes and the
   live DB has three gameweeks, archive rates are silently ignored. No flag,
   no manual step, no calendar to get wrong.

B. POSITIONAL PRIORS. Players with no history in the rates source at all —
   promoted-side squads, new signings, rookies — would otherwise be invisible
   to the optimiser. They fall back to priors fitted from the rates source
   itself: mean rates grouped by (position, £0.5m price bucket), backing off
   to position-wide means when a bucket is thin. Data-grounded, not
   hand-tuned, so a £4.5m Hull midfielder and an £8.0m Coventry striker get
   meaningfully different projections.

Priors apply in warm start too: a January signing has no current-season
history either, and the same fallback is correct there.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Live-history gameweeks required before archive rates are dropped.
COLD_START_MIN_GWS = 3

# Minimum samples for a (position, price-bucket) prior to be trusted.
MIN_BUCKET_SAMPLES = 5

PRICE_BUCKET_TENTHS = 5  # £0.5m bands

RATE_KEYS = ("avg_minutes", "avg_goals", "avg_assists", "cs_rate",
             "avg_bonus", "avg_xg", "avg_xa")


def price_bucket(cost_tenths: int) -> int:
    """£0.5m band. 45,47 -> 9; 50,54 -> 10."""
    return int(cost_tenths) // PRICE_BUCKET_TENTHS


def live_history_gw_count(conn: sqlite3.Connection) -> int:
    """Distinct gameweeks with any player history in the live DB."""
    try:
        return int(conn.execute(
            "SELECT COUNT(DISTINCT gameweek_id) FROM player_gameweek_history"
        ).fetchone()[0] or 0)
    except sqlite3.Error:
        return 0


def is_cold_start(conn: sqlite3.Connection) -> bool:
    return live_history_gw_count(conn) < COLD_START_MIN_GWS


def resolve_archive(archive_db_path: Path | None) -> Path:
    """Locate the archive DB, failing loudly rather than silently degrading."""
    if archive_db_path is not None:
        p = Path(archive_db_path)
        if not p.exists():
            raise RuntimeError(
                f"Cold start requires the archived season at {p}, which does "
                "not exist. Run season_rollover.py (which creates it) or pass "
                "the correct archive path."
            )
        return p
    candidates = sorted((PROJECT_ROOT / "data").glob("fpl_*.db"))
    if not candidates:
        raise RuntimeError(
            "Cold start: live DB has <3 gameweeks of history and no archive "
            "DB was found at data/fpl_*.db. Without one, every player would "
            "project 0.0 and the optimiser would return an arbitrary squad. "
            "Refusing to produce meaningless predictions."
        )
    return candidates[-1]


def fit_positional_priors(conn: sqlite3.Connection,
                          min_gws: int = 3) -> dict:
    """Mean rates by (position, price bucket) and by position, from `conn`.

    Returns {"bucket": {(pos, bucket): rates}, "position": {pos: rates},
             "global": rates} — each a dict of RATE_KEYS.
    """
    rows = conn.execute(f"""
        SELECT p.position, p.current_cost,
               AVG(h.minutes)        AS avg_minutes,
               AVG(h.goals_scored)   AS avg_goals,
               AVG(h.assists)        AS avg_assists,
               AVG(CASE WHEN h.clean_sheets >= 1 AND h.minutes >= 60
                        THEN 1.0 ELSE 0.0 END) AS cs_rate,
               AVG(h.bonus)          AS avg_bonus,
               AVG(h.expected_goals) AS avg_xg,
               AVG(h.expected_assists) AS avg_xa,
               COUNT(*)              AS n
        FROM player_gameweek_history h
        JOIN players p ON p.player_id = h.player_id
        WHERE h.minutes >= 1 AND p.position IS NOT NULL
        GROUP BY h.player_id
        HAVING COUNT(*) >= {int(min_gws)}
    """).fetchall()

    by_bucket: dict[tuple[int, int], list[dict]] = {}
    by_pos: dict[int, list[dict]] = {}
    every: list[dict] = []
    for pos, cost, mins, g, a, cs, bon, xg, xa, _n in rows:
        rate = {"avg_minutes": float(mins or 0), "avg_goals": float(g or 0),
                "avg_assists": float(a or 0), "cs_rate": float(cs or 0),
                "avg_bonus": float(bon or 0), "avg_xg": float(xg or 0),
                "avg_xa": float(xa or 0)}
        by_bucket.setdefault((int(pos), price_bucket(cost)), []).append(rate)
        by_pos.setdefault(int(pos), []).append(rate)
        every.append(rate)

    def mean(rs: list[dict]) -> dict:
        n = max(len(rs), 1)
        return {k: sum(r[k] for r in rs) / n for k in RATE_KEYS}

    return {
        "bucket": {k: mean(v) for k, v in by_bucket.items()
                   if len(v) >= MIN_BUCKET_SAMPLES},
        "position": {k: mean(v) for k, v in by_pos.items()},
        "global": mean(every) if every else {k: 0.0 for k in RATE_KEYS},
    }


def prior_for(priors: dict, position: int, cost_tenths: int) -> dict:
    """Best available prior: bucket -> position -> global."""
    b = priors["bucket"].get((int(position), price_bucket(cost_tenths)))
    if b is not None:
        return b
    p = priors["position"].get(int(position))
    if p is not None:
        return p
    return priors["global"]
