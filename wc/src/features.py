"""
features.py
Feature engineering for the World Cup Fantasy model.

Produces one row per (player, matchday) with:
  - Recent form stats (last N matches from tournament or qualifying)
  - Positional encoding
  - Team + opponent FIFA ranking
  - Tournament context (matchday number, stage)
  - Target column `fantasy_pts` (present in training data, absent at inference)
"""

import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "wc.db"

POSITION_ENC = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
FORM_MATCHES  = 5  # how many recent matches to compute rolling stats over


def _conn() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


def _load_all_stats() -> pd.DataFrame:
    """All player_match_stats joined with match context."""
    conn = _conn()
    df = pd.read_sql_query(
        """
        SELECT
            pms.player_id, pms.match_id, pms.team_id, pms.minutes,
            pms.goals, pms.assists, pms.xg, pms.xa, pms.shots,
            pms.key_passes, pms.yellow_cards, pms.red_cards,
            pms.saves, pms.clean_sheet, pms.fantasy_pts,
            p.position, p.name AS player_name,
            m.tournament, m.matchday, m.stage,
            m.home_team_id, m.away_team_id,
            CASE WHEN pms.team_id = m.home_team_id THEN m.away_team_id
                 ELSE m.home_team_id END AS opp_team_id,
            th.fifa_rank AS team_rank,
            m.match_date
        FROM player_match_stats pms
        JOIN players p ON pms.player_id = p.player_id
        JOIN matches m ON pms.match_id  = m.match_id
        LEFT JOIN teams th ON pms.team_id = th.team_id
        ORDER BY pms.player_id, m.match_date, m.matchday
        """,
        conn,
    )
    conn.close()
    return df


def _load_qualifying_stats() -> pd.DataFrame:
    conn = _conn()
    df = pd.read_sql_query(
        """
        SELECT q.player_id, q.tournament, q.matches, q.minutes,
               q.goals, q.assists, q.xg, q.xa, q.shots,
               p.position
        FROM qualifying_stats q
        JOIN players p ON q.player_id = p.player_id
        """,
        conn,
    )
    conn.close()
    return df


def _load_opp_ranks() -> dict:
    conn = _conn()
    rows = conn.execute("SELECT team_id, fifa_rank FROM teams").fetchall()
    conn.close()
    return {r[0]: (r[1] or 50) for r in rows}


def _rolling_form(group: pd.DataFrame, n: int = FORM_MATCHES) -> pd.DataFrame:
    """
    For each row in group (sorted by match_date), compute rolling stats
    from the previous n matches (excluding the current row).
    """
    cols = ["goals", "assists", "xg", "xa", "shots", "key_passes", "minutes",
            "yellow_cards", "red_cards", "saves", "clean_sheet", "fantasy_pts"]

    for col in cols:
        if col in group.columns:
            group[f"{col}_roll"] = (
                group[col]
                .shift(1)
                .rolling(window=n, min_periods=1)
                .mean()
                .fillna(group[col].mean())
            )
        else:
            group[f"{col}_roll"] = 0.0

    group["xg_per90_roll"] = (
        group["xg_roll"] / (group["minutes_roll"] / 90 + 1e-5)
    ).clip(0, 5)
    group["xa_per90_roll"] = (
        group["xa_roll"] / (group["minutes_roll"] / 90 + 1e-5)
    ).clip(0, 3)
    group["goals_per90_roll"] = (
        group["goals_roll"] / (group["minutes_roll"] / 90 + 1e-5)
    ).clip(0, 3)
    return group


def build_training_features(
    min_minutes: int = 1,
    tournaments: Optional[list] = None,
) -> pd.DataFrame:
    """
    Build labelled training rows from historical WC matches.
    Each row = one player in one match.  Target column = `fantasy_pts`.

    Only includes rows where minutes > 0.
    """
    stats = _load_all_stats()
    if tournaments:
        stats = stats[stats["tournament"].isin(tournaments)]

    opp_ranks = _load_opp_ranks()
    stats["opp_rank"]  = stats["opp_team_id"].map(opp_ranks).fillna(50).astype(int)
    stats["pos_enc"]   = stats["position"].map(POSITION_ENC).fillna(2)
    stats["team_rank"] = stats["team_rank"].fillna(30).astype(int)

    stats = stats[stats["minutes"] >= min_minutes].copy()
    stats = stats.sort_values(["player_id", "match_date", "matchday"])

    rows = []
    for player_id, group in stats.groupby("player_id", sort=False):
        group = group.reset_index(drop=True)
        group = _rolling_form(group)
        rows.append(group)

    df = pd.concat(rows, ignore_index=True)

    # ── Select final feature set ───────────────────────────────────────────
    feature_cols = [
        "player_id", "match_id", "player_name",
        "goals_roll", "assists_roll",
        "xg_roll", "xa_roll", "xg_per90_roll", "xa_per90_roll",
        "goals_per90_roll", "minutes_roll",
        "shots_roll", "key_passes_roll",
        "clean_sheet_roll", "saves_roll",
        "pos_enc", "team_rank", "opp_rank", "matchday",
        "fantasy_pts",  # target
    ]
    return df[[c for c in feature_cols if c in df.columns]].copy()


def build_prediction_features(
    matchday: int,
    tournament: str = "WC2026",
    qualifying_tournament: str = "WCQ2026",
) -> pd.DataFrame:
    """
    Build inference features for all fantasy players in the upcoming matchday.

    Falls back to qualifying_stats when no tournament form is available
    (i.e., matchday 1).  For subsequent matchdays, blends qualifying
    and in-tournament form.

    Returns one row per fantasy player with the same columns as
    build_training_features (minus `fantasy_pts`).
    """
    conn = _conn()

    # Fantasy players + their team fixture this matchday
    fp = pd.read_sql_query(
        """
        SELECT fp.fantasy_id, fp.player_id, fp.name, fp.name_norm,
               fp.team_id, fp.position, fp.price, fp.ownership,
               f.home_team_id, f.away_team_id,
               CASE WHEN fp.team_id = f.home_team_id THEN f.away_team_id
                    ELSE f.home_team_id END AS opp_team_id,
               f.matchday
        FROM fantasy_players fp
        JOIN fixtures f ON fp.team_id IN (f.home_team_id, f.away_team_id)
        WHERE f.matchday = ?
        """,
        conn,
        params=(matchday,),
    )

    if fp.empty:
        conn.close()
        raise RuntimeError(
            f"No fantasy players with a fixture in matchday {matchday}. "
            "Run ingest_fixtures.py and ingest_fantasy_players.py first."
        )

    # In-tournament stats so far (if any matches played)
    in_tourney = pd.read_sql_query(
        """
        SELECT pms.player_id,
               AVG(pms.goals)        AS goals_roll,
               AVG(pms.assists)      AS assists_roll,
               AVG(pms.xg)           AS xg_roll,
               AVG(pms.xa)           AS xa_roll,
               AVG(pms.shots)        AS shots_roll,
               AVG(pms.key_passes)   AS key_passes_roll,
               AVG(pms.minutes)      AS minutes_roll,
               AVG(pms.clean_sheet)  AS clean_sheet_roll,
               AVG(pms.saves)        AS saves_roll
        FROM player_match_stats pms
        JOIN matches m ON pms.match_id = m.match_id
        WHERE m.tournament = ? AND m.matchday < ?
        GROUP BY pms.player_id
        """,
        conn,
        params=(tournament, matchday),
    )

    # Qualifying stats (pre-tournament form)
    qual = pd.read_sql_query(
        """
        SELECT player_id,
               CAST(goals AS REAL) / NULLIF(matches, 0)   AS goals_roll,
               CAST(assists AS REAL) / NULLIF(matches, 0)  AS assists_roll,
               xg / NULLIF(matches, 0)                     AS xg_roll,
               xa / NULLIF(matches, 0)                     AS xa_roll,
               CAST(shots AS REAL) / NULLIF(matches, 0)   AS shots_roll,
               0.0 AS key_passes_roll,
               CAST(minutes AS REAL) / NULLIF(matches, 0)  AS minutes_roll,
               0.0 AS clean_sheet_roll,
               0.0 AS saves_roll
        FROM qualifying_stats
        WHERE tournament = ?
        """,
        conn,
        params=(qualifying_tournament,),
    )

    team_ranks = pd.read_sql_query("SELECT team_id, fifa_rank FROM teams", conn)
    conn.close()

    opp_ranks = team_ranks.set_index("team_id")["fifa_rank"].fillna(50).to_dict()

    form_cols = ["goals_roll", "assists_roll", "xg_roll", "xa_roll",
                 "shots_roll", "key_passes_roll", "minutes_roll",
                 "clean_sheet_roll", "saves_roll"]

    # Build form: prefer in-tournament, fall back to qualifying
    if not in_tourney.empty and not qual.empty:
        # Blend: weighted average (in-tournament gets more weight as matchday increases)
        wt = min((matchday - 1) / 3.0, 1.0)  # 0 → 1 over 3 matchdays
        combined = qual.merge(in_tourney, on="player_id", how="left", suffixes=("_q", "_t"))
        for col in form_cols:
            cq, ct = f"{col}_q", f"{col}_t"
            if cq in combined.columns and ct in combined.columns:
                combined[col] = combined[ct].fillna(combined[cq]) * wt + \
                                combined[cq].fillna(0) * (1 - wt)
            elif cq in combined.columns:
                combined[col] = combined[cq]
            else:
                combined[col] = combined.get(ct, 0.0)
        form = combined[["player_id"] + form_cols]
    elif not in_tourney.empty:
        form = in_tourney
    elif not qual.empty:
        form = qual
    else:
        form = pd.DataFrame({"player_id": []})

    # Merge form onto fantasy players (by player_id if matched, else zeros)
    df = fp.merge(form, on="player_id", how="left")
    for col in form_cols:
        df[col] = df.get(col, pd.Series(dtype=float)).fillna(0.0)

    df["opp_rank"]  = df["opp_team_id"].map(opp_ranks).fillna(50).astype(int)
    df["team_rank"] = df["team_id"].map(opp_ranks).fillna(30).astype(int)
    df["pos_enc"]   = df["position"].map({"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}).fillna(2)

    df["xg_per90_roll"]    = (df["xg_roll"] / (df["minutes_roll"] / 90 + 1e-5)).clip(0, 5)
    df["xa_per90_roll"]    = (df["xa_roll"] / (df["minutes_roll"] / 90 + 1e-5)).clip(0, 3)
    df["goals_per90_roll"] = (df["goals_roll"] / (df["minutes_roll"] / 90 + 1e-5)).clip(0, 3)

    return df


FEATURE_COLS = [
    "goals_roll", "assists_roll",
    "xg_roll", "xa_roll", "xg_per90_roll", "xa_per90_roll",
    "goals_per90_roll", "minutes_roll",
    "shots_roll", "key_passes_roll",
    "clean_sheet_roll", "saves_roll",
    "pos_enc", "team_rank", "opp_rank", "matchday",
]
