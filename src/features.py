"""
features.py
Feature engineering for the FPL ML predictor.

Builds a (player, gameweek) feature matrix from the database, with strict
time-awareness: every feature for (player, gameweek=g) is derived only from
data with gameweek_id < g.

Both training-time and prediction-time pipelines use the same underlying
rolling-feature computation to avoid training/serving skew.
"""

import re
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "fpl.db"

FORM_WINDOWS = [3, 5, 10]
MIN_MINUTES_TO_COUNT = 60

INJURY_KEYWORDS = re.compile(
    r"\b(?:knock|ankle|knee|hamstring|calf|hip|thigh|doubt|injur|fitness|fit)\b",
    re.IGNORECASE,
)
SUSPENSION_KEYWORDS = re.compile(
    r"\b(?:suspended|suspension|red card|ban)\b",
    re.IGNORECASE,
)


# ---------- SQL fetchers ----------

def _fetch_history(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(
        """
        SELECT player_id, gameweek_id, minutes, total_points,
               expected_goals, expected_assists, defensive_contribution
        FROM player_gameweek_history
        """,
        conn,
    )


def _fetch_players(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT player_id, team_id, position, current_cost FROM players",
        conn,
    )


def _fetch_fixtures(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(
        """
        SELECT gameweek_id, home_team_id AS team_id, home_team_difficulty AS fdr, 1 AS is_home
        FROM fixtures
        WHERE gameweek_id IS NOT NULL
        UNION ALL
        SELECT gameweek_id, away_team_id AS team_id, away_team_difficulty AS fdr, 0 AS is_home
        FROM fixtures
        WHERE gameweek_id IS NOT NULL
        """,
        conn,
    )


def _fetch_snapshots(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(
        """
        SELECT player_id, gameweek_id, chance_of_playing_next, news, now_cost
        FROM player_snapshots
        WHERE snapshot_id IN (
            SELECT MAX(snapshot_id)
            FROM player_snapshots
            GROUP BY player_id, gameweek_id
        )
        """,
        conn,
    )


# ---------- Core form computation ----------

def _compute_form_for_targets(
    history: pd.DataFrame,
    targets: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each (player_id, gameweek_id) row in `targets`, compute rolling form
    features using ONLY rows in `history` that:
        (a) belong to that player
        (b) have minutes >= MIN_MINUTES_TO_COUNT
        (c) have gameweek_id strictly less than the target's gameweek_id

    Returns a DataFrame with the same row count as `targets`, keyed by
    (player_id, gameweek_id), with one column per (stat, window) plus
    qualifying_games_{window}.
    """
    qualifying = history[history["minutes"] >= MIN_MINUTES_TO_COUNT].copy()
    qualifying = qualifying.sort_values(["player_id", "gameweek_id"]).reset_index(drop=True)

    stats = ["total_points", "expected_goals", "expected_assists",
             "defensive_contribution", "minutes"]

    # Pre-group qualifying history by player for fast lookup.
    history_by_player: dict[int, pd.DataFrame] = {
        pid: g for pid, g in qualifying.groupby("player_id", sort=False)
    }

    # Initialise output columns
    out = targets[["player_id", "gameweek_id"]].copy().reset_index(drop=True)
    for window in FORM_WINDOWS:
        for stat in stats:
            out[f"{stat}_mean_{window}"] = 0.0
        out[f"qualifying_games_{window}"] = 0

    # Compute per row. For ~25k rows this is fast enough; can be vectorised later if needed.
    for idx, row in out.iterrows():
        player_id = int(row["player_id"])
        target_gw = int(row["gameweek_id"])
        player_history = history_by_player.get(player_id)
        if player_history is None:
            continue
        prior = player_history[player_history["gameweek_id"] < target_gw]
        if prior.empty:
            continue
        for window in FORM_WINDOWS:
            window_rows = prior.tail(window)
            for stat in stats:
                out.at[idx, f"{stat}_mean_{window}"] = float(window_rows[stat].mean())
            out.at[idx, f"qualifying_games_{window}"] = int(len(window_rows))

    return out


# ---------- Other feature builders ----------

def _build_news_features(snapshots: pd.DataFrame) -> pd.DataFrame:
    df = snapshots.copy()
    news = df["news"].fillna("").astype(str)
    df["has_news"] = (news.str.strip() != "").astype(int)
    df["news_injury_flag"] = news.str.contains(INJURY_KEYWORDS).astype(int)
    df["news_suspension_flag"] = news.str.contains(SUSPENSION_KEYWORDS).astype(int)
    return df[[
        "player_id", "gameweek_id",
        "chance_of_playing_next", "now_cost",
        "has_news", "news_injury_flag", "news_suspension_flag",
    ]]


def _build_fixture_features(fixtures: pd.DataFrame) -> pd.DataFrame:
    return (
        fixtures.groupby(["team_id", "gameweek_id"])
        .agg(fdr=("fdr", "mean"),
             num_fixtures=("fdr", "count"),
             is_home=("is_home", "max"))
        .reset_index()
    )


def _apply_defensive_fills(df: pd.DataFrame) -> pd.DataFrame:
    form_cols = [c for c in df.columns
                 if any(c.startswith(f"{s}_mean_") for s in [
                     "total_points", "expected_goals", "expected_assists",
                     "defensive_contribution", "minutes"])
                 or c.startswith("qualifying_games_")]
    df[form_cols] = df[form_cols].fillna(0.0)
    df["chance_of_playing_next"] = df["chance_of_playing_next"].fillna(100).astype(float)
    df["has_news"] = df["has_news"].fillna(0).astype(int)
    df["news_injury_flag"] = df["news_injury_flag"].fillna(0).astype(int)
    df["news_suspension_flag"] = df["news_suspension_flag"].fillna(0).astype(int)
    df["fdr"] = df["fdr"].fillna(3.0)
    df["num_fixtures"] = df["num_fixtures"].fillna(0).astype(int)
    df["is_home"] = df["is_home"].fillna(0).astype(int)
    if "now_cost" in df.columns:
        df["now_cost"] = df["now_cost"].fillna(df.get("current_cost"))
    return df


# ---------- Public API ----------

def build_training_data(
    min_gw: int = 8,
    max_gw: Optional[int] = None,
) -> pd.DataFrame:
    """One row per (player, played-gameweek) with features and target."""
    conn = sqlite3.connect(DB_PATH)
    try:
        history = _fetch_history(conn)
        players = _fetch_players(conn)
        fixtures = _fetch_fixtures(conn)
        snapshots = _fetch_snapshots(conn)
    finally:
        conn.close()

    if max_gw is None:
        max_gw = int(history["gameweek_id"].max())

    samples = history[["player_id", "gameweek_id", "total_points"]].rename(
        columns={"total_points": "actual_points"}
    )
    samples = samples[
        (samples["gameweek_id"] >= min_gw) & (samples["gameweek_id"] <= max_gw)
    ].reset_index(drop=True)

    form = _compute_form_for_targets(history, samples[["player_id", "gameweek_id"]])

    df = samples.merge(form, on=["player_id", "gameweek_id"], how="left")

    news = _build_news_features(snapshots)
    df = df.merge(news, on=["player_id", "gameweek_id"], how="left")

    df = df.merge(players, on="player_id", how="left")

    fix = _build_fixture_features(fixtures)
    df = df.merge(fix, on=["team_id", "gameweek_id"], how="left")

    df = _apply_defensive_fills(df)
    return df


def build_prediction_features(target_gw: int) -> pd.DataFrame:
    """Features for predicting target_gw, using only data with gameweek_id < target_gw for form."""
    conn = sqlite3.connect(DB_PATH)
    try:
        history = _fetch_history(conn)
        players = _fetch_players(conn)
        fixtures = _fetch_fixtures(conn)
        snapshots = _fetch_snapshots(conn)
    finally:
        conn.close()

    targets = players[["player_id"]].copy()
    targets["gameweek_id"] = target_gw

    form = _compute_form_for_targets(history, targets)
    df = targets.merge(form, on=["player_id", "gameweek_id"], how="left")
    df = df.merge(players, on="player_id", how="left")

    # News snapshot at target gameweek; fall back to most recent if absent.
    snapshot_target = snapshots[snapshots["gameweek_id"] == target_gw]
    if snapshot_target.empty:
        snapshot_target = (
            snapshots.sort_values("gameweek_id", ascending=False)
            .drop_duplicates(subset=["player_id"], keep="first")
        )
    news = _build_news_features(snapshot_target).drop(columns=["gameweek_id"])
    df = df.merge(news, on="player_id", how="left")

    fix_target = _build_fixture_features(
        fixtures[fixtures["gameweek_id"] == target_gw]
    )
    df = df.merge(fix_target, on=["team_id", "gameweek_id"], how="left")

    df = _apply_defensive_fills(df)
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    excluded = {"player_id", "gameweek_id", "actual_points", "team_id", "current_cost"}
    return [c for c in df.columns if c not in excluded]