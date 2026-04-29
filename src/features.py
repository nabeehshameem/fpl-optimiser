"""
features.py
Feature engineering for the FPL ML predictor.

Builds a (player, gameweek) feature matrix from the database, with strict
time-awareness: every feature for (player, gameweek=g) is derived only from
data with gameweek_id < g.

Two main entry points:
  - build_training_data(min_gw, max_gw): returns features + target for past gameweeks
  - build_prediction_features(target_gw): returns features for predicting target_gw
"""

import re
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "fpl.db"

# Form windows in gameweeks
FORM_WINDOWS = [3, 5, 10]
MIN_MINUTES_TO_COUNT = 60

# News keyword sets (lowercase matching)
INJURY_KEYWORDS = re.compile(
    r"\b(?:knock|ankle|knee|hamstring|calf|hip|thigh|doubt|injur|fitness|fit)\b",
    re.IGNORECASE,
)
SUSPENSION_KEYWORDS = re.compile(
    r"\b(?:suspended|suspension|red card|ban)\b",
    re.IGNORECASE,
)


# ---------- SQL helpers ----------

def _fetch_history(conn: sqlite3.Connection) -> pd.DataFrame:
    """All historical gameweek rows. Used to compute rolling form features."""
    query = """
        SELECT
            player_id,
            gameweek_id,
            minutes,
            total_points,
            expected_goals,
            expected_assists,
            defensive_contribution
        FROM player_gameweek_history
    """
    return pd.read_sql_query(query, conn)


def _fetch_players(conn: sqlite3.Connection) -> pd.DataFrame:
    query = """
        SELECT player_id, team_id, position, current_cost
        FROM players
    """
    return pd.read_sql_query(query, conn)


def _fetch_fixtures(conn: sqlite3.Connection) -> pd.DataFrame:
    """All fixtures in long format: one row per (gameweek, team), home/away aware."""
    query = """
        SELECT gameweek_id, home_team_id AS team_id, home_team_difficulty AS fdr, 1 AS is_home
        FROM fixtures
        WHERE gameweek_id IS NOT NULL
        UNION ALL
        SELECT gameweek_id, away_team_id AS team_id, away_team_difficulty AS fdr, 0 AS is_home
        FROM fixtures
        WHERE gameweek_id IS NOT NULL
    """
    return pd.read_sql_query(query, conn)


def _fetch_snapshots(conn: sqlite3.Connection) -> pd.DataFrame:
    """Latest snapshot per (player, gameweek). Used for availability + news features."""
    query = """
        SELECT
            player_id,
            gameweek_id,
            chance_of_playing_next,
            news,
            now_cost
        FROM player_snapshots
        WHERE snapshot_id IN (
            SELECT MAX(snapshot_id)
            FROM player_snapshots
            GROUP BY player_id, gameweek_id
        )
    """
    return pd.read_sql_query(query, conn)


# ---------- Feature builders ----------

def _compute_rolling_form_features(history: pd.DataFrame) -> pd.DataFrame:
    """
    For every (player, gameweek) in history, compute rolling features over the
    player's last N qualifying (60+ minute) gameweeks STRICTLY before that gameweek.

    Returns a DataFrame keyed by (player_id, gameweek_id).
    """
    # Keep only qualifying appearances for form calc
    qualifying = history[history["minutes"] >= MIN_MINUTES_TO_COUNT].copy()
    qualifying = qualifying.sort_values(["player_id", "gameweek_id"])

    # We'll compute, for each (player, gameweek), rolling features over the
    # qualifying rows that came BEFORE that gameweek. This is a per-player
    # cumulative operation.

    # Per-player, shift the qualifying stats by 1 so we never include the current row.
    grouped = qualifying.groupby("player_id", group_keys=False)

    out = qualifying[["player_id", "gameweek_id"]].copy()

    for window in FORM_WINDOWS:
        # Rolling over previous N rows (excluding current via shift(1))
        for stat in ["total_points", "expected_goals", "expected_assists",
                     "defensive_contribution", "minutes"]:
            col = f"{stat}_mean_{window}"
            out[col] = (
                grouped[stat]
                .apply(lambda s: s.shift(1).rolling(window=window, min_periods=1).mean())
                .reset_index(level=0, drop=True)
            )

        # Number of qualifying games actually found in the window
        out[f"qualifying_games_{window}"] = (
            grouped["gameweek_id"]
            .apply(lambda s: s.shift(1).rolling(window=window, min_periods=1).count())
            .reset_index(level=0, drop=True)
        )

    return out


def _build_news_features(snapshots: pd.DataFrame) -> pd.DataFrame:
    """Cheap text-based flags from the snapshot's news field."""
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
    """
    For each (team, gameweek), aggregate to one row capturing:
      - mean fdr across that gameweek's fixtures
      - num_fixtures
      - is_home (1.0 if any home fixture, else 0.0; for DGWs averaging is fine)
    """
    agg = fixtures.groupby(["team_id", "gameweek_id"]).agg(
        fdr=("fdr", "mean"),
        num_fixtures=("fdr", "count"),
        is_home=("is_home", "max"),
    ).reset_index()
    return agg


# ---------- Public API ----------

def build_training_data(
    min_gw: int = 8,
    max_gw: Optional[int] = None,
) -> pd.DataFrame:
    """
    Build a training-ready DataFrame: one row per (player, played-gameweek)
    with engineered features and the target column 'actual_points'.

    Args:
        min_gw: skip gameweeks before this (early-season has weak form signal)
        max_gw: include gameweeks up to and including this; None means all available
    """
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

    # Rolling form features (one row per qualifying historical appearance)
    form_features = _compute_rolling_form_features(history)

    # News + availability features (one row per (player, gameweek) snapshot)
    news_features = _build_news_features(snapshots)

    # Fixture features (one row per (team, gameweek))
    fixture_features = _build_fixture_features(fixtures)

    # Target rows: every historical appearance is a training sample
    samples = history[["player_id", "gameweek_id", "total_points"]].rename(
        columns={"total_points": "actual_points"}
    )
    samples = samples[
        (samples["gameweek_id"] >= min_gw) & (samples["gameweek_id"] <= max_gw)
    ]

    # Join form features (left, since some early-career rows lack them)
    df = samples.merge(form_features, on=["player_id", "gameweek_id"], how="left")

    # Join news features
    df = df.merge(news_features, on=["player_id", "gameweek_id"], how="left")

    # Join player metadata
    df = df.merge(players, on="player_id", how="left")

    # Join fixture features by (team, gameweek)
    df = df.merge(fixture_features, on=["team_id", "gameweek_id"], how="left")

    # Defensive fills for missing form (early-season/new player)
    form_cols = [c for c in df.columns if any(c.startswith(f"{s}_mean_") for s in [
        "total_points", "expected_goals", "expected_assists",
        "defensive_contribution", "minutes",
    ]) or c.startswith("qualifying_games_")]
    df[form_cols] = df[form_cols].fillna(0.0)

    df["chance_of_playing_next"] = df["chance_of_playing_next"].fillna(100).astype(float)
    df["has_news"] = df["has_news"].fillna(0).astype(int)
    df["news_injury_flag"] = df["news_injury_flag"].fillna(0).astype(int)
    df["news_suspension_flag"] = df["news_suspension_flag"].fillna(0).astype(int)
    df["fdr"] = df["fdr"].fillna(3.0)
    df["num_fixtures"] = df["num_fixtures"].fillna(0).astype(int)
    df["is_home"] = df["is_home"].fillna(0).astype(int)
    df["now_cost"] = df["now_cost"].fillna(df["current_cost"])

    return df


def build_prediction_features(target_gw: int) -> pd.DataFrame:
    """
    Build features for predicting target_gw. Uses data strictly before target_gw
    for form features, and the latest snapshot at or before target_gw for news.
    """
    conn = sqlite3.connect(DB_PATH)
    try:
        history = _fetch_history(conn)
        players = _fetch_players(conn)
        fixtures = _fetch_fixtures(conn)
        snapshots = _fetch_snapshots(conn)
    finally:
        conn.close()

    # Restrict history to strictly before target_gw
    history_past = history[history["gameweek_id"] < target_gw]

    # Compute rolling form, then take the most recent row per player (the "as-of" view)
    form_all = _compute_rolling_form_features(
        pd.concat([history_past, _make_dummy_target_row(history_past, target_gw)], ignore_index=True)
    )
    form_target = form_all[form_all["gameweek_id"] == target_gw].drop(columns=["gameweek_id"])

    # News snapshot at target gameweek (latest snapshot for that gameweek)
    snapshot_target = snapshots[snapshots["gameweek_id"] == target_gw]
    news = _build_news_features(snapshot_target)

    # Fixtures for target gameweek
    fixture_target = _build_fixture_features(
        fixtures[fixtures["gameweek_id"] == target_gw]
    )

    df = players.copy()
    df["gameweek_id"] = target_gw
    df = df.merge(form_target, on="player_id", how="left")
    df = df.merge(news, on=["player_id", "gameweek_id"], how="left")
    df = df.merge(fixture_target, on=["team_id", "gameweek_id"], how="left")

    # Same defensive fills as training
    form_cols = [c for c in df.columns if any(c.startswith(f"{s}_mean_") for s in [
        "total_points", "expected_goals", "expected_assists",
        "defensive_contribution", "minutes",
    ]) or c.startswith("qualifying_games_")]
    df[form_cols] = df[form_cols].fillna(0.0)
    df["chance_of_playing_next"] = df["chance_of_playing_next"].fillna(100).astype(float)
    df["has_news"] = df["has_news"].fillna(0).astype(int)
    df["news_injury_flag"] = df["news_injury_flag"].fillna(0).astype(int)
    df["news_suspension_flag"] = df["news_suspension_flag"].fillna(0).astype(int)
    df["fdr"] = df["fdr"].fillna(3.0)
    df["num_fixtures"] = df["num_fixtures"].fillna(0).astype(int)
    df["is_home"] = df["is_home"].fillna(0).astype(int)
    df["now_cost"] = df["now_cost"].fillna(df["current_cost"])

    return df


def _make_dummy_target_row(history_past: pd.DataFrame, target_gw: int) -> pd.DataFrame:
    """
    Helper for build_prediction_features: appends a dummy 'current row' per player
    at target_gw so the rolling-shift logic generates a feature row for them.
    The dummy row's stat values don't matter because shift(1) excludes them.
    """
    players_with_history = history_past["player_id"].unique()
    return pd.DataFrame({
        "player_id": players_with_history,
        "gameweek_id": target_gw,
        "minutes": MIN_MINUTES_TO_COUNT,  # dummy, qualifying so it's included in shift
        "total_points": 0,
        "expected_goals": 0.0,
        "expected_assists": 0.0,
        "defensive_contribution": 0,
    })


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return the column names that should be used as model features."""
    excluded = {
        "player_id", "gameweek_id", "actual_points", "team_id", "current_cost",
    }
    return [c for c in df.columns if c not in excluded]