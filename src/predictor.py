"""
predictor.py
Prediction layer for the FPL optimiser.

Defines a Predictor protocol/interface and concrete implementations.
The optimiser layer depends only on the interface, not the implementation.

v1: NaivePredictor (no ML; rolling form * fixture difficulty * availability)
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "fpl.db"


# ---------- Helpers for FDR ----------

def fdr_to_modifier(fdr: Optional[int]) -> float:
    """
    Convert FPL Fixture Difficulty Rating (1-5) to a multiplicative scoring modifier.

    FDR 1 (easiest) -> 1.3x expected points
    FDR 5 (hardest) -> 0.7x expected points
    Linearly interpolated between.
    """
    if fdr is None:
        return 1.0  # no fixture info -> no modifier
    if fdr <= 1:
        return 1.3
    if fdr >= 5:
        return 0.7
    # Linear: 1 -> 1.3, 5 -> 0.7
    return 1.3 - (fdr - 1) * 0.15


# ---------- Naive Predictor ----------

class NaivePredictor:
    """
    A zero-ML baseline predictor.

    Prediction formula:
        predicted_points = recent_form * fixture_modifier * availability

    Where:
        recent_form    = mean total_points over the player's last N played gameweeks
        fixture_mod    = function of FDR for the player's team in the target gameweek
        availability   = chance_of_playing_next_round / 100
    """

    MODEL_NAME = "naive_v1"

    def __init__(self, db_path: Path = DB_PATH, form_window: int = 5):
        self.db_path = db_path
        self.form_window = form_window

    MIN_MINUTES_TO_COUNT = 60  # only "real" appearances count toward form

    def _get_recent_form(self, conn: sqlite3.Connection, target_gw: int) -> pd.DataFrame:
        """
        For each player: mean points and mean minutes over their last N qualifying
        gameweeks before target_gw. A 'qualifying' gameweek requires 60+ minutes
        (the FPL appearance-points threshold).
        """
        query = """
            SELECT
                player_id,
                AVG(total_points) AS recent_form,
                AVG(minutes)      AS avg_minutes,
                COUNT(*)          AS games_in_window
            FROM (
                SELECT
                    player_id,
                    total_points,
                    minutes,
                    ROW_NUMBER() OVER (
                        PARTITION BY player_id
                        ORDER BY gameweek_id DESC
                    ) AS rn
                FROM player_gameweek_history
                WHERE gameweek_id < ?
                  AND minutes >= ?
            )
            WHERE rn <= ?
            GROUP BY player_id
        """
        return pd.read_sql_query(
            query, conn,
            params=(target_gw, self.MIN_MINUTES_TO_COUNT, self.form_window),
        )

    def _get_fixture_difficulty(self, conn: sqlite3.Connection, target_gw: int) -> pd.DataFrame:
        """
        For each team in the target gameweek, return their FDR.
        Handles double gameweeks by averaging FDR across all fixtures that gameweek.
        """
        query = """
            SELECT team_id, AVG(fdr) AS fdr, COUNT(*) AS num_fixtures
            FROM (
                SELECT home_team_id AS team_id, home_team_difficulty AS fdr
                FROM fixtures
                WHERE gameweek_id = ?
                UNION ALL
                SELECT away_team_id AS team_id, away_team_difficulty AS fdr
                FROM fixtures
                WHERE gameweek_id = ?
            )
            GROUP BY team_id
        """
        return pd.read_sql_query(query, conn, params=(target_gw, target_gw))

    def _get_player_meta(self, conn: sqlite3.Connection, target_gw: int) -> pd.DataFrame:
        """
        For each player, fetch their team and best-available availability signal.

        Snapshots are only available for gameweeks where ingest_bootstrap was run.
        For gameweeks before any snapshot was captured, defaults to 100% availability.
        """
        query = """
            SELECT
                p.player_id,
                p.team_id,
                COALESCE(s.chance_of_playing_next, 100) AS availability
            FROM players p
            LEFT JOIN (
                SELECT player_id, chance_of_playing_next,
                       ROW_NUMBER() OVER (
                           PARTITION BY player_id
                           ORDER BY snapshot_time DESC
                       ) AS rn
                FROM player_snapshots
                WHERE gameweek_id <= ?
            ) s ON p.player_id = s.player_id AND s.rn = 1
        """
        df = pd.read_sql_query(query, conn, params=(target_gw,))
        # Defensive: if no snapshots match (e.g., backtesting an early gameweek),
        # availability comes back NaN; default to 100.
        df["availability"] = df["availability"].fillna(100)
        return df

    def predict_all(self, target_gw: int, as_of_gameweek: Optional[int] = None) -> pd.DataFrame:
        """
        Predict expected points for every player for the given gameweek.

        Args:
            target_gw: The gameweek to predict for.
            as_of_gameweek: The latest gameweek whose data the predictor may use.
                            Defaults to target_gw - 1 (i.e. predicting forward).
                            For backtesting, set this explicitly.
        """
        if as_of_gameweek is None:
            as_of_gameweek = target_gw - 1

        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")

        try:
            form_df = self._get_recent_form(conn, as_of_gameweek + 1)
            fdr_df = self._get_fixture_difficulty(conn, target_gw)
            meta_df = self._get_player_meta(conn, as_of_gameweek + 1)
        finally:
            conn.close()

        # Join: player meta + recent form + fixture difficulty
        df = meta_df.merge(form_df, on="player_id", how="left")
        df = df.merge(fdr_df, on="team_id", how="left")

        # Fill missing values defensively
        df["recent_form"] = df["recent_form"].fillna(0.0)
        df["avg_minutes"] = df["avg_minutes"].fillna(0.0)
        df["games_in_window"] = df["games_in_window"].fillna(0).astype(int)
        df["fdr"] = df["fdr"].fillna(3.0)  # neutral if no fixture
        df["num_fixtures"] = df["num_fixtures"].fillna(0).astype(int)
        df["availability"] = df["availability"].fillna(100)

        # Modifiers
        df["fixture_modifier"] = df["fdr"].apply(fdr_to_modifier)
        df["minutes_modifier"] = (df["avg_minutes"] / 90.0).clip(upper=1.0)

        # Sample-size penalty: trust drops if a player has < form_window qualifying games
        df["sample_size_modifier"] = (df["games_in_window"] / self.form_window).clip(upper=1.0)

        # Compute prediction
        df["predicted_points"] = (
            df["recent_form"]
            * df["fixture_modifier"]
            * df["minutes_modifier"]
            * df["sample_size_modifier"]
            * (df["availability"] / 100.0)
            * df["num_fixtures"].clip(lower=1)
        )

        # Players with no fixtures next gameweek -> 0 (blank gameweek)
        df.loc[df["num_fixtures"] == 0, "predicted_points"] = 0.0

        return df

    def write_predictions(self, target_gw: int, predictions_df: pd.DataFrame) -> int:
        """Write predictions to the predictions table. Returns number of rows written."""
        now_iso = datetime.now(timezone.utc).isoformat()
        rows = [
            (int(r.player_id), int(target_gw), self.MODEL_NAME, float(r.predicted_points), now_iso)
            for r in predictions_df.itertuples()
        ]
        conn = sqlite3.connect(self.db_path)
        try:
            conn.executemany(
                """
                INSERT INTO predictions (player_id, gameweek_id, model_name, predicted_points, prediction_time)
                VALUES (?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()
        finally:
            conn.close()
        return len(rows)