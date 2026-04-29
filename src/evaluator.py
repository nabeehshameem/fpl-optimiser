"""
evaluator.py
Backtesting harness for FPL predictors.

Given a predictor and a list of past gameweeks, this:
  1. Runs the predictor "as of" each gameweek's deadline
  2. Compares predictions to actual outcomes from player_gameweek_history
  3. Also benchmarks against FPL's ep_next baseline (read from snapshots)
  4. Computes MAE and RMSE per gameweek and overall
"""

import sqlite3
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from src.predictor import NaivePredictor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "fpl.db"


# ---------- Loaders for actuals and FPL baseline ----------

def get_actuals_for_gameweek(conn: sqlite3.Connection, gameweek: int) -> pd.DataFrame:
    """
    Return one row per player who appeared in this gameweek, with their actual points.
    Players who did not appear (no row in history) are excluded here; the harness
    will treat their predictions as 'predicted but never played' separately.
    """
    query = """
        SELECT player_id, total_points AS actual_points, minutes
        FROM player_gameweek_history
        WHERE gameweek_id = ?
    """
    return pd.read_sql_query(query, conn, params=(gameweek,))


def get_fpl_ep_next_for_gameweek(conn: sqlite3.Connection, gameweek: int) -> pd.DataFrame:
    """
    Return the most recent FPL ep_next snapshot for each player taken at or before
    the deadline of `gameweek`. This is FPL's own prediction baseline.
    """
    query = """
        SELECT s.player_id, s.ep_next AS fpl_ep_next
        FROM player_snapshots s
        WHERE s.snapshot_id IN (
            SELECT MAX(snapshot_id)
            FROM player_snapshots
            WHERE gameweek_id = ?
            GROUP BY player_id
        )
    """
    return pd.read_sql_query(query, conn, params=(gameweek,))


# ---------- Metrics ----------

def mean_absolute_error(predicted: pd.Series, actual: pd.Series) -> float:
    return float(np.mean(np.abs(predicted - actual)))


def root_mean_squared_error(predicted: pd.Series, actual: pd.Series) -> float:
    return float(np.sqrt(np.mean((predicted - actual) ** 2)))


# ---------- Evaluator ----------

class Evaluator:
    """Backtests one or more predictors across a list of historical gameweeks."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path

    def evaluate_gameweek(
        self,
        predictor,
        target_gw: int,
        restrict_to_appeared: bool = True,
    ) -> pd.DataFrame:
        """
        Run predictor for target_gw using only data up to target_gw - 1,
        then join with actuals and FPL's baseline.

        If restrict_to_appeared is True, only evaluates players who actually
        appeared in target_gw. This is a deliberate choice (see notes below).
        """
        # Predictor sees only data strictly before target_gw
        preds = predictor.predict_all(target_gw=target_gw, as_of_gameweek=target_gw - 1)
        preds = preds[["player_id", "predicted_points"]].rename(
            columns={"predicted_points": "model_pred"}
        )

        conn = sqlite3.connect(self.db_path)
        try:
            actuals = get_actuals_for_gameweek(conn, target_gw)
            fpl_baseline = get_fpl_ep_next_for_gameweek(conn, target_gw)
        finally:
            conn.close()

        if restrict_to_appeared:
            df = actuals.merge(preds, on="player_id", how="left")
            df = df.merge(fpl_baseline, on="player_id", how="left")
        else:
            df = preds.merge(actuals, on="player_id", how="left")
            df = df.merge(fpl_baseline, on="player_id", how="left")
            df["actual_points"] = df["actual_points"].fillna(0)

        # Defensive fills
        df["model_pred"] = df["model_pred"].fillna(0)
        df["fpl_ep_next"] = df["fpl_ep_next"].fillna(0)
        df["gameweek"] = target_gw

        return df

    def evaluate_many(
        self,
        predictor,
        gameweeks: List[int],
        restrict_to_appeared: bool = True,
    ) -> pd.DataFrame:
        """Evaluate predictor across multiple gameweeks; concatenate results."""
        per_gw = [
            self.evaluate_gameweek(predictor, gw, restrict_to_appeared=restrict_to_appeared)
            for gw in gameweeks
        ]
        return pd.concat(per_gw, ignore_index=True)

    @staticmethod
    def summarise(df: pd.DataFrame) -> pd.DataFrame:
        """
        Produce per-gameweek and overall MAE/RMSE for both the model and FPL baseline.
        """
        rows = []
        for gw, group in df.groupby("gameweek"):
            rows.append({
                "gameweek": int(gw),
                "n_players": len(group),
                "model_mae": mean_absolute_error(group["model_pred"], group["actual_points"]),
                "model_rmse": root_mean_squared_error(group["model_pred"], group["actual_points"]),
                "fpl_mae": mean_absolute_error(group["fpl_ep_next"], group["actual_points"]),
                "fpl_rmse": root_mean_squared_error(group["fpl_ep_next"], group["actual_points"]),
                "zero_mae": float(group["actual_points"].abs().mean()),
            })
        per_gw_df = pd.DataFrame(rows).sort_values("gameweek")

        overall_row = {
            "gameweek": "OVERALL",
            "n_players": len(df),
            "model_mae": mean_absolute_error(df["model_pred"], df["actual_points"]),
            "model_rmse": root_mean_squared_error(df["model_pred"], df["actual_points"]),
            "fpl_mae": mean_absolute_error(df["fpl_ep_next"], df["actual_points"]),
            "fpl_rmse": root_mean_squared_error(df["fpl_ep_next"], df["actual_points"]),
            "zero_mae": float(group["actual_points"].abs().mean()),
        }
        return pd.concat([per_gw_df, pd.DataFrame([overall_row])], ignore_index=True)