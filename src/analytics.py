"""
analytics.py
Post-prediction analysis tools: differentials, fixture ticker, captain picks,
player comparison, price change predictions, bonus point predictions.

All public functions accept a predictions_df produced by LightGBMPredictor.predict_all()
and return a plain DataFrame ready for printing or downstream use.
"""

import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "fpl.db"

POSITION_NAMES = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
FDR_SYMBOLS = {1: "1", 2: "2", 3: "3", 4: "4", 5: "5"}


# ---------- Private DB helpers ----------

def _player_info() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        """
        SELECT p.player_id, p.web_name, p.position, p.current_cost,
               t.short_name AS team, p.team_id
        FROM players p
        JOIN teams t ON p.team_id = t.team_id
        """,
        conn,
    )
    conn.close()
    return df


def _latest_snapshots() -> pd.DataFrame:
    """Most recent snapshot row per player."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        """
        SELECT player_id, selected_by_percent, now_cost
        FROM player_snapshots
        WHERE snapshot_id IN (
            SELECT MAX(snapshot_id)
            FROM player_snapshots
            GROUP BY player_id
        )
        """,
        conn,
    )
    conn.close()
    df["selected_by_percent"] = pd.to_numeric(df["selected_by_percent"], errors="coerce").fillna(0.0)
    return df


def _enrich(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Merge web_name, team, position (overwrite if stale) and ownership onto predictions."""
    info = _player_info()
    snaps = _latest_snapshots()
    df = predictions_df.copy()

    # Drop stale position/team cols that came through features pipeline so we
    # use the current values from the players table.
    for col in ("web_name", "team"):
        if col in df.columns:
            df = df.drop(columns=[col])

    df = df.merge(info[["player_id", "web_name", "team", "position", "current_cost", "team_id"]],
                  on="player_id", how="left", suffixes=("_feat", ""))
    # Resolve any duplicate cost/position columns left by the merge
    for col in ("position", "current_cost", "team_id"):
        feat_col = f"{col}_feat"
        if feat_col in df.columns:
            df[col] = df[col].fillna(df[feat_col])
            df.drop(columns=[feat_col], inplace=True)

    df = df.merge(snaps[["player_id", "selected_by_percent"]], on="player_id", how="left")
    df["selected_by_percent"] = df["selected_by_percent"].fillna(0.0)
    return df


def _eligible(df: pd.DataFrame) -> pd.DataFrame:
    """Players with fixtures, recent form, and availability."""
    return df[
        (df["num_fixtures"] > 0)
        & (df["qualifying_games_3"] >= 2)
        & (df["chance_of_playing_next"] >= 75)
    ]


# ---------- 1. Differentials ----------

def find_differentials(
    predictions_df: pd.DataFrame,
    top_n: int = 10,
    max_ownership: float = 15.0,
    min_predicted_pts: float = 3.5,
) -> pd.DataFrame:
    """
    High-upside, low-ownership players — the picks that separate managers.

    Returns top_n eligible players with ownership <= max_ownership sorted by
    predicted points descending.
    """
    df = _enrich(predictions_df)
    mask = (
        (df["selected_by_percent"] <= max_ownership)
        & (df["predicted_points"] >= min_predicted_pts)
    )
    result = (
        _eligible(df[mask])
        .sort_values("predicted_points", ascending=False)
        .head(top_n)
        [["web_name", "team", "position", "current_cost",
          "predicted_points", "total_points_mean_3",
          "selected_by_percent", "fdr", "num_fixtures"]]
        .copy()
        .reset_index(drop=True)
    )
    result["position"] = result["position"].map(POSITION_NAMES)
    result["£"] = (result.pop("current_cost") / 10).round(1)
    result["pred"] = result.pop("predicted_points").round(2)
    result["form3"] = result.pop("total_points_mean_3").round(2)
    result["own%"] = result.pop("selected_by_percent").round(1)
    result.index += 1
    return result


# ---------- 2. Fixture Ticker ----------

def fixture_ticker(target_gw: int, num_gws: int = 5) -> pd.DataFrame:
    """
    FDR colour-map for every team over the next num_gws gameweeks.

    Each cell shows H(fdr) or A(fdr) for home/away.
    DGW teams show both fixtures separated by ' + '.
    BGW teams show ' - '.
    """
    conn = sqlite3.connect(DB_PATH)
    fixtures = pd.read_sql_query(
        """
        SELECT gameweek_id, home_team_id AS team_id,
               home_team_difficulty AS fdr, 1 AS is_home
        FROM fixtures WHERE gameweek_id IS NOT NULL
        UNION ALL
        SELECT gameweek_id, away_team_id AS team_id,
               away_team_difficulty AS fdr, 0 AS is_home
        FROM fixtures WHERE gameweek_id IS NOT NULL
        """,
        conn,
    )
    teams = pd.read_sql_query(
        "SELECT team_id, short_name FROM teams ORDER BY short_name", conn
    )
    conn.close()

    gw_range = list(range(target_gw, target_gw + num_gws))
    fixtures = fixtures[fixtures["gameweek_id"].isin(gw_range)]

    # Build a dict: (team_id, gw) -> list of "H(fdr)" strings
    ticker: dict = {}
    for _, row in fixtures.iterrows():
        key = (int(row["team_id"]), int(row["gameweek_id"]))
        tag = f"{'H' if row['is_home'] else 'A'}({int(row['fdr'])})"
        ticker.setdefault(key, []).append(tag)

    rows = []
    for _, team in teams.iterrows():
        r = {"Team": team["short_name"]}
        for gw in gw_range:
            entries = ticker.get((int(team["team_id"]), gw), [])
            r[f"GW{gw}"] = " + ".join(entries) if entries else " - "
        rows.append(r)

    return pd.DataFrame(rows).reset_index(drop=True)


# ---------- 3. Captain Picks ----------

def captain_picks(predictions_df: pd.DataFrame, top_n: int = 7) -> pd.DataFrame:
    """
    Top captaincy candidates ranked by predicted points, with ownership and form context.

    The ILP already picks the captain optimally; this table shows the runner-up
    options so you can make an informed alternative choice.
    """
    df = _enrich(predictions_df)
    eligible = _eligible(df).copy()

    result = (
        eligible
        .sort_values("predicted_points", ascending=False)
        .head(top_n)
        [["web_name", "team", "position", "current_cost",
          "predicted_points", "total_points_mean_3", "total_points_mean_5",
          "fdr", "num_fixtures", "selected_by_percent"]]
        .copy()
        .reset_index(drop=True)
    )
    result.insert(0, "#", range(1, len(result) + 1))
    result["position"] = result["position"].map(POSITION_NAMES)
    result["£"] = (result.pop("current_cost") / 10).round(1)
    result["pred"] = result.pop("predicted_points").round(2)
    result["form3"] = result.pop("total_points_mean_3").round(2)
    result["form5"] = result.pop("total_points_mean_5").round(2)
    result["own%"] = result.pop("selected_by_percent").round(1)
    return result


# ---------- 4. Player Comparison ----------

def compare_players(player_ids: list, predictions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Side-by-side stat comparison for the given player_ids.

    Useful before making a transfer decision: compare the player you're
    selling against candidates you're buying.
    """
    df = _enrich(predictions_df)
    subset = df[df["player_id"].isin(player_ids)].copy()
    if subset.empty:
        raise ValueError(f"No data found for player_ids {player_ids}")

    wanted = [
        "web_name", "team", "position", "current_cost",
        "predicted_points",
        "total_points_mean_3", "total_points_mean_5",
        "expected_goals_mean_3", "expected_assists_mean_3",
        "minutes_mean_3",
        "fdr", "num_fixtures",
        "selected_by_percent", "chance_of_playing_next",
    ]
    cols = [c for c in wanted if c in subset.columns]
    result = subset[cols].copy().reset_index(drop=True)

    result["position"] = result["position"].map(POSITION_NAMES)
    result["current_cost"] = (result["current_cost"] / 10).round(1)
    for col in ["predicted_points", "total_points_mean_3", "total_points_mean_5",
                "expected_goals_mean_3", "expected_assists_mean_3",
                "minutes_mean_3", "selected_by_percent"]:
        if col in result.columns:
            result[col] = result[col].round(2)

    # Transpose so each row is a stat and each column is a player
    result = result.set_index("web_name").T
    return result


# ---------- 5. Price Change Predictions ----------

def price_change_predictions(top_n: int = 20) -> pd.DataFrame:
    """
    Players most likely to change price based on ownership trajectory.

    FPL price rises when a player's net transfer balance crosses ~1% of all
    managers. We proxy this with the delta in selected_by_percent across the
    two most-recent snapshots. Large positive delta → price rise imminent.
    Large negative delta → price fall.
    """
    conn = sqlite3.connect(DB_PATH)
    # Two most recent snapshots per player using a window function
    all_snaps = pd.read_sql_query(
        """
        SELECT player_id, gameweek_id, selected_by_percent, now_cost,
               ROW_NUMBER() OVER (
                   PARTITION BY player_id ORDER BY snapshot_id DESC
               ) AS rn
        FROM player_snapshots
        """,
        conn,
    )
    players = pd.read_sql_query(
        """
        SELECT p.player_id, p.web_name, p.position, p.current_cost,
               t.short_name AS team
        FROM players p
        JOIN teams t ON p.team_id = t.team_id
        """,
        conn,
    )
    conn.close()

    all_snaps["selected_by_percent"] = pd.to_numeric(
        all_snaps["selected_by_percent"], errors="coerce"
    ).fillna(0.0)

    latest = all_snaps[all_snaps["rn"] == 1][["player_id", "selected_by_percent", "now_cost"]]
    prev = all_snaps[all_snaps["rn"] == 2][["player_id", "selected_by_percent"]].rename(
        columns={"selected_by_percent": "prev_ownership"}
    )

    df = latest.merge(prev, on="player_id", how="left")
    df["prev_ownership"] = df["prev_ownership"].fillna(df["selected_by_percent"])
    df["delta"] = df["selected_by_percent"] - df["prev_ownership"]
    df = df.merge(players, on="player_id", how="left")

    df["direction"] = "stable"
    df.loc[df["delta"] > 0.25, "direction"] = "RISE ^"
    df.loc[df["delta"] < -0.25, "direction"] = "FALL v"

    per_side = max(1, top_n // 2)
    rising = df[df["delta"] > 0.25].nlargest(per_side, "delta")
    falling = df[df["delta"] < -0.25].nsmallest(per_side, "delta")
    result = pd.concat([rising, falling]).reset_index(drop=True)

    result["position"] = result["position"].map(POSITION_NAMES)
    result["£"] = (result["current_cost"] / 10).round(1)
    result["own%"] = result["selected_by_percent"].round(1)
    result["d_own"] = result["delta"].round(2)

    return (
        result[["web_name", "team", "position", "£", "own%", "d_own", "direction"]]
        .rename(columns={"web_name": "Player", "team": "Team",
                         "position": "Pos", "d_own": "d_own%", "direction": "Signal"})
        .reset_index(drop=True)
    )


# ---------- 6. Bonus Point Predictions ----------

def bonus_point_predictions(
    predictions_df: pd.DataFrame,
    top_n: int = 15,
) -> pd.DataFrame:
    """
    Players most likely to accumulate bonus points, based on historical BPS
    rates weighted by predicted playing time.

    BPS (bonus point system) correlates with goals, assists, clean sheets,
    saves, tackles, and clearances. Players with a high avg BPS per appearance
    and a fixture this GW are the targets.
    """
    conn = sqlite3.connect(DB_PATH)
    history = pd.read_sql_query(
        """
        SELECT player_id, bps, bonus, minutes
        FROM player_gameweek_history
        WHERE minutes >= 45
        ORDER BY player_id, gameweek_id DESC
        """,
        conn,
    )
    players = pd.read_sql_query(
        """
        SELECT p.player_id, p.web_name, p.position, p.current_cost,
               t.short_name AS team
        FROM players p
        JOIN teams t ON p.team_id = t.team_id
        """,
        conn,
    )
    conn.close()

    # Rolling stats over last 5 qualifying appearances
    recent = (
        history.groupby("player_id")
        .head(5)
        .groupby("player_id")
        .agg(avg_bps=("bps", "mean"), avg_bonus=("bonus", "mean"), n=("bps", "count"))
        .reset_index()
    )

    pred_cols = ["player_id", "predicted_points", "num_fixtures",
                 "minutes_mean_3", "qualifying_games_3", "chance_of_playing_next"]
    pred = predictions_df[[c for c in pred_cols if c in predictions_df.columns]].copy()

    df = pred.merge(recent, on="player_id", how="left")
    df = df.merge(players, on="player_id", how="left")
    df["avg_bps"] = df["avg_bps"].fillna(0.0)
    df["avg_bonus"] = df["avg_bonus"].fillna(0.0)

    eligible = df[
        (df["num_fixtures"] > 0)
        & (df["qualifying_games_3"] >= 2)
        & (df["chance_of_playing_next"] >= 75)
        & (df["avg_bps"] > 0)
    ]

    result = (
        eligible
        .sort_values("avg_bps", ascending=False)
        .head(top_n)
        [["web_name", "team", "position", "current_cost",
          "predicted_points", "avg_bps", "avg_bonus", "num_fixtures"]]
        .copy()
        .reset_index(drop=True)
    )
    result["position"] = result["position"].map(POSITION_NAMES)
    result["£"] = (result.pop("current_cost") / 10).round(1)
    result["pred"] = result.pop("predicted_points").round(2)
    result["avg_bps"] = result["avg_bps"].round(1)
    result["avg_bonus"] = result["avg_bonus"].round(2)
    result.index += 1
    return result
