"""
analytics.py
Captain picks and differentials for World Cup Fantasy.

All functions accept a predictions_df from WCPredictor.predict_matchday()
and return a DataFrame ready for printing.
"""

import sqlite3
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH      = PROJECT_ROOT / "data" / "wc.db"


def _team_names() -> dict:
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT team_id, short_name, name FROM teams").fetchall()
    conn.close()
    return {r[0]: (r[1] or r[2][:4]) for r in rows}


def _eligible(df: pd.DataFrame) -> pd.DataFrame:
    """Players with a fixture and reasonable minute expectation."""
    return df[(df["minutes_roll"] >= 30) | (df["pos_enc"] == 0)]  # GKs always eligible


def captain_picks(
    predictions_df: pd.DataFrame,
    top_n: int = 7,
    squad_ids: list | None = None,
) -> pd.DataFrame:
    """
    Top N captain candidates ranked by predicted points.

    If squad_ids is provided, marks players in your squad with *.

    Returns columns: #, Player, Team, Pos, Price, pred, form_xg, own%
    """
    team_names = _team_names()
    df = predictions_df.copy()

    pos_map = {0: "GK", 1: "DEF", 2: "MID", 3: "FWD"}
    df["Pos"]   = df["pos_enc"].map(pos_map).fillna("MID")
    df["Team"]  = df["team_id"].map(team_names).fillna("?")
    df["Price"] = (df["price"] / 10).round(1)

    eligible = _eligible(df)
    result = (
        eligible
        .sort_values("predicted_pts", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    result.insert(0, "#", range(1, len(result) + 1))

    if squad_ids:
        result["in_squad"] = result["fantasy_id"].isin(squad_ids)
        result["Player"]   = result.apply(
            lambda r: f"{r['name']} *" if r["in_squad"] else r["name"], axis=1
        )
    else:
        result["Player"] = result["name"]

    cols = ["#", "Player", "Team", "Pos", "Price",
            "predicted_pts", "xg_roll", "ownership"]
    out = result[[c for c in cols if c in result.columns]].copy()
    out = out.rename(columns={
        "predicted_pts": "pred", "xg_roll": "form_xG", "ownership": "own%"
    })
    out["pred"]    = out["pred"].round(2)
    out["form_xG"] = out.get("form_xG", pd.Series(0.0)).round(2)
    out["own%"]    = out.get("own%", pd.Series(0.0)).round(1)
    return out


def find_differentials(
    predictions_df: pd.DataFrame,
    top_n: int = 10,
    max_ownership: float = 15.0,
    min_predicted_pts: float = 3.0,
) -> pd.DataFrame:
    """
    Low-ownership, high-expected-output players for this matchday.

    Returns columns: #, Player, Team, Pos, Price, pred, form_xG, own%
    """
    team_names = _team_names()
    df = predictions_df.copy()

    pos_map = {0: "GK", 1: "DEF", 2: "MID", 3: "FWD"}
    df["Pos"]  = df["pos_enc"].map(pos_map).fillna("MID")
    df["Team"] = df["team_id"].map(team_names).fillna("?")
    df["Price"] = (df["price"] / 10).round(1)
    df["own%"]  = df.get("ownership", pd.Series(0.0)).fillna(0.0)

    mask = (
        (df["own%"] <= max_ownership)
        & (df["predicted_pts"] >= min_predicted_pts)
    )
    eligible = _eligible(df[mask])
    result = (
        eligible
        .sort_values("predicted_pts", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    if result.empty:
        return result

    result.insert(0, "#", range(1, len(result) + 1))
    result["Player"] = result["name"]

    cols = ["#", "Player", "Team", "Pos", "Price",
            "predicted_pts", "xg_roll", "own%"]
    out = result[[c for c in cols if c in result.columns]].copy()
    out = out.rename(columns={"predicted_pts": "pred", "xg_roll": "form_xG"})
    out["pred"]    = out["pred"].round(2)
    out["form_xG"] = out.get("form_xG", pd.Series(0.0)).round(2)
    out["own%"]    = out["own%"].round(1)
    return out


def squad_optimiser(
    predictions_df: pd.DataFrame,
    budget: int = 1000,
    formation: tuple = (1, 4, 3, 3),
) -> pd.DataFrame | None:
    """
    Group Stage Squad Optimiser (Stage 3 feature).

    Picks the best XI within budget using ILP.
    formation: (GK, DEF, MID, FWD) — default 4-3-3 (1,4,3,3)

    Returns None if pulp is not installed.
    """
    try:
        import pulp
    except ImportError:
        print("[warn] pulp not installed — squad optimiser disabled. pip install pulp")
        return None

    df = predictions_df[predictions_df["minutes_roll"] >= 20].copy()
    if df.empty:
        return None

    pos_map  = {0: "GK", 1: "DEF", 2: "MID", 3: "FWD"}
    gk_n, def_n, mid_n, fwd_n = formation
    pos_counts = {"GK": gk_n, "DEF": def_n, "MID": mid_n, "FWD": fwd_n}

    df["_pos"] = df["pos_enc"].map(pos_map).fillna("MID")
    df = df.reset_index(drop=True)

    prob   = pulp.LpProblem("WC_XI", pulp.LpMaximize)
    select = {i: pulp.LpVariable(f"s_{i}", cat="Binary") for i in df.index}
    cap    = {i: pulp.LpVariable(f"c_{i}", cat="Binary") for i in df.index}

    prob += pulp.lpSum(
        df.loc[i, "predicted_pts"] * select[i] + df.loc[i, "predicted_pts"] * cap[i]
        for i in df.index
    )
    prob += pulp.lpSum(select.values()) == 11
    prob += pulp.lpSum(cap.values()) == 1
    for i in df.index:
        prob += cap[i] <= select[i]
    prob += pulp.lpSum(df.loc[i, "price"] * select[i] for i in df.index) <= budget

    for pos, count in pos_counts.items():
        pos_players = df[df["_pos"] == pos].index
        prob += pulp.lpSum(select[i] for i in pos_players) == count

    solver = pulp.PULP_CBC_CMD(msg=False)
    if pulp.LpStatus[prob.solve(solver)] != "Optimal":
        return None

    df["selected"]   = [int(round(select[i].value())) for i in df.index]
    df["is_captain"] = [int(round(cap[i].value()))    for i in df.index]

    squad = df[df["selected"] == 1].copy()
    cap_row  = squad[squad["is_captain"] == 1].iloc[0]
    vcap_row = squad[(squad["is_captain"] == 0)].sort_values("predicted_pts", ascending=False).iloc[0]

    print(f"  Expected points (incl. captain bonus): "
          f"{squad['predicted_pts'].sum() + cap_row['predicted_pts']:.2f}")
    print(f"  Total cost: {squad['price'].sum() / 10:.1f}")
    print(f"  Captain: {cap_row['name']}")
    print(f"  VC:      {vcap_row['name']}")
    return squad
