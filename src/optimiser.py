"""
optimiser.py
FPL squad optimisation layer. Takes per-player predicted points and FPL rules,
outputs the optimal 15-player squad and starting XI.

Uses integer linear programming via pulp.
"""

import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd
import pulp

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "fpl.db"


# ---------- FPL constants ----------
SQUAD_SIZE = 15
STARTING_XI = 11
BUDGET = 1000  # £100m in tenths
MAX_PER_CLUB = 3

# Eligibility thresholds — shared by optimise() and optimise_with_transfers()
MIN_QUALIFYING_GAMES_3 = 2
MIN_QUALIFYING_GAMES_5 = 2
MIN_CHANCE_OF_PLAYING = 75

# (position_id, count_in_squad, min_in_xi, max_in_xi)
POSITION_RULES = [
    (1, 2, 1, 1),  # GK
    (2, 5, 3, 5),  # DEF
    (3, 5, 2, 5),  # MID
    (4, 3, 1, 3),  # FWD
]


class SquadOptimiser:
    """Pick the optimal FPL squad and starting XI given player predictions."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path

    def _load_player_data(self) -> pd.DataFrame:
        """Load every player's current cost, position, and team."""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(
            """
            SELECT p.player_id, p.web_name, p.position, p.team_id, p.current_cost,
                   t.short_name AS team
            FROM players p
            JOIN teams t ON p.team_id = t.team_id
            """,
            conn,
        )
        conn.close()
        return df

    def optimise(self, predictions_df: pd.DataFrame) -> dict:
        """
        Args:
            predictions_df: must have columns 'player_id' and 'predicted_points'.

        Returns:
            dict with keys:
              squad: DataFrame of the 15 selected players
              starting_xi: DataFrame of the 11 starters
              bench: DataFrame of the 4 bench players
              captain: row of the captain (highest-predicted in XI)
              vice_captain: row of vice-captain (second-highest in XI)
              total_cost: int, sum of selected players' costs
              expected_points: float, predicted points of starting XI + 1x captain bonus
        """
        players = self._load_player_data()
        df = players.merge(
            predictions_df[[
                "player_id", "predicted_points",
                "qualifying_games_3", "qualifying_games_5", "chance_of_playing_next",
            ]],
            on="player_id", how="left",
        )
        df["predicted_points"] = df["predicted_points"].fillna(0.0)
        df["qualifying_games_3"] = df["qualifying_games_3"].fillna(0)
        df["qualifying_games_5"] = df["qualifying_games_5"].fillna(0)
        df["chance_of_playing_next"] = df["chance_of_playing_next"].fillna(0)

        eligible_mask = (
            (df["qualifying_games_3"] >= MIN_QUALIFYING_GAMES_3)
            & (df["qualifying_games_5"] >= MIN_QUALIFYING_GAMES_5)
            & (df["chance_of_playing_next"] >= MIN_CHANCE_OF_PLAYING)
        )
        n_eligible = int(eligible_mask.sum())
        n_total = len(df)
        print(f"  Eligible players after filters: {n_eligible} / {n_total}")

        # Sanity check: each position needs enough eligible players
        for pos_id, squad_count, _, _ in POSITION_RULES:
            n_in_pos = int(eligible_mask[df["position"] == pos_id].sum())
            if n_in_pos < squad_count:
                raise RuntimeError(
                    f"Only {n_in_pos} eligible players for position {pos_id}, "
                    f"need {squad_count}. Filters too strict."
                )

        df = df[eligible_mask].reset_index(drop=True)

        prob = pulp.LpProblem("FPL_Squad", pulp.LpMaximize)

        # Decision variables
        select = {
            row.player_id: pulp.LpVariable(f"select_{row.player_id}", cat="Binary")
            for row in df.itertuples()
        }
        start = {
            row.player_id: pulp.LpVariable(f"start_{row.player_id}", cat="Binary")
            for row in df.itertuples()
        }

        # Objective: starter points + soft bench preference.
        # 1.0x for starters; 0.1x for bench. Bench still matters, but starting XI dominates.
        BENCH_WEIGHT = 0.1
        prob += pulp.lpSum(
            row.predicted_points * start[row.player_id]
            + BENCH_WEIGHT * row.predicted_points * (select[row.player_id] - start[row.player_id])
            for row in df.itertuples()
        )

        # Constraint: squad size = 15
        prob += pulp.lpSum(select.values()) == SQUAD_SIZE, "squad_size"

        # Constraint: starting XI size = 11
        prob += pulp.lpSum(start.values()) == STARTING_XI, "xi_size"

        # Constraint: player can start only if selected
        for pid in select:
            prob += start[pid] <= select[pid], f"start_iff_select_{pid}"

        # Constraint: budget
        prob += pulp.lpSum(
            row.current_cost * select[row.player_id] for row in df.itertuples()
        ) <= BUDGET, "budget"

        # Constraints: position counts in squad and XI
        for pos_id, squad_count, xi_min, xi_max in POSITION_RULES:
            pos_players = df[df["position"] == pos_id]
            prob += pulp.lpSum(
                select[row.player_id] for row in pos_players.itertuples()
            ) == squad_count, f"squad_pos_{pos_id}"
            prob += pulp.lpSum(
                start[row.player_id] for row in pos_players.itertuples()
            ) >= xi_min, f"xi_min_pos_{pos_id}"
            prob += pulp.lpSum(
                start[row.player_id] for row in pos_players.itertuples()
            ) <= xi_max, f"xi_max_pos_{pos_id}"

        # Constraint: max 3 players per club
        for team_id in df["team_id"].unique():
            team_players = df[df["team_id"] == team_id]
            prob += pulp.lpSum(
                select[row.player_id] for row in team_players.itertuples()
            ) <= MAX_PER_CLUB, f"club_limit_{team_id}"

        # Solve
        solver = pulp.PULP_CBC_CMD(msg=False)
        status = prob.solve(solver)

        if pulp.LpStatus[status] != "Optimal":
            raise RuntimeError(f"Optimiser did not find optimal solution: {pulp.LpStatus[status]}")

        # Extract results
        df["selected"] = df["player_id"].map(lambda p: int(round(select[p].value())))
        df["starting"] = df["player_id"].map(lambda p: int(round(start[p].value())))

        squad = df[df["selected"] == 1].copy()
        xi = squad[squad["starting"] == 1].sort_values("predicted_points", ascending=False)
        bench = squad[squad["starting"] == 0].sort_values("predicted_points", ascending=False)

        # Captain & vice: highest two predicted in XI
        captain = xi.iloc[0]
        vice_captain = xi.iloc[1]

        total_cost = int(squad["current_cost"].sum())
        expected_points = float(xi["predicted_points"].sum() + captain["predicted_points"])

        return {
            "squad": squad,
            "starting_xi": xi,
            "bench": bench,
            "captain": captain,
            "vice_captain": vice_captain,
            "total_cost": total_cost,
            "expected_points": expected_points,

        }
    def optimise_with_transfers(
        self,
        predictions_df: pd.DataFrame,
        current_squad: list,
        free_transfers: int = 1,
        max_transfers: int = 3,
        transfer_hit_cost: int = 4,
    ) -> dict:
        """Given a current 15-player squad, find the optimal transfer move(s)."""
        if len(current_squad) != SQUAD_SIZE:
            raise ValueError(f"current_squad must have {SQUAD_SIZE} players, got {len(current_squad)}")

        players = self._load_player_data()
        df = players.merge(
            predictions_df[[
                "player_id", "predicted_points",
                "qualifying_games_3", "qualifying_games_5", "chance_of_playing_next",
            ]],
            on="player_id", how="left",
        )
        df["predicted_points"] = df["predicted_points"].fillna(0.0)
        df["qualifying_games_3"] = df["qualifying_games_3"].fillna(0)
        df["qualifying_games_5"] = df["qualifying_games_5"].fillna(0)
        df["chance_of_playing_next"] = df["chance_of_playing_next"].fillna(0)

        current_set = set(current_squad)

        eligible_mask = (
            (df["qualifying_games_3"] >= MIN_QUALIFYING_GAMES_3)
            & (df["qualifying_games_5"] >= MIN_QUALIFYING_GAMES_5)
            & (df["chance_of_playing_next"] >= MIN_CHANCE_OF_PLAYING)
        ) | df["player_id"].isin(current_set)
        df = df[eligible_mask].reset_index(drop=True)

        prob = pulp.LpProblem("FPL_Transfers", pulp.LpMaximize)

        select = {row.player_id: pulp.LpVariable(f"select_{row.player_id}", cat="Binary")
                  for row in df.itertuples()}
        start = {row.player_id: pulp.LpVariable(f"start_{row.player_id}", cat="Binary")
                 for row in df.itertuples()}
        hit = pulp.LpVariable("hit", lowBound=0, cat="Continuous")

        BENCH_WEIGHT = 0.1
        prob += (
            pulp.lpSum(
                row.predicted_points * start[row.player_id]
                + BENCH_WEIGHT * row.predicted_points * (select[row.player_id] - start[row.player_id])
                for row in df.itertuples()
            )
            - transfer_hit_cost * hit
        )

        prob += pulp.lpSum(select.values()) == SQUAD_SIZE
        prob += pulp.lpSum(start.values()) == STARTING_XI
        for pid in select:
            prob += start[pid] <= select[pid]
        prob += pulp.lpSum(
            row.current_cost * select[row.player_id] for row in df.itertuples()
        ) <= BUDGET

        for pos_id, squad_count, xi_min, xi_max in POSITION_RULES:
            pos_players = df[df["position"] == pos_id]
            prob += pulp.lpSum(select[r.player_id] for r in pos_players.itertuples()) == squad_count
            prob += pulp.lpSum(start[r.player_id] for r in pos_players.itertuples()) >= xi_min
            prob += pulp.lpSum(start[r.player_id] for r in pos_players.itertuples()) <= xi_max

        for team_id in df["team_id"].unique():
            team_players = df[df["team_id"] == team_id]
            prob += pulp.lpSum(select[r.player_id] for r in team_players.itertuples()) <= MAX_PER_CLUB

        new_player_selections = [select[r.player_id] for r in df.itertuples()
                                  if r.player_id not in current_set]
        num_transfers_var = pulp.lpSum(new_player_selections)
        prob += num_transfers_var <= max_transfers
        prob += hit >= num_transfers_var - free_transfers

        solver = pulp.PULP_CBC_CMD(msg=False)
        status = prob.solve(solver)
        if pulp.LpStatus[status] != "Optimal":
            raise RuntimeError(f"Transfer optimiser failed: {pulp.LpStatus[status]}")

        df["selected"] = df["player_id"].map(lambda p: int(round(select[p].value())))
        df["starting"] = df["player_id"].map(lambda p: int(round(start[p].value())))

        squad = df[df["selected"] == 1].copy()
        new_set = set(squad["player_id"])
        transfers_in_ids = list(new_set - current_set)
        transfers_out_ids = list(current_set - new_set)
        num_transfers = len(transfers_in_ids)
        hit_value = max(0, num_transfers - free_transfers)
        hit_points = hit_value * transfer_hit_cost

        xi = squad[squad["starting"] == 1].sort_values("predicted_points", ascending=False)
        bench = squad[squad["starting"] == 0].sort_values("predicted_points", ascending=False)
        captain = xi.iloc[0]
        vice_captain = xi.iloc[1]

        all_players = self._load_player_data()
        transfers_in = all_players[all_players["player_id"].isin(transfers_in_ids)].copy()
        transfers_out = all_players[all_players["player_id"].isin(transfers_out_ids)].copy()
        transfers_in = transfers_in.merge(
            predictions_df[["player_id", "predicted_points"]], on="player_id", how="left"
        )
        transfers_out = transfers_out.merge(
            predictions_df[["player_id", "predicted_points"]], on="player_id", how="left"
        )

        total_cost = int(squad["current_cost"].sum())
        gross_xi_points = float(xi["predicted_points"].sum() + captain["predicted_points"])
        net_expected_points = gross_xi_points - hit_points

        return {
            "squad": squad,
            "starting_xi": xi,
            "bench": bench,
            "captain": captain,
            "vice_captain": vice_captain,
            "total_cost": total_cost,
            "gross_xi_points": gross_xi_points,
            "hit_points": hit_points,
            "expected_points": net_expected_points,
            "num_transfers": num_transfers,
            "transfers_in": transfers_in,
            "transfers_out": transfers_out,
        }