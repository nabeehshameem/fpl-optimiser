"""
Dixon-Coles projection predictor behind the standard predictor interface.

Extracted from fpl/src/fpl_optimizer.py (now deleted): the projection logic —
per-player historical rates scaled by DC fixture-difficulty adjustments and
converted through FPL scoring rules — is the genuinely new idea that module
contained. Its duplicate MILP optimiser was not, so squad optimisation stays
in src/optimiser.py.

Interface contract (same as NaivePredictor / LightGBMPredictor):
    predict_all(target_gw, as_of_gameweek=None) -> DataFrame
        including player_id, predicted_points, and the eligibility columns
        the optimiser filters on (qualifying_games_3/5, chance_of_playing_next).
    write_predictions(target_gw, df) -> int

The eligibility/meta columns come from the SAME build_prediction_features()
pipeline the LightGBM predictor uses, so a bake-off through run_evaluation.py
compares point-prediction quality under an identical eligibility regime —
never two predictors with silently different player pools.

Changes from the source module, both deliberate:
  * FIXED double fixture adjustment: exp goals conceded previously multiplied
    xga_adj into _adjusted_xga and then again into the concede penalty, so
    GK/DEF were over-penalised in hard fixtures inconsistently with their
    clean-sheet reward (which applied it once). Now applied exactly once.
  * Removed dead SQL in team season averages (first query was discarded).

Note on naming: "xG" in team averages is actual goals from the fixtures
table used as an xG proxy, consistent with what fpl_dc_params.json was fit on.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.features import build_prediction_features
from src.cold_start import (
    COLD_START_MIN_GWS, fit_positional_priors, is_cold_start, prior_for,
    resolve_archive,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "fpl.db"
MODEL_PATH = PROJECT_ROOT / "models" / "fpl_dc_params.json"

# ── FPL scoring constants (2025/26 rules) ────────────────────────────────────
PT_APP60 = 2.0
PT_APP_SUB = 1.0
PT_CS = {"GK": 4.0, "DEF": 4.0, "MID": 1.0, "FWD": 0.0}
PT_GOAL = {"GK": 10.0, "DEF": 6.0, "MID": 5.0, "FWD": 4.0}
PT_ASSIST = 3.0
PT_SAVE3 = 1.0
PT_CONCEDE = -1.0  # per 2 goals conceded (GK/DEF)

POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
POSITION_IDS = {v: k for k, v in POSITION_MAP.items()}
MIN_APP_MINUTES = 1


class DCPredictor:
    """Project FPL points from historical rates x DC fixture adjustments."""

    MODEL_NAME = "dc_projection_v1"

    def __init__(self, db_path: Path = DB_PATH, model_path: Path = MODEL_PATH,
                 archive_db_path: Path | None = None):
        self.db_path = db_path
        self.model_path = model_path
        # Only consulted while the live DB holds < COLD_START_MIN_GWS
        # gameweeks of history; auto-resolved from data/fpl_*.db if None.
        self.archive_db_path = archive_db_path

    # ── public interface ─────────────────────────────────────────────────────

    def predict_all(self, target_gw: int,
                    as_of_gameweek: Optional[int] = None) -> pd.DataFrame:
        """Same signature and output schema as the other predictors.

        as_of_gameweek is accepted for interface compatibility; rates are
        computed from player_gameweek_history, which the evaluation harness
        restricts to pre-target data by construction.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cold = is_cold_start(conn)
        finally:
            conn.close()

        df = (self._cold_start_frame(target_gw) if cold
              else build_prediction_features(target_gw=target_gw))
        projections = self._project(target_gw)
        df["predicted_points"] = df["player_id"].map(projections).fillna(0.0)
        if "num_fixtures" in df.columns:
            df.loc[df["num_fixtures"] == 0, "predicted_points"] = 0.0
        return df

    def write_predictions(self, target_gw: int,
                          predictions_df: pd.DataFrame) -> int:
        now_iso = datetime.now(timezone.utc).isoformat()
        rows = [
            (int(r.player_id), int(target_gw), self.MODEL_NAME,
             float(r.predicted_points), now_iso)
            for r in predictions_df.itertuples()
        ]
        conn = sqlite3.connect(self.db_path)
        try:
            conn.executemany(
                "INSERT INTO predictions (player_id, gameweek_id, model_name, "
                "predicted_points, prediction_time) VALUES (?, ?, ?, ?, ?)",
                rows,
            )
            conn.commit()
        finally:
            conn.close()
        return len(rows)

    def _cold_start_frame(self, target_gw: int) -> pd.DataFrame:
        """Minimal frame with the required schema, built without form history.

        Eligibility gates are opened: with no current-season appearances,
        qualifying-games filters would exclude every player. This mirrors the
        gate lock_model_squad.py already applies when max_hist_gw < 3.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            players = pd.read_sql_query(
                "SELECT player_id, team_id FROM players "
                "WHERE position IS NOT NULL", conn)
            fixtures = pd.read_sql_query(
                "SELECT home_team_id, away_team_id FROM fixtures "
                "WHERE gameweek_id = ?", conn, params=(target_gw,))
            try:
                snaps = pd.read_sql_query(
                    "SELECT player_id, chance_of_playing_next FROM ("
                    "  SELECT player_id, chance_of_playing_next,"
                    "         ROW_NUMBER() OVER (PARTITION BY player_id"
                    "           ORDER BY gameweek_id DESC) rn"
                    "  FROM player_snapshots) WHERE rn = 1", conn)
            except Exception:
                snaps = pd.DataFrame(columns=["player_id",
                                              "chance_of_playing_next"])
        finally:
            conn.close()

        playing = set(fixtures["home_team_id"]) | set(fixtures["away_team_id"])
        df = players.copy()
        df["num_fixtures"] = df["team_id"].isin(playing).astype(int)
        df["qualifying_games_3"] = COLD_START_MIN_GWS
        df["qualifying_games_5"] = 5
        if not snaps.empty:
            df = df.merge(snaps, on="player_id", how="left")
        else:
            df["chance_of_playing_next"] = 100
        df["chance_of_playing_next"] = df["chance_of_playing_next"].fillna(100)
        return df

    # ── projection internals (ported) ────────────────────────────────────────

    def _load_dc(self) -> dict:
        if not self.model_path.exists():
            return {}
        return json.loads(self.model_path.read_text())

    def _player_positions(self, conn) -> dict[int, str]:
        rows = conn.execute(
            "SELECT player_id, position, team_id FROM players "
            "WHERE position IS NOT NULL"
        ).fetchall()
        return {int(pid): (POSITION_MAP.get(pos, "MID"), int(tid))
                for pid, pos, tid in rows}

    def _compute_player_rates(self, conn, min_gws: int = 3) -> dict[int, dict]:
        rows = conn.execute("""
            SELECT player_id,
                   AVG(minutes)      AS avg_min,
                   AVG(goals_scored) AS avg_goals,
                   AVG(assists)      AS avg_assists,
                   AVG(CASE WHEN clean_sheets >= 1 AND minutes >= 60  -- >=: DGW rows sum to 2
                            THEN 1.0 ELSE 0.0 END) AS cs_rate,
                   AVG(bonus)            AS avg_bonus,
                   COUNT(*)              AS gw_count,
                   AVG(expected_goals)   AS avg_xg,
                   AVG(expected_assists) AS avg_xa
            FROM   player_gameweek_history
            WHERE  minutes >= ?
            GROUP  BY player_id
            HAVING COUNT(*) >= ?
        """, (MIN_APP_MINUTES, min_gws)).fetchall()
        out: dict[int, dict] = {}
        for pid, avg_min, avg_goals, avg_assists, cs_rate, avg_bonus, n, xg, xa in rows:
            out[int(pid)] = {
                "avg_minutes": float(avg_min or 0),
                "avg_goals": float(avg_goals or 0),
                "avg_assists": float(avg_assists or 0),
                "cs_rate": float(cs_rate or 0),
                "avg_bonus": float(avg_bonus or 0),
                "gw_count": int(n),
                "avg_xg": float(xg or 0),
                "avg_xa": float(xa or 0),
            }
        return out

    def _compute_team_season_avgs(self, conn) -> dict[int, dict]:
        rows = conn.execute("""
            SELECT team_id,
                   AVG(xg_for)     AS avg_xg_for,
                   AVG(xg_against) AS avg_xg_against
            FROM (
                SELECT home_team_id AS team_id,
                       CAST(home_score AS REAL) AS xg_for,
                       CAST(away_score AS REAL) AS xg_against
                FROM fixtures WHERE finished = 1
                UNION ALL
                SELECT away_team_id,
                       CAST(away_score AS REAL),
                       CAST(home_score AS REAL)
                FROM fixtures WHERE finished = 1
            )
            GROUP BY team_id
        """).fetchall()
        return {int(t): {"avg_xg_for": float(f or 1.2),
                         "avg_xg_against": float(a or 1.2)}
                for t, f, a in rows}

    @staticmethod
    def _fixture_adjustments(home_id: int, away_id: int, dc: dict,
                             team_avgs: dict, is_home: bool) -> tuple[float, float]:
        team_params = dc.get("team_params", {})
        form_adj = dc.get("form_adjustments", {})
        home_adv = dc.get("home_adv", 1.20)
        if not team_params:
            return 1.0, 1.0
        mean_atk = float(np.mean([v["attack"] for v in team_params.values()]))
        mean_def = float(np.mean([v["defense"] for v in team_params.values()]))

        def p(tid): return team_params.get(str(tid),
                                           {"attack": mean_atk, "defense": mean_def})

        def form(tid):
            f = form_adj.get(str(tid), [1.0, 1.0])
            return tuple(f) if len(f) == 2 else (1.0, 1.0)

        h_p, a_p = p(home_id), p(away_id)
        h_atk, h_def = form(home_id)
        a_atk, a_def = form(away_id)
        mu_h = h_p["attack"] * h_atk * a_p["defense"] * a_def * home_adv
        mu_a = a_p["attack"] * a_atk * h_p["defense"] * h_def

        tid = home_id if is_home else away_id
        avgs = team_avgs.get(tid, {"avg_xg_for": 1.2, "avg_xg_against": 1.2})
        if is_home:
            xg_adj = mu_h / max(avgs["avg_xg_for"], 0.5)
            xga_adj = mu_a / max(avgs["avg_xg_against"], 0.5)
        else:
            xg_adj = mu_a / max(avgs["avg_xg_for"], 0.5)
            xga_adj = mu_h / max(avgs["avg_xg_against"], 0.5)
        return (max(0.3, min(2.5, xg_adj)), max(0.3, min(2.5, xga_adj)))

    @staticmethod
    def _cs_probability(mu_against: float) -> float:
        from scipy.stats import poisson
        return float(poisson.pmf(0, max(mu_against, 0.01)))

    def _project(self, target_gw: int,
                 explain: set | None = None) -> dict[int, float]:
        """{player_id: projected_points} for target_gw."""
        conn = sqlite3.connect(self.db_path)
        try:
            positions = self._player_positions(conn)
            costs = dict(conn.execute(
                "SELECT player_id, current_cost FROM players"))
            cold = is_cold_start(conn)
            if cold:
                # Rates and priors from the archived season; everything else
                # (fixtures, prices, availability) stays live.
                #
                # FPL reassigns player_ids between seasons, so archive rates
                # must be matched by (web_name, position), not by player_id.
                # Keying by id silently gives a 26/27 GK the xG of whoever
                # happened to hold that id in 25/26 (e.g. an attacking MID).
                arch = sqlite3.connect(resolve_archive(self.archive_db_path))
                try:
                    archive_rates_by_id = self._compute_player_rates(arch)
                    arch_names = {
                        int(pid): (web_name, pos)
                        for pid, web_name, pos in arch.execute(
                            "SELECT player_id, web_name, position FROM players")
                    }
                    # {(web_name, position): rates}
                    # Build with collision detection: if two archive players
                    # share a name+position key (e.g. two "Gray" MIDs from
                    # different clubs), mark the key ambiguous so we fall back
                    # to prior rather than silently picking whichever row was
                    # read last.
                    _seen: set = set()
                    _ambiguous: set = set()
                    archive_rates_by_name: dict = {}
                    for pid, r in archive_rates_by_id.items():
                        if pid not in arch_names:
                            continue
                        key = arch_names[pid]
                        if key in _seen:
                            _ambiguous.add(key)
                            archive_rates_by_name.pop(key, None)
                        elif key not in _ambiguous:
                            _seen.add(key)
                            archive_rates_by_name[key] = r
                    priors = fit_positional_priors(arch)
                    # Re-key to 26/27 player_ids via name+position match
                    # (conn is still open here, inside the outer try block)
                    live_names = {
                        int(pid): (web_name, pos)
                        for pid, web_name, pos in conn.execute(
                            "SELECT player_id, web_name, position FROM players")
                    }
                    rates = {
                        pid: archive_rates_by_name[key]
                        for pid, key in live_names.items()
                        if key in archive_rates_by_name
                    }
                finally:
                    arch.close()
            else:
                rates = self._compute_player_rates(conn)
                priors = fit_positional_priors(conn)
            team_avgs = self._compute_team_season_avgs(conn)
            fixture_rows = conn.execute(
                "SELECT home_team_id, away_team_id FROM fixtures "
                "WHERE gameweek_id = ?", (target_gw,)
            ).fetchall()
        finally:
            conn.close()

        dc = self._load_dc()
        team_params = dc.get("team_params", {})
        form_adj = dc.get("form_adjustments", {})
        mean_atk = (float(np.mean([v["attack"] for v in team_params.values()]))
                    if team_params else 1.0)
        mean_def = (float(np.mean([v["defense"] for v in team_params.values()]))
                    if team_params else 1.0)

        team_fixture_adj: dict[int, tuple[float, float]] = {}
        for h, a in fixture_rows:
            team_fixture_adj[int(h)] = self._fixture_adjustments(h, a, dc, team_avgs, True)
            team_fixture_adj[int(a)] = self._fixture_adjustments(h, a, dc, team_avgs, False)

        out: dict[int, float] = {}
        self.explained: dict[int, dict] = {}
        for pid, (pos, tid) in positions.items():
            r = rates.get(pid)
            if r is None:
                # No history in the rates source: promoted-side player, new
                # signing, or rookie. Use the fitted (position, price) prior
                # rather than projecting zero and hiding them from the
                # optimiser entirely.
                r = dict(prior_for(priors, POSITION_IDS[pos], costs.get(pid, 45)))
                r.setdefault("gw_count", 0)
            avg_min = r["avg_minutes"]

            if avg_min >= 75:
                start_prob = 0.95
            elif avg_min >= 60:
                start_prob = 0.85
            elif avg_min >= 45:
                start_prob = 0.65
            elif avg_min >= 25:
                start_prob = 0.40
            else:
                start_prob = 0.15

            p60 = start_prob * max(0.0, min(1.0, (avg_min - 30) / 40))
            app_pts = p60 * PT_APP60 + (start_prob - p60) * PT_APP_SUB

            xg_adj, xga_adj = team_fixture_adj.get(tid, (1.0, 1.0))

            # Expected goals against — fixture adjustment applied EXACTLY ONCE
            if team_params:
                tp = team_params.get(str(tid), {"attack": mean_atk, "defense": mean_def})
                fa = form_adj.get(str(tid), [1.0, 1.0])
                adjusted_xga = tp["defense"] * fa[1] * mean_atk * xga_adj
            else:
                adjusted_xga = 1.2 * xga_adj

            cs_prob = self._cs_probability(adjusted_xga)
            cs_pts = cs_prob * PT_CS[pos] * p60 if pos != "FWD" else 0.0

            if pos in ("GK", "DEF"):
                concede_pts = (max(0.0, (adjusted_xga - 1.0) * 0.5)
                               * PT_CONCEDE * p60)
            else:
                concede_pts = 0.0

            eff_xg = r["avg_xg"] if r["avg_xg"] > 0 else r["avg_goals"]
            goal_pts = eff_xg * xg_adj * PT_GOAL[pos] * start_prob
            eff_xa = r["avg_xa"] if r["avg_xa"] > 0 else r["avg_assists"]
            assist_pts = eff_xa * xg_adj * PT_ASSIST * start_prob

            if pos == "GK":
                exp_sot = max(0.0, adjusted_xga * 2.5)
                exp_saves = max(0.0, exp_sot - adjusted_xga)
                save_pts = (exp_saves / 3.0) * PT_SAVE3 * p60
            else:
                save_pts = 0.0

            bonus_pts = r["avg_bonus"] * start_prob
            total = (app_pts + cs_pts + concede_pts + goal_pts
                     + assist_pts + save_pts + bonus_pts)
            out[pid] = round(max(0.0, total), 2)

            if explain and pid in explain:
                self.explained[pid] = {
                    "position": pos, "team_id": tid,
                    "from_prior": rates.get(pid) is None,
                    "avg_minutes": round(avg_min, 1),
                    "start_prob": round(start_prob, 3), "p60": round(p60, 3),
                    "xg_adj": round(xg_adj, 3), "xga_adj": round(xga_adj, 3),
                    "adjusted_xga": round(adjusted_xga, 3),
                    "cs_prob": round(cs_prob, 3),
                    "eff_xg": round(eff_xg, 4), "eff_xa": round(eff_xa, 4),
                    "PTS_appearance": round(app_pts, 2),
                    "PTS_clean_sheet": round(cs_pts, 2),
                    "PTS_concede": round(concede_pts, 2),
                    "PTS_goals": round(goal_pts, 2),
                    "PTS_assists": round(assist_pts, 2),
                    "PTS_saves": round(save_pts, 2),
                    "PTS_bonus": round(bonus_pts, 2),
                    "TOTAL": round(max(0.0, total), 2),
                }
        return out
