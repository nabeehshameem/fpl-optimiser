"""
fpl/src/dc_predictor.py
Dixon-Coles Poisson model trained on Premier League 2025/26 match data.

Model:
  mu_home = alpha_home * beta_away * home_adv
  mu_away = alpha_away * beta_home

  alpha = attack strength, beta = defensive weakness.
  Dixon-Coles tau correction applied to low-score cells (0-0/1-0/0-1/1-1).
  Recency decay: more recent matches weighted higher.

Usage:
  from fpl.src.dc_predictor import FPLDCPredictor
  p = FPLDCPredictor()
  p.fit()
  p.save()

  p = FPLDCPredictor()
  p.load()
  result = p.predict(home_team_id, away_team_id)
"""

import json
import sqlite3
from datetime import date
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH      = PROJECT_ROOT / "data" / "fpl.db"
MODEL_PATH   = PROJECT_ROOT / "models" / "fpl_dc_params.json"

MAX_GOALS = 8


def _dc_tau(x: int, y: int, mu_h: float, mu_a: float, rho: float) -> float:
    if x == 0 and y == 0:
        return 1.0 - mu_h * mu_a * rho
    if x == 1 and y == 0:
        return 1.0 + mu_a * rho
    if x == 0 and y == 1:
        return 1.0 + mu_h * rho
    if x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


def _recency_weight(kickoff_time: str, decay: float = 0.8) -> float:
    """Exponential recency decay. Recent = higher weight (1.0), season start ≈ 0.45."""
    try:
        d = date.fromisoformat(str(kickoff_time)[:10])
    except (ValueError, TypeError):
        return 0.5
    days_ago = (date.today() - d).days
    return float(np.exp(-decay * days_ago / 365.0))


class FPLDCPredictor:
    """
    Dixon-Coles Poisson model for PL match prediction.
    Fits on one full PL season (380 games); predicts any team-pair scoreline.
    """

    def __init__(self, db_path: Path | None = None,
                 model_path: Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path else DB_PATH
        self.model_path = Path(model_path) if model_path else MODEL_PATH
        self.team_params: dict[str, dict]   = {}   # {short_name: {attack, defense}}
        self.team_names:  dict[int, str]    = {}   # {team_id: name}
        self.team_short_names: dict[int, str] = {}  # {archive_team_id: short_name}
        self.home_adv: float  = 1.20
        self.rho:      float  = -0.10
        self._fitted:  bool   = False
        self.elo_ratings:      dict[int, float] = {}
        self.form_adjustments: dict[str, tuple[float, float]] = {}  # {short_name: (atk_mult, def_mult)}

    # ── Data loading ──────────────────────────────────────────────────────────

    def _load_fixtures(self) -> list[dict]:
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("""
            SELECT f.fixture_id, f.home_team_id, f.away_team_id,
                   f.home_score, f.away_score, f.kickoff_time,
                   th.name AS home_name, ta.name AS away_name
            FROM   fixtures f
            JOIN   teams th ON f.home_team_id = th.team_id
            JOIN   teams ta ON f.away_team_id = ta.team_id
            WHERE  f.home_score IS NOT NULL AND f.away_score IS NOT NULL
              AND  f.finished = 1
            ORDER  BY f.kickoff_time
        """).fetchall()
        conn.close()

        out = []
        for fixture_id, hid, aid, hs, as_, ko, hname, aname in rows:
            out.append({
                "home_id":    hid,
                "away_id":    aid,
                "home_goals": int(hs),
                "away_goals": int(as_),
                "kickoff":    str(ko or ""),
                "weight":     _recency_weight(ko),
                "home_name":  hname,
                "away_name":  aname,
            })
        return out

    def _load_team_names(self) -> dict[int, str]:
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("SELECT team_id, name, short_name FROM teams").fetchall()
        conn.close()
        self.team_short_names = {r[0]: r[2] for r in rows}
        return {r[0]: r[1] for r in rows}

    # ── ELO ───────────────────────────────────────────────────────────────────

    def _compute_elo(self, fixtures: list[dict]) -> dict[int, float]:
        """Simple ELO from PL match history, chronological order."""
        K = 25
        ratings: dict[int, float] = {}
        for m in fixtures:
            h, a = m["home_id"], m["away_id"]
            hg, ag = m["home_goals"], m["away_goals"]
            r_h = ratings.get(h, 1000.0)
            r_a = ratings.get(a, 1000.0)
            exp_h = 1.0 / (1.0 + 10.0 ** ((r_a - r_h) / 400.0))
            act_h = 1.0 if hg > ag else (0.0 if hg < ag else 0.5)
            mov = min(float(np.log1p(abs(hg - ag))) * 0.5 + 1.0, 2.5)
            delta = K * mov * (act_h - exp_h)
            ratings[h] = r_h + delta
            ratings[a] = r_a - delta
        return ratings

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit(self) -> dict:
        """Fit Dixon-Coles on PL 2025/26 fixtures."""
        fixtures = self._load_fixtures()
        self.team_names = self._load_team_names()

        if not fixtures:
            raise RuntimeError("No finished fixtures in DB. Check data/fpl.db.")

        team_ids = sorted({m["home_id"] for m in fixtures} | {m["away_id"] for m in fixtures})
        idx = {tid: i for i, tid in enumerate(team_ids)}
        n   = len(team_ids)

        home_idx = np.array([idx[m["home_id"]]   for m in fixtures], dtype=np.int32)
        away_idx = np.array([idx[m["away_id"]]   for m in fixtures], dtype=np.int32)
        hg       = np.array([m["home_goals"]      for m in fixtures], dtype=np.float64)
        ag       = np.array([m["away_goals"]      for m in fixtures], dtype=np.float64)
        weights  = np.array([m["weight"]          for m in fixtures], dtype=np.float64)

        from scipy.special import gammaln
        log_fact_h = gammaln(hg + 1)
        log_fact_a = gammaln(ag + 1)

        m00 = (hg == 0) & (ag == 0)
        m10 = (hg == 1) & (ag == 0)
        m01 = (hg == 0) & (ag == 1)
        m11 = (hg == 1) & (ag == 1)

        # L2 regularisation toward neutral (log 1.0 = 0) — no FIFA rank priors for PL
        REG_LAMBDA = 5.0

        self.elo_ratings = self._compute_elo(fixtures)
        # ELO → log attack prior: ±600 ELO ≈ ±1 log unit
        log_prior_atk = np.array(
            [(self.elo_ratings.get(tid, 1000.0) - 1000.0) / 600.0 for tid in team_ids]
        )
        log_prior_def = -log_prior_atk * 0.5  # defense correlates inversely but less tightly

        def neg_ll(params: np.ndarray) -> float:
            atk = np.exp(params[:n])
            dfn = np.exp(params[n:2 * n])
            ha  = np.exp(params[2 * n])
            rho = params[2 * n + 1]

            mu_h = atk[home_idx] * dfn[away_idx] * ha
            mu_a = atk[away_idx] * dfn[home_idx]

            log_p = (hg * np.log(np.maximum(mu_h, 1e-12)) - mu_h - log_fact_h
                   + ag * np.log(np.maximum(mu_a, 1e-12)) - mu_a - log_fact_a)

            tau = np.ones(len(hg))
            tau[m00] = 1.0 - mu_h[m00] * mu_a[m00] * rho
            tau[m10] = 1.0 + mu_a[m10] * rho
            tau[m01] = 1.0 + mu_h[m01] * rho
            tau[m11] = 1.0 - rho
            if np.any(tau <= 0):
                return 1e10

            ll = float(np.dot(weights, np.log(np.maximum(tau, 1e-12)) + log_p))

            # Regularisation (skip first team — fixed for identifiability)
            reg = REG_LAMBDA * (
                float(np.sum((params[1:n]     - log_prior_atk[1:]) ** 2)) +
                float(np.sum((params[n:2 * n] - log_prior_def)     ** 2))
            )
            return -ll + reg

        x0 = np.zeros(2 * n + 2)
        x0[:n]       = log_prior_atk
        x0[0]        = 0.0
        x0[n:2 * n]  = log_prior_def
        x0[2 * n]    = np.log(1.20)   # PL home advantage ~20%
        x0[2 * n + 1] = -0.10

        bounds = (
            [(0.0, 0.0)]
            + [(-3.0, 3.0)] * (n - 1)
            + [(-3.0, 3.0)] * n
            + [(np.log(1.0), np.log(1.5))]   # home adv: 0–50%
            + [(-0.5, 0.2)]
        )

        result = minimize(
            neg_ll, x0, method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 500, "maxfun": 100000, "ftol": 1e-7, "gtol": 1e-3},
        )

        opt     = result.x
        log_atk = opt[:n]
        log_def = opt[n:2 * n]
        self.home_adv = float(np.exp(opt[2 * n]))
        self.rho      = float(opt[2 * n + 1])

        self.team_params = {
            self.team_short_names.get(tid, str(tid)): {
                "attack":  float(np.exp(log_atk[i])),
                "defense": float(np.exp(log_def[i])),
            }
            for i, tid in enumerate(team_ids)
        }
        self._fitted = True

        self.form_adjustments = self._compute_form_adjustments(fixtures)

        return {
            "n_teams":     n,
            "n_fixtures":  len(fixtures),
            "home_adv":    round(self.home_adv, 4),
            "rho":         round(self.rho, 4),
            "converged":   result.success,
            "message":     result.message,
        }

    def _compute_form_adjustments(
        self, fixtures: list[dict], window_days: int = 120, n_matches: int = 10
    ) -> dict[int, tuple[float, float]]:
        """
        Compare actual goals vs DC-expected in each team's last n_matches.
        Returns {team_id: (atk_mult, def_mult)} capped at ±20%, dampened 50%.
        """
        from datetime import timedelta
        cutoff = str(date.today() - timedelta(days=window_days))
        recent = sorted(
            [m for m in fixtures if m.get("kickoff", "") >= cutoff],
            key=lambda x: x["kickoff"],
        )

        records: dict[str, list[tuple[float, float, float, float]]] = {}
        for m in recent:
            h, a = m["home_id"], m["away_id"]
            h_sn = self.team_short_names.get(h, str(h))
            a_sn = self.team_short_names.get(a, str(a))
            hg_act, ag_act = float(m["home_goals"]), float(m["away_goals"])

            h_p = self.team_params.get(h_sn, {"attack": 1.0, "defense": 1.0})
            a_p = self.team_params.get(a_sn, {"attack": 1.0, "defense": 1.0})

            exp_h = h_p["attack"] * a_p["defense"] * self.home_adv
            exp_a = a_p["attack"] * h_p["defense"]

            records.setdefault(h_sn, []).append((hg_act, exp_h, ag_act, exp_a))
            records.setdefault(a_sn, []).append((ag_act, exp_a, hg_act, exp_h))

        adjustments: dict[str, tuple[float, float]] = {}
        for sn, team_records in records.items():
            last = team_records[-n_matches:]
            if len(last) < 4:
                continue
            n = len(last)
            recency_w = np.linspace(0.75, 1.0, n)

            scored_ratios   = np.array([min(r[0], 6.0) / max(min(r[1], 2.5), 0.1) for r in last])
            conceded_ratios = np.array([max(0.35, min(r[2], 6.0) / max(min(r[3], 2.5), 0.1)) for r in last])

            raw_atk = float(np.average(scored_ratios,   weights=recency_w))
            raw_def = float(np.average(conceded_ratios, weights=recency_w))

            atk_mult = max(0.80, min(1.20, 1.0 + 0.50 * (raw_atk - 1.0)))
            def_mult = max(0.80, min(1.20, 1.0 + 0.50 * (raw_def - 1.0)))
            adjustments[sn] = (atk_mult, def_mult)

        return adjustments

    # ── Prediction ────────────────────────────────────────────────────────────

    def _team_params(self, team_id: int) -> tuple[float, float]:
        sn = self.team_short_names.get(team_id, str(team_id))
        if sn in self.team_params:
            p = self.team_params[sn]
            return p["attack"], p["defense"]
        return 1.0, 1.0

    def predict(
        self,
        home_id: int,
        away_id: int,
        max_goals: int = MAX_GOALS,
    ) -> dict:
        """
        Predict scoreline distribution for home_id vs away_id.

        Returns:
          home_name, away_name, home_xg, away_xg,
          win_pct, draw_pct, loss_pct (home team perspective),
          home_cs_pct, away_cs_pct (clean sheet probabilities),
          most_likely: top 10 (h_goals, a_goals, pct),
          prob_matrix: np.ndarray,
        """
        if not self._fitted and not self.team_params:
            raise RuntimeError("Model not fitted. Call fit() or load() first.")

        atk_h, def_h = self._team_params(home_id)
        atk_a, def_a = self._team_params(away_id)

        h_sn = self.team_short_names.get(home_id, str(home_id))
        a_sn = self.team_short_names.get(away_id, str(away_id))
        h_atk_m, h_def_m = self.form_adjustments.get(h_sn, (1.0, 1.0))
        a_atk_m, a_def_m = self.form_adjustments.get(a_sn, (1.0, 1.0))

        mu_h = atk_h * h_atk_m * def_a * a_def_m * self.home_adv
        mu_a = atk_a * a_atk_m * def_h * h_def_m

        mu_h = min(mu_h, 6.0)
        mu_a = min(mu_a, 6.0)

        goals = np.arange(max_goals + 1)
        mat   = np.outer(poisson.pmf(goals, mu_h), poisson.pmf(goals, mu_a))
        for x in range(2):
            for y in range(2):
                mat[x, y] *= max(_dc_tau(x, y, mu_h, mu_a, self.rho), 0.0)
        mat /= mat.sum()

        win_pct  = float(np.tril(mat, -1).sum()) * 100
        draw_pct = float(np.diag(mat).sum())      * 100
        loss_pct = float(np.triu(mat, 1).sum())   * 100

        # CS probabilities: P(team concedes 0)
        home_cs_pct = float(mat[:, 0].sum()) * 100   # home keeps CS = away scores 0
        away_cs_pct = float(mat[0, :].sum()) * 100   # away keeps CS = home scores 0

        flat = sorted(
            [(i, j, float(mat[i, j]) * 100)
             for i in range(max_goals + 1)
             for j in range(max_goals + 1)],
            key=lambda x: x[2], reverse=True,
        )

        return {
            "home_id":      home_id,
            "away_id":      away_id,
            "home_name":    self.team_names.get(home_id, str(home_id)),
            "away_name":    self.team_names.get(away_id, str(away_id)),
            "home_xg":      round(mu_h, 2),
            "away_xg":      round(mu_a, 2),
            "win_pct":      round(win_pct, 1),
            "draw_pct":     round(draw_pct, 1),
            "loss_pct":     round(loss_pct, 1),
            "home_cs_pct":  round(home_cs_pct, 1),
            "away_cs_pct":  round(away_cs_pct, 1),
            "most_likely":  [(h, a, round(p, 1)) for h, a, p in flat[:10]],
            "prob_matrix":  mat,
        }

    def predict_season(self, fixtures: list[tuple[int, int]]) -> list[dict]:
        """Predict a list of (home_id, away_id) fixture pairs."""
        return [self.predict(h, a) for h, a in fixtures]

    def team_strengths(self) -> list[dict]:
        """All teams sorted by attack strength."""
        return sorted(
            [
                {
                    "short_name": sn,
                    "attack":   round(p["attack"], 3),
                    "defense":  round(p["defense"], 3),
                    "form_atk": round(self.form_adjustments.get(sn, (1.0, 1.0))[0], 3),
                    "form_def": round(self.form_adjustments.get(sn, (1.0, 1.0))[1], 3),
                }
                for sn, p in self.team_params.items()
            ],
            key=lambda x: x["attack"],
            reverse=True,
        )

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        import datetime as _dt
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model_path.write_text(json.dumps(
            {
                "trained_at":       _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "home_adv":         self.home_adv,
                "rho":              self.rho,
                "team_params":      {k: v for k, v in self.team_params.items()},
                "team_names":       {str(k): v for k, v in self.team_names.items()},
                "team_short_names": {str(k): v for k, v in self.team_short_names.items()},
                "elo_ratings":      {str(k): v for k, v in self.elo_ratings.items()},
                "form_adjustments": {k: list(v) for k, v in self.form_adjustments.items()},
            },
            indent=2,
        ))
        print(f"FPL DC model saved: {self.model_path}")

    def load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"No FPL DC model at {self.model_path}. Run: python scripts/train_dc.py"
            )
        data = json.loads(self.model_path.read_text())
        self.home_adv         = data["home_adv"]
        self.rho              = data["rho"]
        self.team_params      = {k: v for k, v in data["team_params"].items()}
        self.team_names       = {int(k): v for k, v in data.get("team_names", {}).items()}
        self.team_short_names = {int(k): v for k, v in data.get("team_short_names", {}).items()}
        self.elo_ratings      = {int(k): v for k, v in data.get("elo_ratings", {}).items()}
        self.form_adjustments = {k: tuple(v) for k, v in data.get("form_adjustments", {}).items()}
        self._fitted          = True
