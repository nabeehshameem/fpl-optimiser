"""
score_predictor.py
Dixon-Coles Poisson model for WC match score prediction.

Data sources blended during fitting:
  1. WC2022 StatsBomb matches   — weight 1.00  (gold standard: WC context, rich data)
  2. WC2018 StatsBomb matches   — weight 0.60  (older WC context)
  3. Recent international results (recent_results table, last ~2 years):
       Tier A  WCQ / Euros / Copa América  — base weight 0.85, then decayed by recency
       Tier B  Nations Leagues             — base weight 0.70, decayed by recency
     Recency decay: exp(-0.6 * days_ago / 365)
     so a Tier-A match 6 months ago ≈ 0.74, 18 months ago ≈ 0.52

Model:
  Expected goals:
    mu_home = alpha_home * beta_away * home_adv
    mu_away = alpha_away * beta_home

  alpha = attack strength, beta = defensive weakness.
  Dixon-Coles tau correction applied to 0-0 / 1-0 / 0-1 / 1-1.

Usage:
  from wc.src.score_predictor import DCPredictor
  p = DCPredictor()
  p.fit()
  p.save()

  p = DCPredictor()
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
DB_PATH      = PROJECT_ROOT / "data" / "wc.db"
MODEL_PATH   = PROJECT_ROOT / "models" / "dc_params.json"

MAX_GOALS = 8

# Tier weights from ingest_recent_form — must stay in sync
TIER_WEIGHTS = {"A": 0.85, "B": 0.70, "C": 0.35}
TOURNAMENT_TIERS = {
    "FIFA World Cup qualification (CONMEBOL)":    "A",
    "FIFA World Cup qualification (UEFA)":         "A",
    "FIFA World Cup qualification (AFC)":          "A",
    "FIFA World Cup qualification (CAF)":          "A",
    "FIFA World Cup qualification (CONCACAF)":     "A",
    "FIFA World Cup qualification (OFC)":          "A",
    "FIFA World Cup qualification":                "A",
    "UEFA European Championship":                  "A",
    "UEFA European Championship qualification":    "A",
    "Copa América":                                "A",
    "Africa Cup of Nations":                       "A",
    "AFC Asian Cup":                               "A",
    "CONCACAF Gold Cup":                           "A",
    "AFC Asian Cup qualification":                 "B",
    "UEFA Nations League A":                       "B",
    "UEFA Nations League B":                       "B",
    "UEFA Nations League C":                       "B",
    "UEFA Nations League D":                       "B",
    "UEFA Nations League":                         "B",
    "CONCACAF Nations League":                     "B",
    "CAF Africa Cup of Nations qualification":     "B",
    "CONMEBOL-UEFA Cup of Champions":              "B",
    "Intercontinental Playoff":                    "B",
    "Friendly":                                    "C",
}

# Normalised team name aliases: martj42 name (lowercase) → canonical name
# Add more as needed for WC2026 qualification mismatches
TEAM_ALIASES: dict[str, str] = {
    "united states":                "usa",
    "south korea":                  "korea republic",
    "north korea":                  "korea dpr",
    "iran":                         "ir iran",
    "china":                        "china pr",
    "chinese taipei":               "chinese taipei",
    "cape verde":                   "cape verde islands",
    "republic of ireland":          "republic of ireland",
    "northern ireland":             "northern ireland",
    "trinidad & tobago":            "trinidad and tobago",
    "st. kitts & nevis":            "saint kitts and nevis",
    "st. lucia":                    "saint lucia",
    "st. vincent & the grenadines": "saint vincent and the grenadines",
    "côte d'ivoire":                "ivory coast",
    "cote d'ivoire":                "ivory coast",
    "dr congo":                     "dr congo",
    "democratic republic of the congo": "dr congo",
    "guinea-bissau":                "guinea-bissau",
    "são tomé & príncipe":          "sao tome and principe",
    "eswatini":                     "swaziland",
    "north macedonia":              "north macedonia",
    "czech republic":               "czech republic",
    "türkiye":                      "turkey",
    "curacao":                      "curacao",
    "antigua & barbuda":            "antigua and barbuda",
}


def _norm_name(name: str) -> str:
    return name.lower().strip()


def _canonical(name: str) -> str:
    n = _norm_name(name)
    return TEAM_ALIASES.get(n, n)


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


def _rank_prior(rank: int) -> float:
    """FIFA rank → rough attack strength. Rank 1 ≈ 1.4, rank 48 ≈ 0.7."""
    return max(0.5, 1.45 - (rank - 1) * 0.015)


def _recency_weight(match_date_str: str, decay: float = 0.6) -> float:
    """Exponential decay: exp(-decay * years_ago). Recent = higher weight."""
    try:
        d = date.fromisoformat(str(match_date_str)[:10])
    except ValueError:
        return 0.5
    days_ago = (date.today() - d).days
    return float(np.exp(-decay * days_ago / 365.0))


class DCPredictor:
    """
    Dixon-Coles Poisson score predictor.

    team_params is keyed by a canonical team name (lowercase string).
    predict() accepts team_id integers and resolves them to names via the DB.
    """

    def __init__(self) -> None:
        self.team_params: dict[str, dict] = {}   # {canonical_name: {attack, defense}}
        self.home_adv: float = 1.05
        self.rho: float = -0.1
        self._id_to_name: dict[int, str] = {}    # populated by fit() / load()
        self._fitted = False

    # ── Data loading ──────────────────────────────────────────────────────────

    def _load_wc_matches(self) -> list[dict]:
        """WC2018/2022 from StatsBomb matches table. Already name-resolved."""
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute("""
            SELECT m.home_team_id, m.away_team_id,
                   m.home_score, m.away_score,
                   m.tournament, m.match_date,
                   th.name AS home_name, ta.name AS away_name
            FROM matches m
            LEFT JOIN teams th ON m.home_team_id = th.team_id
            LEFT JOIN teams ta ON m.away_team_id = ta.team_id
            WHERE m.home_score IS NOT NULL AND m.away_score IS NOT NULL
        """).fetchall()
        conn.close()

        out = []
        for r in rows:
            home_name = _canonical(r[6] or str(r[0]))
            away_name = _canonical(r[7] or str(r[1]))
            tourney   = r[4] or ""
            # WC2022 = weight 1.0, WC2018 = 0.6; no recency decay (already decayed by era)
            base_w = 1.0 if "2022" in tourney else 0.6
            out.append({
                "home_key":   home_name, "away_key": away_name,
                "home_goals": int(r[2]), "away_goals": int(r[3]),
                "weight":     base_w,
                "home_id":    r[0],      "away_id":   r[1],
            })
        return out

    def _load_recent_matches(self) -> list[dict]:
        """Recent international results from recent_results table."""
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute("""
            SELECT match_date, home_team, away_team,
                   home_score, away_score, tournament
            FROM recent_results
            WHERE home_score IS NOT NULL AND away_score IS NOT NULL
        """).fetchall()
        conn.close()

        out = []
        for r in rows:
            match_date, home, away, hg, ag, tourney = r
            tier     = TOURNAMENT_TIERS.get(tourney or "", "C")
            tier_w   = TIER_WEIGHTS.get(tier, 0.4)
            recency  = _recency_weight(match_date)
            weight   = tier_w * recency
            if weight < 0.05:   # skip very old / low-quality matches
                continue
            out.append({
                "home_key":   _canonical(home), "away_key": _canonical(away),
                "home_goals": int(hg), "away_goals": int(ag),
                "weight":     weight,
                "home_id":    None, "away_id": None,
            })
        return out

    def _build_id_name_map(self) -> None:
        """Populate _id_to_name from the teams table."""
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute("SELECT team_id, name FROM teams").fetchall()
        conn.close()
        self._id_to_name = {r[0]: _canonical(r[1]) for r in rows}

    def _load_team_ranks(self) -> dict[str, float]:
        """canonical_name → FIFA rank-based prior strength."""
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute(
            "SELECT name, fifa_rank FROM teams WHERE fifa_rank IS NOT NULL"
        ).fetchall()
        conn.close()
        return {_canonical(r[0]): _rank_prior(int(r[1])) for r in rows}

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit(self) -> dict:
        """
        Fit Dixon-Coles on WC historical + recent international form.
        Returns a dict with fit diagnostics.
        """
        wc_matches     = self._load_wc_matches()
        recent_matches = self._load_recent_matches()
        all_matches    = wc_matches + recent_matches

        if not all_matches:
            raise RuntimeError(
                "No match data found. Run ingest_sb.py and/or ingest_recent_form.py first."
            )

        # Collect all unique team keys
        team_keys = sorted({
            key
            for m in all_matches
            for key in (m["home_key"], m["away_key"])
        })
        idx = {k: i for i, k in enumerate(team_keys)}
        n   = len(team_keys)

        # Precompute arrays outside the optimizer loop for vectorised neg_ll
        home_idx   = np.array([idx[m["home_key"]]   for m in all_matches], dtype=np.int32)
        away_idx   = np.array([idx[m["away_key"]]   for m in all_matches], dtype=np.int32)
        hg         = np.array([m["home_goals"]       for m in all_matches], dtype=np.float64)
        ag         = np.array([m["away_goals"]       for m in all_matches], dtype=np.float64)
        weights    = np.array([m["weight"]           for m in all_matches], dtype=np.float64)

        # Precompute log-factorials (constant across optimizer calls)
        from scipy.special import gammaln
        log_fact_h = gammaln(hg + 1)
        log_fact_a = gammaln(ag + 1)

        # DC low-score masks (only 0-0 / 1-0 / 0-1 / 1-1 rows need tau correction)
        m00 = (hg == 0) & (ag == 0)
        m10 = (hg == 1) & (ag == 0)
        m01 = (hg == 0) & (ag == 1)
        m11 = (hg == 1) & (ag == 1)

        def neg_ll(params: np.ndarray) -> float:
            atk = np.exp(params[:n])
            dfn = np.exp(params[n:2 * n])
            ha  = np.exp(params[2 * n])
            rho = params[2 * n + 1]

            mu_h = atk[home_idx] * dfn[away_idx] * ha
            mu_a = atk[away_idx] * dfn[home_idx]

            # Vectorised Poisson log-PMF: x*log(mu) - mu - log(x!)
            log_p = (hg * np.log(np.maximum(mu_h, 1e-12)) - mu_h - log_fact_h
                   + ag * np.log(np.maximum(mu_a, 1e-12)) - mu_a - log_fact_a)

            # Dixon-Coles tau correction (vectorised)
            tau = np.ones(len(hg))
            tau[m00] = 1.0 - mu_h[m00] * mu_a[m00] * rho
            tau[m10] = 1.0 + mu_a[m10] * rho
            tau[m01] = 1.0 + mu_h[m01] * rho
            tau[m11] = 1.0 - rho
            if np.any(tau <= 0):
                return 1e10

            return -float(np.dot(weights, np.log(np.maximum(tau, 1e-12)) + log_p))

        x0         = np.zeros(2 * n + 2)
        x0[2 * n]  = np.log(1.05)
        x0[2 * n + 1] = -0.1

        bounds = (
            [(0.0, 0.0)]                              # first team fixed for identifiability
            + [(-3.0, 3.0)] * (n - 1)
            + [(-3.0, 3.0)] * n
            + [(np.log(0.85), np.log(1.4))]
            + [(-0.5, 0.2)]
        )

        result = minimize(
            neg_ll, x0, method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 3000, "ftol": 1e-12, "gtol": 1e-8},
        )

        opt     = result.x
        log_atk = opt[:n]
        log_def = opt[n:2 * n]
        self.home_adv = float(np.exp(opt[2 * n]))
        self.rho      = float(opt[2 * n + 1])

        self.team_params = {
            key: {
                "attack":  float(np.exp(log_atk[i])),
                "defense": float(np.exp(log_def[i])),
            }
            for i, key in enumerate(team_keys)
        }
        self._build_id_name_map()
        self._fitted = True

        wc_count     = len(wc_matches)
        recent_count = len(recent_matches)
        return {
            "n_teams":        n,
            "n_wc_matches":   wc_count,
            "n_recent":       recent_count,
            "n_total":        len(all_matches),
            "home_adv":       round(self.home_adv, 4),
            "rho":            round(self.rho, 4),
            "converged":      result.success,
            "message":        result.message,
        }

    # ── Prediction ────────────────────────────────────────────────────────────

    def _resolve_name(self, team_id: int) -> str:
        """team_id → canonical name. Falls back to str(team_id)."""
        if not self._id_to_name:
            self._build_id_name_map()
        return self._id_to_name.get(team_id, str(team_id))

    def _team_atk_def(self, team_key: str) -> tuple[float, float]:
        """Return (attack, defense) falling back to FIFA rank prior."""
        if team_key in self.team_params:
            p = self.team_params[team_key]
            return p["attack"], p["defense"]
        rank_priors = self._load_team_ranks()
        s = rank_priors.get(team_key, _rank_prior(30))
        return s, 1.0 / s

    def _display_name(self, team_id: int) -> str:
        """Return a display-friendly team name from the DB."""
        conn = sqlite3.connect(DB_PATH)
        row  = conn.execute(
            "SELECT name FROM teams WHERE team_id = ?", (team_id,)
        ).fetchone()
        conn.close()
        return row[0] if row else str(team_id)

    def predict(
        self,
        home_id: int,
        away_id: int,
        home_advantage: bool = False,
        max_goals: int = MAX_GOALS,
    ) -> dict:
        """
        Predict scoreline probabilities for one match.

        Returns:
          home_name, away_name, home_xg, away_xg,
          win_pct, draw_pct, loss_pct  (home team perspective),
          most_likely: [(home_goals, away_goals, pct), ...]  top 10,
          prob_matrix: np.ndarray [max_goals+1, max_goals+1]
        """
        if not self._fitted and not self.team_params:
            raise RuntimeError("Model not fitted. Call fit() or load() first.")

        home_key = self._resolve_name(home_id)
        away_key = self._resolve_name(away_id)

        atk_h, def_h = self._team_atk_def(home_key)
        atk_a, def_a = self._team_atk_def(away_key)
        ha = self.home_adv if home_advantage else 1.0

        mu_h = atk_h * def_a * ha
        mu_a = atk_a * def_h

        goals = np.arange(max_goals + 1)
        mat   = np.outer(poisson.pmf(goals, mu_h), poisson.pmf(goals, mu_a))
        for x in range(2):
            for y in range(2):
                mat[x, y] *= max(_dc_tau(x, y, mu_h, mu_a, self.rho), 0.0)
        mat /= mat.sum()

        win_pct  = float(np.tril(mat, -1).sum()) * 100
        draw_pct = float(np.diag(mat).sum())      * 100
        loss_pct = float(np.triu(mat, 1).sum())   * 100

        flat = sorted(
            [(i, j, float(mat[i, j]) * 100)
             for i in range(max_goals + 1)
             for j in range(max_goals + 1)],
            key=lambda x: x[2], reverse=True,
        )

        return {
            "home_id":     home_id,
            "away_id":     away_id,
            "home_name":   self._display_name(home_id),
            "away_name":   self._display_name(away_id),
            "home_xg":     round(mu_h, 2),
            "away_xg":     round(mu_a, 2),
            "win_pct":     round(win_pct, 1),
            "draw_pct":    round(draw_pct, 1),
            "loss_pct":    round(loss_pct, 1),
            "most_likely": [(h, a, round(p, 1)) for h, a, p in flat[:10]],
            "prob_matrix": mat,
        }

    def predict_matchday(self, matchday: int) -> list[dict]:
        """Predict all fixtures in a given matchday."""
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute(
            "SELECT home_team_id, away_team_id FROM fixtures WHERE matchday = ?",
            (matchday,),
        ).fetchall()
        conn.close()
        if not rows:
            raise RuntimeError(
                f"No fixtures for matchday {matchday}. Run ingest_fixtures.py first."
            )
        return [self.predict(r[0], r[1]) for r in rows]

    def team_strengths(self) -> list[dict]:
        """All teams sorted by attack strength descending."""
        return sorted(
            [{"name": k, "attack": round(v["attack"], 3), "defense": round(v["defense"], 3)}
             for k, v in self.team_params.items()],
            key=lambda x: x["attack"],
            reverse=True,
        )

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        MODEL_PATH.write_text(json.dumps(
            {
                "home_adv":    self.home_adv,
                "rho":         self.rho,
                "team_params": self.team_params,
                "id_to_name":  {str(k): v for k, v in self._id_to_name.items()},
            },
            indent=2,
        ))
        print(f"DC model saved: {MODEL_PATH}")

    def load(self) -> None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"No DC model at {MODEL_PATH}. Run: python wc/scripts/train_dc.py"
            )
        data = json.loads(MODEL_PATH.read_text())
        self.home_adv    = data["home_adv"]
        self.rho         = data["rho"]
        self.team_params = data["team_params"]
        self._id_to_name = {int(k): v for k, v in data.get("id_to_name", {}).items()}
        self._fitted     = True
