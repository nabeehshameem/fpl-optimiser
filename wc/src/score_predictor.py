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
    "UEFA Euro":                                   "A",
    "Copa América":                                "A",
    "Africa Cup of Nations":                       "A",
    "African Cup of Nations":                      "A",
    "AFC Asian Cup":                               "A",
    "CONCACAF Gold Cup":                           "A",
    "Gold Cup":                                    "A",
    "AFC Asian Cup qualification":                 "B",
    "UEFA Nations League A":                       "B",
    "UEFA Nations League B":                       "B",
    "UEFA Nations League C":                       "B",
    "UEFA Nations League D":                       "B",
    "UEFA Nations League":                         "B",
    "CONCACAF Nations League":                     "B",
    "CAF Africa Cup of Nations qualification":     "B",
    "African Cup of Nations qualification":        "B",
    "Gold Cup qualification":                      "B",
    "CONCACAF Series":                             "B",
    "FIFA Series":                                 "B",
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
    # SofaScore name variants
    "czechia":                      "czech republic",
    "bosnia & herzegovina":         "bosnia and herzegovina",
    "cabo verde":                   "cape verde islands",
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
    """FIFA rank → rough attack strength. Rank 1 ≈ 1.75, rank 80 ≈ 0.40."""
    return max(0.4, 1.75 - (rank - 1) * 0.017)


def _recency_weight(match_date_str: str, decay: float = 1.0) -> float:
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
        self.elo_ratings: dict[str, float] = {}  # canonical_name → ELO (set by fit)
        self.form_adjustments: dict[str, tuple[float, float]] = {}  # → (atk_mult, def_mult)

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
            # WC2022 = weight 1.0, WC2018 = 0.2 (8-year-old data, different squads)
            base_w = 1.0 if "2022" in tourney else 0.2
            out.append({
                "home_key":   home_name, "away_key": away_name,
                "home_goals": int(r[2]), "away_goals": int(r[3]),
                "weight":     base_w,
                "neutral":    True,
                "home_id":    r[0],       "away_id":   r[1],
                "match_date": str(r[5] or "2018-01-01"),
                "tier":       "A",
            })
        return out

    def _load_wc2026_matches(self) -> list[dict]:
        """WC2026 results from match_stats — highest weight, exact current tournament."""
        try:
            conn = sqlite3.connect(DB_PATH)
            rows = conn.execute("""
                SELECT match_date, home_team, away_team, home_score, away_score
                FROM match_stats
                WHERE home_score IS NOT NULL AND away_score IS NOT NULL
            """).fetchall()
            conn.close()
        except Exception:
            return []
        out = []
        for match_date, home, away, hg, ag in rows:
            out.append({
                "home_key":   _canonical(home),
                "away_key":   _canonical(away),
                "home_goals": int(hg),
                "away_goals": int(ag),
                "weight":     4.0,  # 4× WC2022 — current tournament, exact squads, most relevant signal
                "neutral":    True,
                "home_id":    None,
                "away_id":    None,
                "match_date": str(match_date),
                "tier":       "A",
            })
        return out

    def _load_recent_matches(self) -> list[dict]:
        """Recent international results from recent_results table."""
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute("""
            SELECT match_date, home_team, away_team,
                   home_score, away_score, tournament, neutral
            FROM recent_results
            WHERE home_score IS NOT NULL AND away_score IS NOT NULL
        """).fetchall()
        conn.close()

        today = date.today()
        out = []
        for r in rows:
            match_date, home, away, hg, ag, tourney = r[0], r[1], r[2], r[3], r[4], r[5]
            tier     = TOURNAMENT_TIERS.get(tourney or "", "C")
            tier_w   = TIER_WEIGHTS.get(tier, 0.4)
            recency  = _recency_weight(match_date)
            # Pre-tournament window (90 days): actual WC squads in friendlies
            # get near-competitive weight — most relevant form signal available
            try:
                days_ago = (today - date.fromisoformat(str(match_date)[:10])).days
            except (ValueError, TypeError):
                days_ago = 999
            if days_ago <= 90:
                tier_w = max(tier_w, 0.80)
            weight   = tier_w * recency
            if weight < 0.05:   # skip very old / low-quality matches
                continue
            out.append({
                "home_key":   _canonical(home), "away_key": _canonical(away),
                "home_goals": int(hg), "away_goals": int(ag),
                "weight":     weight,
                "neutral":    bool(r[6]) if r[6] else False,
                "home_id":    None, "away_id": None,
                "match_date": str(match_date or ""),
                "tier":       tier,
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

    def _load_match_xg(self) -> dict[tuple, tuple]:
        """xG from match_stats keyed by (home_canon, away_canon, date). Returns {} on failure."""
        try:
            conn = sqlite3.connect(DB_PATH)
            rows = conn.execute(
                "SELECT match_date, home_team, away_team, xg_home, xg_away "
                "FROM match_stats WHERE xg_home IS NOT NULL"
            ).fetchall()
            conn.close()
        except Exception:
            return {}
        return {
            (_canonical(home), _canonical(away), str(d)[:10]): (float(xg_h), float(xg_a))
            for d, home, away, xg_h, xg_a in rows
        }

    # ── ELO + form ────────────────────────────────────────────────────────────

    def _compute_elo(self, all_matches: list[dict]) -> dict[str, float]:
        """
        Compute ELO ratings from all training matches in chronological order.
        K-factor: tier A = 30, B = 20, C = 10. Margin-of-victory multiplier applied.
        Used to build data-driven DC regularisation priors.
        """
        K_BY_TIER = {"A": 30, "B": 20, "C": 10}
        ratings: dict[str, float] = {}

        for m in sorted(all_matches, key=lambda x: x.get("match_date", "2000-01-01")):
            h, a = m["home_key"], m["away_key"]
            hg, ag = m["home_goals"], m["away_goals"]
            K = K_BY_TIER.get(m.get("tier", "C"), 10)

            r_h = ratings.get(h, 1000.0)
            r_a = ratings.get(a, 1000.0)

            exp_h = 1.0 / (1.0 + 10.0 ** ((r_a - r_h) / 400.0))
            act_h = 1.0 if hg > ag else (0.0 if hg < ag else 0.5)

            # Bigger wins move the needle more, capped to prevent blowout inflation
            mov = min(float(np.log1p(abs(hg - ag))) * 0.5 + 1.0, 2.5)

            delta = K * mov * (act_h - exp_h)
            ratings[h] = r_h + delta
            ratings[a] = r_a - delta

        return ratings

    def _compute_form_adjustments(
        self, all_matches: list[dict], window_days: int = 300, n_matches: int = 8
    ) -> dict[str, tuple[float, float]]:
        """
        Post-fit: for each team compare actual goals scored/conceded vs DC-expected
        in their last `n_matches` within `window_days`. Returns (atk_mult, def_mult)
        capped at ±20% and dampened 40% toward 1.0 to avoid over-reacting to noise.
        Called after fit() so DC params are available.
        """
        from datetime import timedelta
        cutoff = str(date.today() - timedelta(days=window_days))
        # Pre-tournament friendlies (last 90 days) are the most relevant signal
        # even if tier C — include them. Older tier C friendlies remain excluded.
        recent_cutoff = str(date.today() - timedelta(days=90))
        recent = sorted(
            [
                m for m in all_matches
                if m.get("match_date", "") >= cutoff
                and (
                    m.get("tier", "C") in ("A", "B")
                    or m.get("match_date", "") >= recent_cutoff
                )
            ],
            key=lambda x: x["match_date"],
        )

        xg_lookup = self._load_match_xg()

        # Tier weights: friendlies count half — they're unreliable pre-tournament prep
        # (squads rotated, motivations mixed) vs qualifiers/tournaments which are competitive.
        TIER_WEIGHT = {"A": 1.0, "B": 0.8, "C": 0.5}

        # Accumulate per team: list of (actual_scored, exp_scored, actual_conceded, exp_conceded, tier_weight)
        records: dict[str, list[tuple[float, float, float, float, float]]] = {}

        for m in recent:
            h, a = m["home_key"], m["away_key"]
            tw = TIER_WEIGHT.get(m.get("tier", "C"), 0.5)
            # Use xG when available — more reliable form signal than noisy actual goals
            xg_key = (_canonical(h), _canonical(a), str(m["match_date"])[:10])
            if xg_key in xg_lookup:
                hg, ag = xg_lookup[xg_key]
            else:
                hg, ag = float(m["home_goals"]), float(m["away_goals"])

            atk_h, def_h = self._team_atk_def(h)
            atk_a, def_a = self._team_atk_def(a)
            ha = self.home_adv if not m.get("neutral", False) else 1.0

            exp_h = atk_h * def_a * ha
            exp_a = atk_a * def_h

            records.setdefault(h, []).append((hg, exp_h, ag, exp_a, tw))
            records.setdefault(a, []).append((ag, exp_a, hg, exp_h, tw))

        adjustments: dict[str, tuple[float, float]] = {}
        for team, team_records in records.items():
            last = team_records[-n_matches:]
            if len(last) < 3:
                continue

            n = len(last)
            recency_w = np.linspace(0.5, 1.0, n)
            tier_w    = np.array([r[4] for r in last])
            w         = recency_w * tier_w

            # Cap expected at 2.5 so a sky-high DC baseline against weak opponents
            # doesn't suppress the form signal for strong teams (e.g. Argentina 5-0
            # Zambia expected 4.8 → ratio 1.04 vs capped-at-2.5 → ratio 2.0)
            scored_ratios   = np.array([min(r[0], 6.0) / max(min(r[1], 2.5), 0.1) for r in last])
            conceded_ratios = np.array([min(r[2], 6.0) / max(min(r[3], 2.5), 0.1) for r in last])

            raw_atk = float(np.average(scored_ratios, weights=w))
            raw_def = float(np.average(conceded_ratios, weights=w))

            # Dampen 55% toward 1.0 and cap at ±30%
            atk_mult = max(0.70, min(1.30, 1.0 + 0.55 * (raw_atk - 1.0)))
            def_mult = max(0.70, min(1.30, 1.0 + 0.55 * (raw_def - 1.0)))

            adjustments[team] = (atk_mult, def_mult)

        return adjustments

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit(self) -> dict:
        """
        Fit Dixon-Coles on WC historical + recent international form.
        Returns a dict with fit diagnostics.
        """
        wc_matches      = self._load_wc_matches()
        wc2026_matches  = self._load_wc2026_matches()
        recent_matches  = self._load_recent_matches()
        # WC2026 goes last so form adjustments use it as the most recent signal
        all_matches     = wc_matches + recent_matches + wc2026_matches

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
        # 0 = neutral venue (no home advantage), 1 = genuine home game
        home_mask  = np.array([0.0 if m.get("neutral", False) else 1.0 for m in all_matches], dtype=np.float64)

        # Precompute log-factorials (constant across optimizer calls)
        from scipy.special import gammaln
        log_fact_h = gammaln(hg + 1)
        log_fact_a = gammaln(ag + 1)

        # DC low-score masks (only 0-0 / 1-0 / 0-1 / 1-1 rows need tau correction)
        m00 = (hg == 0) & (ag == 0)
        m10 = (hg == 1) & (ag == 0)
        m01 = (hg == 0) & (ag == 1)
        m11 = (hg == 1) & (ag == 1)

        # L2 regularisation: blend FIFA rank priors with ELO-derived priors.
        # Pure ELO lets recent-tournament outliers (Morocco WC2022 run) dominate;
        # pure FIFA rank ignores current form. 55/45 blend captures both.
        # Non-WC teams fit freely as reference points.
        REG_LAMBDA = 7.0
        rank_map   = self._load_team_ranks()   # {canonical: FIFA-rank prior strength}

        elo_ratings      = self._compute_elo(all_matches)
        self.elo_ratings = elo_ratings

        # FIFA rank → log attack
        log_rank_atk = np.array(
            [np.log(rank_map.get(k, _rank_prior(35))) for k in team_keys]
        )
        # ELO → log attack: ±600 ELO pts ≈ ±1 log unit (moderate scale)
        log_elo_atk = np.array(
            [(elo_ratings.get(k, 1000.0) - 1000.0) / 600.0 for k in team_keys]
        )

        # Blended prior: 55% historical prestige, 45% recent form
        log_prior_atk = 0.55 * log_rank_atk + 0.45 * log_elo_atk
        log_prior_def = -log_prior_atk

        # Mask: 1.0 for WC teams (those with a FIFA rank entry)
        wc_mask = np.array(
            [1.0 if k in rank_map else 0.0 for k in team_keys]
        )

        def neg_ll(params: np.ndarray) -> float:
            atk = np.exp(params[:n])
            dfn = np.exp(params[n:2 * n])
            ha  = np.exp(params[2 * n])
            rho = params[2 * n + 1]

            # Apply home advantage only to non-neutral matches
            ha_mult = 1.0 + home_mask * (ha - 1.0)
            mu_h = atk[home_idx] * dfn[away_idx] * ha_mult
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

            ll = float(np.dot(weights, np.log(np.maximum(tau, 1e-12)) + log_p))

            # Regularisation toward blended priors, WC teams only (skip first — fixed at 0)
            reg = REG_LAMBDA * (
                float(np.sum(wc_mask[1:] * (params[1:n]     - log_prior_atk[1:]) ** 2)) +
                float(np.sum(wc_mask    * (params[n:2 * n]  - log_prior_def)     ** 2))
            )
            return -ll + reg

        # Warm-start from blended priors for faster, more stable convergence
        x0           = np.zeros(2 * n + 2)
        x0[:n]       = log_prior_atk
        x0[0]        = 0.0             # first team fixed for identifiability
        x0[n:2 * n]  = log_prior_def
        x0[2 * n]    = np.log(1.10)    # start at 10% home advantage for non-neutral games
        x0[2 * n + 1] = -0.1

        bounds = (
            [(0.0, 0.0)]                              # first team fixed for identifiability
            + [(-3.0, 3.0)] * (n - 1)
            + [(-3.0, 3.0)] * n
            + [(np.log(0.98), np.log(1.3))]           # home adv: 0.98–1.30, WC mostly neutral
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
            key: {
                "attack":  float(np.exp(log_atk[i])),
                "defense": float(np.exp(log_def[i])),
            }
            for i, key in enumerate(team_keys)
        }
        self._build_id_name_map()
        self._fitted = True

        # Compute form adjustments now that DC params are available
        self.form_adjustments = self._compute_form_adjustments(all_matches)

        return {
            "n_teams":        n,
            "n_wc_matches":   len(wc_matches),
            "n_wc2026":       len(wc2026_matches),
            "n_recent":       len(recent_matches),
            "n_total":        len(all_matches),
            "home_adv":       round(self.home_adv, 4),
            "rho":            round(self.rho, 4),
            "converged":      result.success,
            "message":        result.message,
            "n_form_teams":   len(self.form_adjustments),
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
        knockout: bool = False,
        max_goals: int = MAX_GOALS,
        draw_boost: float = 0.04,
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

        # Form adjustments: (attack_mult, defense_conceding_mult) per team.
        # h_atk_mult: home team scoring more/less than DC expects recently.
        # a_def_mult: away team conceding more/less than DC expects recently.
        # Both shift mu_h; symmetric logic applies to mu_a.
        h_atk_mult, h_def_mult = self.form_adjustments.get(home_key, (1.0, 1.0))
        a_atk_mult, a_def_mult = self.form_adjustments.get(away_key, (1.0, 1.0))

        mu_h = atk_h * h_atk_mult * def_a * a_def_mult * ha
        mu_a = atk_a * a_atk_mult * def_h * h_def_mult

        mu_h = min(mu_h, 6.0)
        mu_a = min(mu_a, 6.0)

        # Heavy-favourite dampening: when one team's expected goals > 2.5× the
        # other's, organised defending slightly suppresses the favourite's output.
        # Reduced from 0.90→0.95: group-stage data showed high-scoring blowouts
        # (Netherlands 5-1, Senegal 5-0) were systematically underpredicted.
        _ratio = mu_h / max(mu_a, 0.01)
        if _ratio > 2.5:
            mu_h *= 0.95
        elif _ratio < 1.0 / 2.5:
            mu_a *= 0.95

        goals = np.arange(max_goals + 1)
        mat   = np.outer(poisson.pmf(goals, mu_h), poisson.pmf(goals, mu_a))
        for x in range(2):
            for y in range(2):
                mat[x, y] *= max(_dc_tau(x, y, mu_h, mu_a, self.rho), 0.0)
        mat /= mat.sum()

        win_pct  = float(np.tril(mat, -1).sum()) * 100
        draw_pct = float(np.diag(mat).sum())      * 100
        loss_pct = float(np.triu(mat, 1).sum())   * 100

        if draw_boost > 0:
            boost_pts = draw_boost * 100
            d_new = min(draw_pct + boost_pts, 99.0)
            excess = d_new - draw_pct
            ha_total = win_pct + loss_pct
            if ha_total > 0:
                win_pct  -= excess * win_pct  / ha_total
                loss_pct -= excess * loss_pct / ha_total
            draw_pct = d_new

        flat = sorted(
            [(i, j, float(mat[i, j]) * 100)
             for i in range(max_goals + 1)
             for j in range(max_goals + 1)],
            key=lambda x: x[2], reverse=True,
        )

        result = {
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

        if knockout:
            # Redistribute draw probability: ET (30 min ≈ mu/3) then coin-flip pens
            et_goals = np.arange(max_goals + 1)
            et_mat = np.outer(poisson.pmf(et_goals, mu_h / 3.0), poisson.pmf(et_goals, mu_a / 3.0))
            et_mat /= et_mat.sum()
            p_et_home = float(np.tril(et_mat, -1).sum())
            p_et_away = float(np.triu(et_mat, 1).sum())
            p_et_draw = float(np.diag(et_mat).sum())
            d = draw_pct / 100.0
            result["ko_win_pct"]  = round(win_pct  + d * (p_et_home + p_et_draw * 0.5) * 100, 1)
            result["ko_loss_pct"] = round(loss_pct + d * (p_et_away + p_et_draw * 0.5) * 100, 1)

        return result

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
        results = []
        for home_id, away_id in rows:
            home_key = self._resolve_name(home_id)
            is_host = home_key in {"usa", "mexico", "canada"}
            results.append(self.predict(home_id, away_id, home_advantage=is_host))
        return results

    # ── Tournament simulator (vectorised) ───────────────────────────────────

    # WC2026 confirmed group draw (same as in fantasy_optimizer)
    WC2026_GROUPS: dict[str, list[str]] = {
        "A": ["Mexico",        "South Korea",  "South Africa",          "Czech Republic"],
        "B": ["Canada",        "Switzerland",  "Qatar",                 "Bosnia and Herzegovina"],
        "C": ["Brazil",        "Morocco",      "Scotland",              "Haiti"],
        "D": ["United States", "Australia",    "Paraguay",              "Turkey"],
        "E": ["Germany",       "Curaçao",      "Ivory Coast",           "Ecuador"],
        "F": ["Netherlands",   "Japan",        "Tunisia",               "Sweden"],
        "G": ["Belgium",       "Iran",         "Egypt",                 "New Zealand"],
        "H": ["Spain",         "Uruguay",      "Saudi Arabia",          "Cape Verde"],
        "I": ["France",        "Senegal",      "Norway",                "Iraq"],
        "J": ["Argentina",     "Austria",      "Algeria",               "Jordan"],
        "K": ["Portugal",      "Colombia",     "Uzbekistan",            "DR Congo"],
        "L": ["England",       "Croatia",      "Panama",                "Ghana"],
    }

    def _sim_match_goals(
        self, home_key: str, away_key: str,
        rng: np.random.Generator, home_advantage: bool = False,
    ) -> tuple[int, int]:
        """Sample one match result from the DC-Poisson model."""
        atk_h, def_h = self._team_atk_def(home_key)
        atk_a, def_a = self._team_atk_def(away_key)
        ha = self.home_adv if home_advantage else 1.0
        h_atk_mult, h_def_mult = self.form_adjustments.get(home_key, (1.0, 1.0))
        a_atk_mult, a_def_mult = self.form_adjustments.get(away_key, (1.0, 1.0))
        mu_h = atk_h * h_atk_mult * def_a * a_def_mult * ha
        mu_a = atk_a * a_atk_mult * def_h * h_def_mult
        _ratio = mu_h / max(mu_a, 0.01)
        if _ratio > 2.5:
            mu_h *= 0.95
        elif _ratio < 1.0 / 2.5:
            mu_a *= 0.95
        hg = int(rng.poisson(mu_h))
        ag = int(rng.poisson(mu_a))
        # DC low-score correction: accept/reject via tau
        tau = _dc_tau(hg, ag, mu_h, mu_a, self.rho)
        if tau < 0:
            tau = 0.0
        # Re-sample the specific outcome if needed (importance weight via bernoulli)
        # Simple approach: adjust by tau proportionally (works since tau ≈ 1 for most)
        if rng.random() > tau:
            hg = int(rng.poisson(mu_h))
            ag = int(rng.poisson(mu_a))
        return hg, ag

    def _sim_knockout_match(
        self, home_key: str, away_key: str, rng: np.random.Generator,
    ) -> str:
        """Simulate a knockout match (ET + pens if draw). Returns winner key."""
        hg, ag = self._sim_match_goals(home_key, away_key, rng, home_advantage=False)
        if hg != ag:
            return home_key if hg > ag else away_key
        # Extra time: ~30 min ≈ 1/3 of normal time lambda
        atk_h, def_h = self._team_atk_def(home_key)
        atk_a, def_a = self._team_atk_def(away_key)
        et_h = int(rng.poisson(atk_h * def_a / 3.0))
        et_a = int(rng.poisson(atk_a * def_h / 3.0))
        if et_h != et_a:
            return home_key if et_h > et_a else away_key
        # Penalty shootout: coin flip
        return home_key if rng.random() < 0.5 else away_key

    def _sim_group_stage(
        self, rng: np.random.Generator,
    ) -> tuple[list[str], list[tuple[str, int, int, int]]]:
        """
        Simulate all 12 groups.  Returns:
          qualified: list[str] canonical team keys — 24 group tops/runners-up
                     + best 8 3rd-place teams  (32 total)
          third_place_table: [(team_key, pts, gd, gf), ...]  all 12 3rd-place teams
        """
        top2: list[str] = []
        thirds: list[tuple[str, int, int, int]] = []

        for group_name, teams in self.WC2026_GROUPS.items():
            keys = [_canonical(t) for t in teams]
            # Track (pts, gd, gf) per team
            stats: dict[str, list[int]] = {k: [0, 0, 0] for k in keys}

            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    hk, ak = keys[i], keys[j]
                    hg, ag = self._sim_match_goals(hk, ak, rng)
                    diff = hg - ag
                    if diff > 0:
                        stats[hk][0] += 3
                    elif diff < 0:
                        stats[ak][0] += 3
                    else:
                        stats[hk][0] += 1
                        stats[ak][0] += 1
                    stats[hk][1] += diff;  stats[ak][1] -= diff
                    stats[hk][2] += hg;    stats[ak][2] += ag

            ranked = sorted(keys, key=lambda k: (stats[k][0], stats[k][1], stats[k][2]), reverse=True)
            top2.extend(ranked[:2])
            thirds.append((ranked[2], stats[ranked[2]][0], stats[ranked[2]][1], stats[ranked[2]][2]))

        # Best 8 third-place teams
        thirds.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)
        qualified_thirds = [t[0] for t in thirds[:8]]

        return top2 + qualified_thirds, thirds

    def _build_r32_bracket(self, qualified: list[str]) -> list[tuple[str, str]]:
        """
        Pair the 32 qualifiers into 16 R32 matches.

        Layout mirrors the official WC2026 bracket:
        - First 24 entries are ordered: [1A,2A, 1B,2B, …, 1L,2L]
          (as returned by _sim_group_stage's top2 list, pair-by-pair per group)
        - Entries 24-31 are the 8 best 3rd-place teams (seeded 25-32)

        Standard seeding: 1 vs 32, 2 vs 31 … preserving opposite-half separation.
        """
        n = len(qualified)  # should be 32
        # Simple seeded bracket: top seeds meet bottom seeds
        pairs: list[tuple[str, str]] = []
        for i in range(n // 2):
            pairs.append((qualified[i], qualified[n - 1 - i]))
        return pairs

    @staticmethod
    def _guaranteed_1st(group_keys: list[str], played_map: dict) -> set[int]:
        """Return positions (0-3) mathematically guaranteed to finish 1st.
        Conservative: a team is marked guaranteed-1st only when no other team
        can accumulate strictly more points in any remaining-game scenario.
        Does not model H2H tiebreakers — safe because we only call this when
        the pts gap is genuinely unbeatable (e.g. Mexico/USA/Germany at 6pts
        with only H2H rivals able to reach 6pts and H2H already won).
        """
        from itertools import product as iproduct
        n = len(group_keys)
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

        base_pts = [0] * n
        remaining = []
        for i, j in all_pairs:
            hi, ai = group_keys[i], group_keys[j]
            if (hi, ai) in played_map:
                hg, ag = played_map[(hi, ai)]
            elif (ai, hi) in played_map:
                ag, hg = played_map[(ai, hi)]
            else:
                remaining.append((i, j))
                continue
            diff = hg - ag
            if diff > 0:   base_pts[i] += 3
            elif diff < 0: base_pts[j] += 3
            else:          base_pts[i] += 1; base_pts[j] += 1

        guaranteed = set()
        for t in range(n):
            always_1st = True
            for outcomes in iproduct(range(3), repeat=len(remaining)):
                pts = list(base_pts)
                for k, (i, j) in enumerate(remaining):
                    o = outcomes[k]
                    if o == 0:   pts[i] += 3
                    elif o == 1: pts[i] += 1; pts[j] += 1
                    else:        pts[j] += 3
                if any(pts[u] > pts[t] for u in range(n) if u != t):
                    always_1st = False
                    break
            if always_1st:
                guaranteed.add(t)
        return guaranteed

    @staticmethod
    def _guaranteed_top2(group_keys: list[str], played_map: dict) -> set[int]:
        """Return positions (0-3) that are mathematically guaranteed to finish top-2.

        Brute-forces all possible remaining game outcomes (3^R combinations where
        R ≤ 6). Conservative: uses strict pts comparison only (ignores H2H
        tiebreakers), so a team is only marked guaranteed if they can't be leapfrogged
        on pure points in any scenario.
        """
        from itertools import product as iproduct
        n = len(group_keys)
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

        base_pts = [0] * n
        remaining = []
        for i, j in all_pairs:
            hi, ai = group_keys[i], group_keys[j]
            if (hi, ai) in played_map:
                hg, ag = played_map[(hi, ai)]
            elif (ai, hi) in played_map:
                ag, hg = played_map[(ai, hi)]
            else:
                remaining.append((i, j))
                continue
            diff = hg - ag
            if diff > 0:   base_pts[i] += 3
            elif diff < 0: base_pts[j] += 3
            else:          base_pts[i] += 1; base_pts[j] += 1

        guaranteed = set()
        for t in range(n):
            always_top2 = True
            for outcomes in iproduct(range(3), repeat=len(remaining)):
                pts = list(base_pts)
                for k, (i, j) in enumerate(remaining):
                    o = outcomes[k]
                    if o == 0:   pts[i] += 3
                    elif o == 1: pts[i] += 1; pts[j] += 1
                    else:        pts[j] += 3
                if sum(1 for u in range(n) if u != t and pts[u] > pts[t]) >= 2:
                    always_top2 = False
                    break
            if always_top2:
                guaranteed.add(t)
        return guaranteed

    def simulate_tournament(self, n_sim: int = 50_000) -> list[dict]:
        """
        Monte Carlo simulation of the full WC2026 tournament.

        Fully vectorised: all 72 group-stage match goals are sampled in two
        numpy calls; knockout rounds are batched per-round across n_sim.
        Typical runtime: <1 s for n_sim=50 000.

        Returns a list of dicts (one per team) sorted by win probability.
        """
        if not self._fitted and not self.team_params:
            raise RuntimeError("Model not loaded. Call load() first.")

        rng = np.random.default_rng(42)

        # ── Team index (48 WC2026 teams) ──────────────────────────────────────
        all_teams  = [t for teams in self.WC2026_GROUPS.values() for t in teams]
        all_keys   = [_canonical(t) for t in all_teams]
        T          = len(all_keys)
        key_to_idx = {k: i for i, k in enumerate(all_keys)}

        mean_atk = float(np.mean([v["attack"]  for v in self.team_params.values()]))
        mean_def = float(np.mean([v["defense"] for v in self.team_params.values()]))

        atk_arr = np.array([self.team_params.get(k, {"attack":  mean_atk})["attack"]  for k in all_keys])
        def_arr = np.array([self.team_params.get(k, {"defense": mean_def})["defense"] for k in all_keys])

        # lam[i, j] = expected goals scored by team i vs team j
        lam = np.outer(atk_arr, def_arr)  # (T, T)

        # ── Build group match index (72 matches) ──────────────────────────────
        group_lists: list[list[int]] = []
        gm_h_lst, gm_a_lst = [], []
        gm_g_lst, gm_hp_lst, gm_ap_lst = [], [], []

        for g_idx, g_teams in enumerate(self.WC2026_GROUPS.values()):
            g_idxs = [key_to_idx[_canonical(t)] for t in g_teams]
            group_lists.append(g_idxs)
            for pi in range(4):
                for pj in range(pi + 1, 4):
                    gm_h_lst.append(g_idxs[pi]); gm_a_lst.append(g_idxs[pj])
                    gm_g_lst.append(g_idx);       gm_hp_lst.append(pi); gm_ap_lst.append(pj)

        M    = len(gm_h_lst)  # 72
        gm_h = np.array(gm_h_lst, dtype=np.int32)
        gm_a = np.array(gm_a_lst, dtype=np.int32)

        mu_h_gm = lam[gm_h, gm_a]   # (72,)
        mu_a_gm = lam[gm_a, gm_h]   # (72,)

        # ── Sample all n_sim × 72 group goals in two calls ────────────────────
        hg_all = rng.poisson(np.tile(mu_h_gm, (n_sim, 1)))  # (n_sim, 72)
        ag_all = rng.poisson(np.tile(mu_a_gm, (n_sim, 1)))  # (n_sim, 72)

        # ── Pin already-played group matches to actual results ────────────────
        conn = sqlite3.connect(DB_PATH)
        played_rows = conn.execute(
            "SELECT home_team, away_team, home_score, away_score FROM match_stats"
        ).fetchall()
        conn.close()
        played_map = {
            (_canonical(h), _canonical(a)): (int(hs), int(as_))
            for h, a, hs, as_ in played_rows
            if hs is not None and as_ is not None
        }

        # Detect teams mathematically guaranteed to finish top-2 or 1st.
        # Locked to exactly 100% after Monte Carlo — facts override predictions.
        guaranteed_r32: set[int] = set()
        guaranteed_1st: set[int] = set()
        for g_idx, g_teams in enumerate(self.WC2026_GROUPS.values()):
            g_keys = [_canonical(t) for t in g_teams]
            for pos in self._guaranteed_top2(g_keys, played_map):
                guaranteed_r32.add(key_to_idx[g_keys[pos]])
            for pos in self._guaranteed_1st(g_keys, played_map):
                guaranteed_1st.add(key_to_idx[g_keys[pos]])

        for m in range(M):
            h_key = all_keys[gm_h[m]]
            a_key = all_keys[gm_a[m]]
            if (h_key, a_key) in played_map:
                hg, ag = played_map[(h_key, a_key)]
                hg_all[:, m] = hg
                ag_all[:, m] = ag
            elif (a_key, h_key) in played_map:
                # Actual fixture had teams in opposite order vs group-list enumeration.
                # played_map stores (actual_home_goals, actual_away_goals), so swap for
                # the simulation's perspective where h_key is the simulated "home".
                actual_home_g, actual_away_g = played_map[(a_key, h_key)]
                hg_all[:, m] = actual_away_g   # h_key was the actual away team
                ag_all[:, m] = actual_home_g   # a_key was the actual home team

        # ── Accumulate group standings ────────────────────────────────────────
        pts_arr = np.zeros((n_sim, 12, 4), dtype=np.int32)
        gd_arr  = np.zeros((n_sim, 12, 4), dtype=np.int32)
        gf_arr  = np.zeros((n_sim, 12, 4), dtype=np.int32)
        # Pairwise H2H: h2h_pts_arr[s,g,i,j] = pts team-i earned in their game vs team-j.
        # Each pair plays exactly once, so we assign (not +=).
        h2h_pts_arr = np.zeros((n_sim, 12, 4, 4), dtype=np.int32)
        h2h_gd_arr  = np.zeros((n_sim, 12, 4, 4), dtype=np.int32)

        for m in range(M):
            g, hp, ap = gm_g_lst[m], gm_hp_lst[m], gm_ap_lst[m]
            h_goals   = hg_all[:, m]
            a_goals   = ag_all[:, m]
            diff      = (h_goals - a_goals).astype(np.int32)
            h_pts = np.where(diff > 0, 3, np.where(diff == 0, 1, 0)).astype(np.int32)
            a_pts = np.where(diff < 0, 3, np.where(diff == 0, 1, 0)).astype(np.int32)
            pts_arr[:, g, hp] += h_pts;  pts_arr[:, g, ap] += a_pts
            gd_arr[:, g, hp]  += diff;   gd_arr[:, g, ap]  -= diff
            gf_arr[:, g, hp]  += h_goals.astype(np.int32)
            gf_arr[:, g, ap]  += a_goals.astype(np.int32)
            h2h_pts_arr[:, g, hp, ap] = h_pts
            h2h_pts_arr[:, g, ap, hp] = a_pts
            h2h_gd_arr[:, g, hp, ap]  = diff
            h2h_gd_arr[:, g, ap, hp]  = -diff

        del hg_all, ag_all  # free ~60 MB before ranking arrays are built

        # ── Rank teams within each group (FIFA tiebreaker hierarchy) ─────────
        # 1. Overall points
        # 2. H2H points among tied teams only   ← FIFA rule, not GD
        # 3. H2H GD among tied teams only
        # 4. Overall GD
        # 5. Overall GF
        #
        # Vectorised H2H: for each team i, mask to only opponents tied on pts,
        # then sum h2h_pts/gd over those opponents. This is exact for 2-team
        # ties (most common) and correct for 3-team circular ties that stay
        # level after H2H pts/GD.
        pts_i    = pts_arr.astype(np.int64)[:, :, :, np.newaxis]   # (n_sim,12,4,1)
        pts_j    = pts_arr.astype(np.int64)[:, :, np.newaxis, :]   # (n_sim,12,1,4)
        tied     = (pts_i == pts_j).astype(np.int32)               # (n_sim,12,4,4)

        h2h_pts_tied = (h2h_pts_arr.astype(np.int64) * tied).sum(axis=-1)  # (n_sim,12,4)
        h2h_gd_tied  = (h2h_gd_arr.astype(np.int64)  * tied).sum(axis=-1)  # (n_sim,12,4)
        del h2h_pts_arr, h2h_gd_arr, tied, pts_i, pts_j

        OFFSET = 50   # shift negative GD values to keep sort key positive
        sort_key = (pts_arr.astype(np.int64) * 100_000_000
                    + h2h_pts_tied            *  10_000_000
                    + (h2h_gd_tied + OFFSET)  *     100_000
                    + (gd_arr.astype(np.int64) + OFFSET) * 1_000
                    + gf_arr.astype(np.int64))              # (n_sim, 12, 4)
        del h2h_pts_tied, h2h_gd_tied
        ranks = np.argsort(-sort_key, axis=-1)              # (n_sim, 12, 4)

        group_arrs = [np.array(g, dtype=np.int32) for g in group_lists]

        # top2_idx[s, g, r] = global team index of rank-r team in group g for sim s
        top2_idx = np.stack(
            [group_arrs[g][ranks[:, g, :2]] for g in range(12)], axis=1
        )  # (n_sim, 12, 2)

        third_pos = ranks[:, :, 2]  # (n_sim, 12) — group position of 3rd-place team
        third_idx = np.stack(
            [group_arrs[g][third_pos[:, g]] for g in range(12)], axis=1
        )  # (n_sim, 12)

        sim_idx   = np.arange(n_sim)[:, None]  # (n_sim, 1)
        grp_idx   = np.arange(12)[None, :]     # (1, 12)
        third_pts = pts_arr[sim_idx, grp_idx, third_pos]
        third_gd  = gd_arr[ sim_idx, grp_idx, third_pos]
        third_gf  = gf_arr[ sim_idx, grp_idx, third_pos]

        # Best 8 3rd-place teams per sim
        third_sort  = (third_pts.astype(np.int64) * 1_000_000
                       + third_gd.astype(np.int64) * 1_000
                       + third_gf.astype(np.int64))             # (n_sim, 12)
        best8_local     = np.argsort(-third_sort, axis=-1)[:, :8]  # (n_sim, 8)
        best8_team_idx  = third_idx[np.arange(n_sim)[:, None], best8_local]  # (n_sim, 8)

        r32_teams = np.concatenate(
            [top2_idx.reshape(n_sim, 24), best8_team_idx], axis=1
        )  # (n_sim, 32)

        # Save 1st-place finisher per group per sim before freeing top2_idx
        first_idx = top2_idx[:, :, 0].copy()  # (n_sim, 12)

        del pts_arr, gd_arr, gf_arr, sort_key, ranks, top2_idx, third_idx

        # ── Count stage qualifications ────────────────────────────────────────
        r32_counts   = np.zeros(T, dtype=np.int64)
        first_counts = np.zeros(T, dtype=np.int64)
        r16_counts   = np.zeros(T, dtype=np.int64)
        qf_counts    = np.zeros(T, dtype=np.int64)
        sf_counts    = np.zeros(T, dtype=np.int64)
        final_counts = np.zeros(T, dtype=np.int64)
        win_counts   = np.zeros(T, dtype=np.int64)

        np.add.at(r32_counts,   r32_teams.ravel(), 1)
        np.add.at(first_counts, first_idx.ravel(), 1)

        # Lock guaranteed qualifiers / group winners to exactly 100%.
        for gidx in guaranteed_r32:
            r32_counts[gidx] = n_sim
        for gidx in guaranteed_1st:
            first_counts[gidx] = n_sim

        # ── Bracket: use confirmed fixtures when group stage is complete ─────
        all_group_matches_played = all(
            (all_keys[gm_h[m]], all_keys[gm_a[m]]) in played_map
            or (all_keys[gm_a[m]], all_keys[gm_h[m]]) in played_map
            for m in range(M)
        )
        if all_group_matches_played:
            # All 32 teams confirmed — use the official drawn R32 fixtures so
            # each team's path probabilities reflect their actual opponent.
            _r32_order = [
                "South Africa",          "Canada",
                "Netherlands",           "Morocco",
                "Germany",               "Paraguay",
                "France",                "Sweden",
                "Portugal",              "Croatia",
                "Spain",                 "Austria",
                "United States",         "Bosnia and Herzegovina",
                "Belgium",               "Senegal",
                "Brazil",                "Japan",
                "Ivory Coast",           "Norway",
                "Mexico",                "Ecuador",
                "England",               "DR Congo",
                "Argentina",             "Cape Verde",
                "Australia",             "Egypt",
                "Switzerland",           "Algeria",
                "Colombia",              "Ghana",
            ]
            bracket_idx = np.array(
                [key_to_idx[_canonical(t)] for t in _r32_order], dtype=np.int32
            )
            current = np.tile(bracket_idx, (n_sim, 1))  # (n_sim, 32)
        else:
            # Group stage in progress — seed by DC quality (1 vs 32, 2 vs 31…)
            quality     = atk_arr / def_arr
            r32_quality = quality[r32_teams]
            seed_order  = np.argsort(-r32_quality, axis=-1)
            seeded      = r32_teams[np.arange(n_sim)[:, None], seed_order]
            bracket_order = []
            for i in range(16):
                bracket_order.extend([i, 31 - i])
            current = seeded[:, bracket_order]

        # ── Simulate 5 knockout rounds (vectorised per-round) ─────────────────
        # R32 (32→16): r16_counts tracks who won R32 and plays in R16
        # R16 (16→8): qf_pct; QF (8→4): sf_pct; SF (4→2): final_pct; F (2→1): win_pct
        for cnt_arr in (r16_counts, qf_counts, sf_counts, final_counts, win_counts):
            h_mat = current[:, 0::2]  # (n_sim, n_matches) home teams
            a_mat = current[:, 1::2]  # (n_sim, n_matches) away teams

            mu_h = lam[h_mat, a_mat]   # (n_sim, n_matches)
            mu_a = lam[a_mat, h_mat]

            hg = rng.poisson(mu_h)
            ag = rng.poisson(mu_a)

            draw = hg == ag
            if draw.any():
                et_h = rng.poisson(mu_h[draw] / 3.0)
                et_a = rng.poisson(mu_a[draw] / 3.0)
                sd   = et_h == et_a
                if sd.any():
                    coin     = rng.random(int(sd.sum())) < 0.5
                    et_h[sd] = coin.astype(np.int64)
                    et_a[sd] = (~coin).astype(np.int64)
                hg[draw] += et_h
                ag[draw] += et_a

            winners = np.where(hg > ag, h_mat, a_mat)  # (n_sim, n_matches)

            # Pin played knockout results: when all sims agree on the participants
            # (i.e. both teams are known/confirmed), override with actual result.
            # Penalty shootouts stored as equal scores are left as simulated (~50/50).
            n_matches = h_mat.shape[1]
            for mi in range(n_matches):
                h0, a0 = int(h_mat[0, mi]), int(a_mat[0, mi])
                if not (np.all(h_mat[:, mi] == h0) and np.all(a_mat[:, mi] == a0)):
                    continue  # different sims disagree on participants — skip
                h_key, a_key = all_keys[h0], all_keys[a0]
                if (h_key, a_key) in played_map:
                    hg_r, ag_r = played_map[(h_key, a_key)]
                    if hg_r != ag_r:
                        winners[:, mi] = h0 if hg_r > ag_r else a0
                elif (a_key, h_key) in played_map:
                    ag_r, hg_r = played_map[(a_key, h_key)]
                    if hg_r != ag_r:
                        winners[:, mi] = a0 if ag_r > hg_r else h0

            if cnt_arr is not None:
                np.add.at(cnt_arr, winners.ravel(), 1)
            current = winners  # advance; shape halves each round

        # ── Build results ─────────────────────────────────────────────────────
        group_of = {
            _canonical(t): g
            for g, teams in self.WC2026_GROUPS.items()
            for t in teams
        }
        name_map = {_canonical(t): t for t in all_teams}

        results = [
            {
                "team":           name_map.get(k, k),
                "group":          group_of.get(k, "?"),
                "group_1st_pct":  round(first_counts[i]  / n_sim * 100, 1),
                "r32_pct":        round(r32_counts[i]    / n_sim * 100, 1),
                "r16_pct":        round(r16_counts[i]    / n_sim * 100, 1),
                "qf_pct":         round(qf_counts[i]     / n_sim * 100, 1),
                "sf_pct":         round(sf_counts[i]     / n_sim * 100, 1),
                "final_pct":      round(final_counts[i]  / n_sim * 100, 1),
                "win_pct":        round(win_counts[i]    / n_sim * 100, 1),
            }
            for i, k in enumerate(all_keys)
        ]
        results.sort(key=lambda x: x["win_pct"], reverse=True)
        return results

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
        import datetime as _dt
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        MODEL_PATH.write_text(json.dumps(
            {
                "trained_at":       _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "home_adv":         self.home_adv,
                "rho":              self.rho,
                "team_params":      self.team_params,
                "id_to_name":       {str(k): v for k, v in self._id_to_name.items()},
                "elo_ratings":      self.elo_ratings,
                "form_adjustments": {k: list(v) for k, v in self.form_adjustments.items()},
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
        self.home_adv         = data["home_adv"]
        self.rho              = data["rho"]
        self.team_params      = data["team_params"]
        self._id_to_name      = {int(k): v for k, v in data.get("id_to_name", {}).items()}
        self.trained_at       = data.get("trained_at")
        self.elo_ratings      = data.get("elo_ratings", {})
        self.form_adjustments = {k: tuple(v) for k, v in data.get("form_adjustments", {}).items()}
        self._fitted          = True
