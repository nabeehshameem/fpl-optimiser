"""
fantasy_optimizer.py
Projects fantasy points per player and optimises squad selection.

Point projection uses Monte Carlo simulation (N=50,000 by default):
  - Draws team goals from Poisson distributions using actual WC2026 group
    fixtures and opponent-specific DC attack/defence ratings.
  - Averages binary events (clean sheets, conceding) across all simulations
    so projected points reflect real schedule difficulty, not just mean opponent.
  - Adds expected Qualification Booster value (+2 x P(team advances from R32)).

Squad optimization uses scipy.optimize.milp (binary MILP).

The MILP models three binary variable sets per player:
  x_i  in squad (15 players)
  s_i  in starting XI (11 of the 15)
  c_i  captain (1 of the 11)

Objective: maximise sum(s_i * pts_i) + sum(c_i * pts_i)
  — bench slots (x_i=1, s_i=0) contribute nothing to the objective, so the
    solver naturally fills them with the cheapest eligible players, freeing
    budget for higher-quality starters.

WC Fantasy rules enforced:
  Squad:         2 GK + 5 DEF + 5 MID + 3 FWD = 15 players
  Starting XI:   1 GK, 3-5 DEF, 3-5 MID, 1-3 FWD (all valid WC formations)
  Budget:        $100m (1000 in $0.1m units)
  Team cap:      max 3 players per country (group stage)
"""

import json
import sqlite3
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH      = PROJECT_ROOT / "data" / "wc.db"
MODEL_PATH   = PROJECT_ROOT / "models" / "dc_params.json"

SQUAD_RULES    = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
BUDGET_DEFAULT = 1000  # $100.0m

# WC Fantasy 2026 scoring
PT_APPEARANCE = 1.8   # expected pts/match: 1pt any app + ~0.8 chance of 60+ min bonus
PT_CS      = {"GK": 5.0, "DEF": 5.0, "MID": 1.0, "FWD": 0.0}
PT_GOAL    = {"GK": 9.0, "DEF": 7.0, "MID": 6.0, "FWD": 5.0}
PT_ASSIST  = 3.0
PT_SAVE3   = 1.0   # per 3 saves (GK only)
PT_CONCEDE = -1.0  # per goal beyond the first (0 for 1 goal, -1 for 2, -2 for 3...)
# Rough stat bonuses per group stage (3 matches): MID tackles/chances, FWD shots
PT_STAT_BONUS = {"GK": 0.0, "DEF": 0.0, "MID": 1.5, "FWD": 1.0}
# Scouting bonus (+2 if >4pts AND <5% ownership) — ownership data unavailable
# pre-tournament; differentials already rewarded via higher pts/$ ratio in the MILP.

GOAL_SHARE   = {"GK": 0.01, "DEF": 0.07, "MID": 0.30, "FWD": 0.62}
ASSIST_SHARE = {"GK": 0.01, "DEF": 0.12, "MID": 0.50, "FWD": 0.37}
ASSIST_RATIO = 0.85

# Players confirmed unavailable for specific matchdays.
# Keys are lowercase substrings of the player name; values are sets of MD numbers missed.
# Update as injury/suspension news changes before each matchday.
PLAYER_UNAVAILABLE: dict[str, set[int]] = {
    # yamal/williams: not in unavailable — de la Fuente says "could be ready June 15" but start not guaranteed; handled via starter_prob 0.60
    # messi: Grade 1 strain only, expected fully fit for Argentina opener June 16, removed from unavailable
    "neymar": {1},     # Grade 2 calf (May 17), race for MD1 vs Morocco June 13 — user: maybe back MD2
    "davies": {1},     # ACL (March) + hamstring (May, 3rd injury of season) — MD1 very much in doubt
}

# Probability that a player starts (gets 60+ min) in a given match.
# Default = 1.0 (no discount). Apply to squad fillers, rotation players,
# and anyone with significant competition for their starting spot.
# Keys are lowercase substrings of the player name.
PLAYER_STARTER_PROB: dict[str, float] = {
    # ── Argentina ────────────────────────────────────────────────────────
    "messi":        0.90,  # Grade 1 strain only — expected fully fit, guaranteed starter when healthy
    "soul":         0.50,  # Soulé — young, not yet a regular starter
    "lo celso":     0.45,  # squad rotational player
    "barco":        0.40,  # fringe squad
    "buend":        0.55,  # competes with De Paul / Mac Allister
    "romero":       0.25,  # partially torn MCL (mid-April), World Cup in jeopardy

    # ── Germany ──────────────────────────────────────────────────────────
    "sané":         0.62,  # rotation risk — Nagelsmann XI leans on Musiala/Wirtz; Sané not guaranteed starter
    "goretzka":     0.55,  # deep rotation behind Musiala/Wirtz/Kimmich
    "leweling":     0.60,  # competes for wide role

    # ── England ──────────────────────────────────────────────────────────
    "mainoo":       0.65,  # competes with Rice/Bellingham/Foden
    "rogers":       0.60,  # squad rotation
    "gordon":       0.60,  # competes with Saka/Palmer
    "eze":          0.65,  # competes with Palmer/Foden
    "palmer":       0.70,  # competes with Eze/Foden

    # ── Spain ────────────────────────────────────────────────────────────
    "yamal":        0.60,  # hamstring — de la Fuente: "could be ready June 15 but doesn't guarantee he'll play"
    "williams":     0.60,  # same coach quote applies — MD1 start not guaranteed
    "olmo":         0.75,  # competes with Pedri when both fit
    "gavi":         0.70,  # returning from injury, competes with Zubimendi
    "merino":       0.35,  # stress fracture in foot (Feb), targeting CL final return — fitness risk
    "zubimendi":    0.75,  # rotates with Rodri

    # ── France ───────────────────────────────────────────────────────────
    "cherki":       0.65,  # young, competes for wide/10 role
    "dou":          0.70,  # Doué — rotation

    # ── Brazil ───────────────────────────────────────────────────────────
    "neymar":       0.60,  # injury history, fitness uncertainty at 34
    "casemiro":     0.75,  # aging, some rotation risk

    # ── Norway ───────────────────────────────────────────────────────────
    "sorloth":      0.60,  # backup striker to Haaland
    "stig":          0.50,  # L. Østigård — Norway 3rd in group; tough France/Senegal fixtures
    "nusa":         0.65,  # young, rotation with Ødegaard/Aursnes

    # ── Portugal ─────────────────────────────────────────────────────────
    "ronaldo":      0.75,  # 41 years old in 2026; still likely starts but rotation risk

    # ── Belgium ──────────────────────────────────────────────────────────
    "tielemans":    0.75,  # rotates in Belgium's evolving midfield

    # ── New Zealand ───────────────────────────────────────────────────────
    "o. sail":      0.70,  # NZ #1 GK — save bonus vs Belgium overstated; model discount applied

    # ── Netherlands ──────────────────────────────────────────────────────
    "timber":       0.40,  # groin injury, no game since March 14 — fitness uncertain
    "koopmeiners":  0.70,  # competes with Gravenberch/Reijnders
    "madueke":      0.65,  # wide rotation
    "schouten":     0.65,  # DM rotation

    # ── Ghana ────────────────────────────────────────────────────────────
    "kudus":        0.35,  # quad injury Jan + hamstring setback Apr, WC participation in doubt

    # ── Morocco ──────────────────────────────────────────────────────────
    "hakimi":       0.80,  # hamstring (PSG CL semi), expected to recover in time

    # ── Turkey ───────────────────────────────────────────────────────────
    "güler":   0.75,  # pulled hamstring (April), on track to recover for tournament

    # ── Croatia ──────────────────────────────────────────────────────────
    "modri":        0.80,  # cheekbone fracture, shut down for season but confident of recovery

    # ── Canada ───────────────────────────────────────────────────────────
    "davies":       0.55,  # ACL (March) + hamstring (May), day-by-day rehab

    # ── Argentina / general bench GKs ────────────────────────────────────
    "beiranvand":   0.90,  # Iran starter GK — high prob but noted for completeness
}

# Extra projected points awarded to players with a confirmed set-piece role.
# Values are over the full 3-match group stage; _project_mc scales by (n_m / 3).
#
# Penalty taker formula: 1.2 PKs awarded per team × 0.75 conversion × goal_pts[pos]
#   FWD (5 pts/goal) → +4.5 pts   MID (6 pts/goal) → +5.4 pts
# Corner / FK taker:  extra assist flow from delivering dead balls → +0.8–1.5 pts
#
# Scouting bonus (+2 if >4 pts AND <5% ownership) is not modelled — ownership data
# is unavailable pre-tournament.  Differentials are already indirectly rewarded by
# the MILP's pts/$ objective (higher pts per unit of budget = preferred).
PLAYER_SETPIECE_BONUS: dict[str, float] = {
    # ── Penalty takers ───────────────────────────────────────────────────
    "messi":      4.5,   # ARG
    "mbapp":      4.5,   # FRA — K. Mbappé (accent-safe prefix key)
    "haaland":    4.5,   # NOR
    "ronaldo":    4.5,   # POR — C. Ronaldo still designated taker
    "kane":       4.5,   # ENG
    "havertz":    4.5,   # GER — confirmed FWD/PK role
    "lukaku":     4.5,   # BEL
    "vinicius":   4.5,   # BRA — first-choice taker over Rodrygo

    # ── Corner / FK takers ───────────────────────────────────────────────
    "de bruyne":  1.5,   # BEL — primary FK delivery
    "degaard":    1.5,   # NOR — M. Ødegaard (Ø prefix skipped; "degaard" matches)
    "bellingham": 1.0,   # ENG
    "güler":      1.5,   # TUR — FK specialist
    "de paul":    1.0,   # ARG
    "pedri":      1.0,   # ESP
    "hakimi":     0.8,   # MAR — overlapping FK role
}

# Confirmed WC2026 group draw (official, April 2026)
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


def _canonical(name: str) -> str:
    aliases = {
        "united states":                    "usa",
        "south korea":                      "korea republic",
        "iran":                             "ir iran",
        "china":                            "china pr",
        "cape verde":                       "cape verde islands",
        "democratic republic of the congo": "dr congo",
        "bosnia & herzegovina":             "bosnia and herzegovina",
    }
    n = name.lower().strip()
    return aliases.get(n, n)


def _load_players(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("""
        SELECT fp.fantasy_id, fp.name, fp.position, fp.price,
               t.name AS team_name
        FROM   fantasy_players fp
        LEFT JOIN teams t ON fp.team_id = t.team_id
        WHERE  fp.position IS NOT NULL AND fp.price IS NOT NULL AND fp.price > 0
    """).fetchall()
    return [
        {"id": r[0], "name": r[1], "pos": r[2], "price": r[3], "team": r[4] or "Unknown"}
        for r in rows
    ]


def _load_dc() -> dict:
    if not MODEL_PATH.exists():
        return {}
    return json.loads(MODEL_PATH.read_text())


def _get_qual_probs(predictor=None) -> dict[str, dict]:
    """
    Returns {canonical_team: {qf_pct, sf_pct, final_pct, win_pct}} from a fast
    tournament simulation. qf_pct = P(team wins their R32 match and reaches R16).
    Used to compute expected Qualification Booster value per player.
    """
    try:
        if predictor is None:
            from wc.src.score_predictor import DCPredictor
            p = DCPredictor()
            p.load()
            predictor = p
        results = predictor.simulate_tournament(n_sim=10_000)
        return {
            _canonical(r["team"]): {
                "qf_pct":    r["qf_pct"],
                "sf_pct":    r["sf_pct"],
                "final_pct": r["final_pct"],
                "win_pct":   r["win_pct"],
            }
            for r in results
        }
    except Exception:
        return {}


def _build_fixture_lambdas(dc: dict, matchday: int | None = None) -> dict[str, list[tuple[float, float]]]:
    """
    For each WC team compute (lambda_for, lambda_against) per match.
    matchday=None  → all 3 group matches per team (original behaviour).
    matchday=1/2/3 → only the fixture(s) in that matchday from the fixtures table.
    Returns canonical_name -> [(lf, la), ...].
    """
    team_params = dc.get("team_params", {})
    if not team_params:
        return {}

    mean_atk = float(np.mean([v["attack"]  for v in team_params.values()]))
    mean_def = float(np.mean([v["defense"] for v in team_params.values()]))

    def _p(name: str) -> dict:
        key = _canonical(name)
        return team_params.get(key, {"attack": mean_atk, "defense": mean_def})

    result: dict[str, list[tuple[float, float]]] = {}

    if matchday is not None:
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute("""
            SELECT ht.name, at.name
            FROM   fixtures f
            JOIN   teams ht ON f.home_team_id = ht.team_id
            JOIN   teams at ON f.away_team_id = at.team_id
            WHERE  f.matchday = ?
        """, (matchday,)).fetchall()
        conn.close()
        for home, away in rows:
            hk, ak = _canonical(home), _canonical(away)
            hp, ap = _p(home), _p(away)
            result.setdefault(hk, []).append((hp["attack"] * ap["defense"], ap["attack"] * hp["defense"]))
            result.setdefault(ak, []).append((ap["attack"] * hp["defense"], hp["attack"] * ap["defense"]))
    else:
        for group_teams in WC2026_GROUPS.values():
            for team in group_teams:
                tk = _canonical(team)
                tp = _p(team)
                result[tk] = [
                    (tp["attack"] * _p(opp)["defense"],
                     _p(opp)["attack"] * tp["defense"])
                    for opp in group_teams if opp != team
                ]
    return result


def _project_mc(
    players: list[dict],
    dc: dict,
    n_sim: int = 50_000,
    qual_probs: dict | None = None,
    matchday: int | None = None,
) -> list[dict]:
    """
    Monte Carlo projection using actual WC2026 group fixtures.

    matchday=None  → project over all 3 group matches (full-tournament view).
    matchday=1/2/3 → project only the fixtures played on that matchday.

    qual_probs: optional {canonical_team: {qf_pct, ...}} for Qualification Booster.
    """
    rng = np.random.default_rng(42)

    team_params = dc.get("team_params", {})
    if team_params:
        mean_atk = float(np.mean([v["attack"]  for v in team_params.values()]))
        mean_def = float(np.mean([v["defense"] for v in team_params.values()]))
    else:
        mean_atk = mean_def = 1.0

    fixture_lambdas = _build_fixture_lambdas(dc, matchday=matchday)
    n_matches_default = 1 if matchday is not None else 3
    fallback = [(mean_atk * mean_def, mean_atk * mean_def)] * n_matches_default

    # Pre-compute per-team average goals-against lambda.
    # Used to apply a defensive context discount for GK/DEF from high-conceding teams.
    team_avg_la: dict[str, float] = {
        tk: float(np.mean([la for _, la in matches]))
        for tk, matches in fixture_lambdas.items()
        if matches
    }

    # Pre-simulate all teams' match goals once — shape (n_sim, n_matches) per team
    team_sims: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    all_teams = [t for grp in WC2026_GROUPS.values() for t in grp]
    for team in all_teams:
        tk = _canonical(team)
        if tk in team_sims:
            continue
        matches = fixture_lambdas.get(tk, fallback)
        if not matches:
            matches = fallback
        gf = np.stack([rng.poisson(lf, n_sim) for lf, _  in matches], axis=1)
        ga = np.stack([rng.poisson(la, n_sim) for _,  la in matches], axis=1)
        team_sims[tk] = (gf, ga)

    for p in players:
        tk  = _canonical(p["team"])
        pos = p["pos"]
        # Price-based star factor: $4.5m -> 0.7, $10.5m -> 1.4
        sf = 0.7 + (max(45, min(105, p["price"])) - 45) / 60.0 * 0.7

        if tk in team_sims:
            gf_arr, ga_arr = team_sims[tk]
        else:
            lam    = mean_atk * mean_def
            gf_arr = np.stack([rng.poisson(lam, n_sim)] * n_matches_default, axis=1)
            ga_arr = np.stack([rng.poisson(lam, n_sim)] * n_matches_default, axis=1)

        n_m = gf_arr.shape[1]  # actual matches for this team in the scope

        # Clean-sheet points
        cs_pts = (ga_arr == 0).astype(np.float32).sum(axis=1) * PT_CS.get(pos, 0.0)

        # Concede penalty: -1 per goal beyond the first (GK/DEF only)
        if pos in ("GK", "DEF"):
            concede_pts = np.maximum(0, ga_arr - 1).astype(np.float32).sum(axis=1) * PT_CONCEDE
        else:
            concede_pts = 0.0

        # Goal contribution
        gf_total = gf_arr.sum(axis=1).astype(np.float32)
        pt_g = gf_total * GOAL_SHARE.get(pos, 0.1) * sf * PT_GOAL.get(pos, 5.0)

        # Assist contribution
        pt_a = gf_total * ASSIST_RATIO * ASSIST_SHARE.get(pos, 0.1) * sf * PT_ASSIST

        # Saves (GK only)
        if pos == "GK":
            save_pts = (ga_arr.sum(axis=1) * 2.0 / 3.0).astype(np.float32) * PT_SAVE3
        else:
            save_pts = 0.0

        # Appearance + stat bonus scaled to actual match count
        appearance = n_m * PT_APPEARANCE
        stat_bonus = PT_STAT_BONUS.get(pos, 0.0) * (n_m / 3)

        match_avg = float((appearance + pt_g + pt_a + cs_pts + concede_pts + save_pts + stat_bonus).mean())

        # Defensive context discount for GK/DEF from high-conceding teams.
        # Teams with avg goals-against > 0.75 across their 3 fixtures get a graded
        # penalty (max 25% for the weakest defensive groups).  This is additional to
        # what the MC already captures — it penalises variance risk and the tendency
        # for the Iraq-game spike to over-value defenders from mixed-fixture teams.
        if pos in ("GK", "DEF") and matchday is None:
            avg_la = team_avg_la.get(tk, 1.0)
            if avg_la > 0.75:
                def_ctx = max(0.75, 1.0 - (avg_la - 0.75) * 0.25)
                match_avg *= def_ctx

        # Qualification Booster (only meaningful for full-tournament view)
        qual_bonus = 0.0
        if qual_probs and matchday is None:
            qp = qual_probs.get(tk, {})
            qual_bonus = 2.0 * qp.get("qf_pct", 0.0) / 100.0

        # Set-piece / penalty taker bonus (scaled to actual match count)
        name_lower = p["name"].lower()
        for sp_key, sp_bonus in PLAYER_SETPIECE_BONUS.items():
            if sp_key in name_lower:
                match_avg += sp_bonus * (n_m / 3)
                break

        projected = match_avg + qual_bonus

        # Injury discount: scale down by available matches
        for inj_key, missed_mds in PLAYER_UNAVAILABLE.items():
            if inj_key in name_lower:
                if matchday is not None and matchday in missed_mds:
                    projected = 0.0   # unavailable this matchday
                elif matchday is None and missed_mds:
                    available = max(0, n_m - len(missed_mds))
                    projected = projected * (available / n_m) if n_m > 0 else 0.0
                break

        # Starter probability discount: players unlikely to start every match
        for prob_key, prob in PLAYER_STARTER_PROB.items():
            if prob_key in name_lower:
                projected *= prob
                match_avg *= prob
                break

        p["projected_pts"] = round(projected, 2)
        p["pts_per_match"] = round(match_avg / max(n_m, 1), 2)
        p["qual_bonus"]    = round(qual_bonus, 2)

    return players


def optimise(budget: int = BUDGET_DEFAULT, predictor=None, booster: str | None = None,
             locked_player_ids: list[int] | None = None) -> dict:
    """
    Select the optimal 15-player squad with explicit starting XI (11) and bench (4).

    Bench slots earn 15% of their projected pts in the objective (BENCH_WEIGHT=0.15),
    so the solver picks quality backup options rather than just the cheapest fillers.

    booster: one of "wildcard", "12th_man", "max_captain", "qualification_booster", or None.
      - wildcard:               no optimizer change (squad already built from scratch).
      - 12th_man:               returns best external player as twelfth_man in result.
      - max_captain:            returns top starters by single-match ceiling; captain auto-assigns.
      - qualification_booster:  returns per-starter qual bonus breakdown and total.

    Returns squad, starters, bench, captain, total_pts (starters + captain bonus).
    Pass predictor= to reuse an already-loaded DCPredictor (avoids re-loading model).
    """
    conn    = sqlite3.connect(DB_PATH)
    players = _load_players(conn)
    conn.close()

    if not players:
        raise RuntimeError("No fantasy players in DB. Run: python wc/scripts/seed_fantasy_players.py")

    dc         = _load_dc()
    qual_probs = _get_qual_probs(predictor)
    players    = _project_mc(players, dc, qual_probs=qual_probs)
    n          = len(players)

    pts    = np.array([p["projected_pts"] for p in players], dtype=float)
    prices = np.array([p["price"]         for p in players], dtype=float)
    pos_l  = [p["pos"]  for p in players]
    teams  = [p["team"] for p in players]

    # Variables: [x_0..x_{n-1}, s_0..s_{n-1}, c_0..c_{n-1}]
    # Objective: minimise −(sum(s_i*pts_i) + sum(c_i*pts_i))
    # Bench players earn BENCH_WEIGHT × pts rather than 0.
    # Decompose: x_i gets w×pts, s_i gets (1−w)×pts, so a starter still totals pts.
    # A bench player earns w×pts, giving the solver an incentive to pick useful backups.
    BENCH_WEIGHT = 0.15
    obj = np.concatenate([-pts * BENCH_WEIGHT, -pts * (1 - BENCH_WEIGHT), -pts])

    rows: list[np.ndarray] = []
    lbs:  list[float]      = []
    ubs:  list[float]      = []

    def _xrow(v): return np.concatenate([v,          np.zeros(n), np.zeros(n)])
    def _srow(v): return np.concatenate([np.zeros(n), v,          np.zeros(n)])
    def _crow(v): return np.concatenate([np.zeros(n), np.zeros(n), v         ])

    # ── Cardinality ───────────────────────────────────────────────────────────
    rows.append(_xrow(np.ones(n))); lbs.append(15.0); ubs.append(15.0)  # squad = 15
    rows.append(_srow(np.ones(n))); lbs.append(11.0); ubs.append(11.0)  # starters = 11
    rows.append(_crow(np.ones(n))); lbs.append(1.0);  ubs.append(1.0)   # captain = 1

    # ── Squad position counts (on x) ─────────────────────────────────────────
    for pos, count in SQUAD_RULES.items():
        v = np.array([1.0 if p == pos else 0.0 for p in pos_l])
        rows.append(_xrow(v)); lbs.append(float(count)); ubs.append(float(count))

    # ── Starting XI formation bounds (on s) ──────────────────────────────────
    # Covers all valid WC Fantasy formations: 4-4-2, 4-3-3, 4-5-1, 3-4-3, 3-5-2, 5-4-1, 5-3-2
    for pos, lo, hi in [("GK", 1, 1), ("DEF", 3, 5), ("MID", 3, 5), ("FWD", 1, 3)]:
        v = np.array([1.0 if p == pos else 0.0 for p in pos_l])
        rows.append(_srow(v)); lbs.append(float(lo)); ubs.append(float(hi))

    # ── Budget (on x) ────────────────────────────────────────────────────────
    rows.append(_xrow(prices)); lbs.append(0.0); ubs.append(float(budget))

    # ── Per-team cap = 3 (on x) ───────────────────────────────────────────────
    for team in set(teams):
        v = np.array([1.0 if t == team else 0.0 for t in teams])
        rows.append(_xrow(v)); lbs.append(0.0); ubs.append(3.0)

    # ── Hierarchy: c_i <= s_i <= x_i (block-matrix form) ────────────────────
    I_n = np.eye(n)
    Z_n = np.zeros((n, n))
    # s_i - x_i <= 0: row per player, [-I | I | 0]
    sx_block = np.hstack([-I_n, I_n, Z_n])
    # c_i - s_i <= 0: row per player, [0 | -I | I]
    cs_block = np.hstack([Z_n, -I_n, I_n])

    for row in sx_block:
        rows.append(row); lbs.append(-np.inf); ubs.append(0.0)
    for row in cs_block:
        rows.append(row); lbs.append(-np.inf); ubs.append(0.0)

    A           = np.vstack(rows)
    constraints = LinearConstraint(A, lb=np.array(lbs), ub=np.array(ubs))

    # Lock specified players into squad by forcing x_i lower bound = 1
    lb_arr = np.zeros(3 * n)
    ub_arr = np.ones(3 * n)
    locked_ids = set(locked_player_ids or [])
    for i, player in enumerate(players):
        if player["id"] in locked_ids:
            lb_arr[i] = 1.0

    result = milp(obj, constraints=constraints,
                  integrality=np.ones(3 * n), bounds=Bounds(lb_arr, ub_arr))

    if result.status != 0:
        raise RuntimeError(f"Optimizer failed: {result.message}")

    x_sol = result.x[:n]
    s_sol = result.x[n:2 * n]
    c_sol = result.x[2 * n:]

    pos_order = list(SQUAD_RULES.keys())
    squad = []
    for i, player in enumerate(players):
        if x_sol[i] > 0.5:
            player["is_starter"] = bool(s_sol[i] > 0.5)
            player["is_captain"] = bool(c_sol[i] > 0.5)
            squad.append(player)

    # Sort: starters before bench, then by position order, then by projected pts
    squad.sort(key=lambda p: (
        pos_order.index(p["pos"]),
        not p["is_starter"],
        -p["projected_pts"],
    ))

    captain  = next(p for p in squad if p["is_captain"])
    starters = [p for p in squad if p["is_starter"]]
    bench    = [p for p in squad if not p["is_starter"]]

    out: dict = {
        "squad":      squad,
        "starters":   starters,
        "bench":      bench,
        "total_pts":  round(sum(p["projected_pts"] for p in starters) + captain["projected_pts"], 1),
        "total_cost": int(sum(p["price"] for p in squad)),
        "captain":    captain,
        "booster":    booster,
    }

    if booster == "12th_man":
        squad_ids = {p["id"] for p in squad}
        external  = sorted(
            [p for p in players if p["id"] not in squad_ids],
            key=lambda p: p["projected_pts"], reverse=True,
        )
        out["twelfth_man"] = external[0] if external else None

    elif booster == "max_captain":
        # Auto-assigns to highest scorer in XI — rank starters by single-match ceiling
        by_ceiling = sorted(starters, key=lambda p: p.get("pts_per_match", 0.0), reverse=True)
        out["max_cap_candidates"] = by_ceiling[:3]
        top_ppg = by_ceiling[0].get("pts_per_match", 0.0) if by_ceiling else 0.0
        # Expected max bonus ≈ top single-match pts × 2, discounted ~8% vs manual pick
        out["expected_max_cap_pts"] = round(top_ppg * 2 * 0.92, 1)

    elif booster == "qualification_booster":
        out["qual_booster_breakdown"] = [
            {"name": p["name"], "team": p["team"], "pos": p["pos"],
             "qual_bonus": p.get("qual_bonus", 0.0)}
            for p in starters
        ]
        out["qual_booster_total"] = round(sum(p.get("qual_bonus", 0.0) for p in starters), 2)

    return out


def captain_picks(top_n: int = 10, predictor=None, matchday: int | None = None) -> list[dict]:
    conn    = sqlite3.connect(DB_PATH)
    players = _load_players(conn)
    conn.close()
    if not players:
        return []
    dc         = _load_dc()
    qual_probs = _get_qual_probs(predictor) if matchday is None else None
    players    = _project_mc(players, dc, qual_probs=qual_probs, matchday=matchday)
    outfield   = [p for p in players if p["pos"] != "GK" and p["projected_pts"] > 0]
    outfield.sort(key=lambda p: p["projected_pts"], reverse=True)
    return outfield[:top_n]


def live_captain_advice(matchday: int, predictor=None) -> dict:
    """
    For a live matchday: determine which fixtures have already kicked off and
    return the best captain options from teams that haven't played yet.

    is_live=True when some fixtures have started and some haven't — that's the
    window where changing your captain is still meaningful.
    """
    from datetime import datetime, timezone

    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT f.home_team_id, f.away_team_id, f.kickoff_time
        FROM   fixtures f
        WHERE  f.matchday = ?
    """, (matchday,)).fetchall()
    conn.close()

    if not rows:
        return {"is_live": False, "matchday": matchday,
                "played_team_count": 0, "remaining_team_count": 0, "remaining_picks": []}

    now = datetime.now(timezone.utc)
    played_ids: set[int]    = set()
    remaining_ids: set[int] = set()

    for home_id, away_id, kickoff_str in rows:
        if not kickoff_str:
            remaining_ids.update([home_id, away_id])
            continue
        try:
            ko = datetime.fromisoformat(kickoff_str.replace("Z", "+00:00"))
            if ko.tzinfo is None:
                ko = ko.replace(tzinfo=timezone.utc)
            if ko <= now:
                played_ids.update([home_id, away_id])
            else:
                remaining_ids.update([home_id, away_id])
        except Exception:
            remaining_ids.update([home_id, away_id])

    is_live = bool(played_ids) and bool(remaining_ids)

    remaining_picks: list[dict] = []
    if remaining_ids:
        conn = sqlite3.connect(DB_PATH)
        team_rows = conn.execute(
            "SELECT team_id, name FROM teams WHERE team_id IN (%s)"
            % ",".join("?" * len(remaining_ids)),
            list(remaining_ids),
        ).fetchall()
        conn.close()
        remaining_canonical = {_canonical(name) for _, name in team_rows}
        all_picks = captain_picks(top_n=30, predictor=predictor, matchday=matchday)
        remaining_picks = [p for p in all_picks if _canonical(p["team"]) in remaining_canonical][:10]

    return {
        "is_live":              is_live,
        "matchday":             matchday,
        "played_team_count":    len(played_ids),
        "remaining_team_count": len(remaining_ids),
        "remaining_picks":      remaining_picks,
    }


def get_projected_players(predictor=None) -> list[dict]:
    """All fantasy players with projected points — used to populate the squad builder."""
    conn    = sqlite3.connect(DB_PATH)
    players = _load_players(conn)
    conn.close()
    if not players:
        return []
    dc         = _load_dc()
    qual_probs = _get_qual_probs(predictor)
    players    = _project_mc(players, dc, qual_probs=qual_probs)
    players.sort(key=lambda p: p.get("projected_pts", 0.0), reverse=True)
    return players
