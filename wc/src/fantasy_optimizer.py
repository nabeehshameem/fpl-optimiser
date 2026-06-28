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
BUDGET_DEFAULT = 1050  # $105.0m (increased for knockout phase)

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
PLAYER_UNAVAILABLE: dict[str, set[int]] = {
}

# Players who miss specific knockout rounds due to injury.
# Values are sets of qual_probs keys to EXCLUDE from the e_ko_matches sum.
# Example: "raphinha": {"r32_pct"} → Raphinha skips R32, projected from R16 onward.
PLAYER_KO_SKIP_ROUNDS: dict[str, set[str]] = {
    "raphinha": {"r32_pct"},  # ankle injury; expected return R16/QF
}

# Named starter probability tiers — use these instead of raw floats.
# IMPACT_SUB and below = genuinely not expected to start; ROTATION = genuine competition.
class P:
    NAILED     = 1.00  # near-certain starter when fit
    EXPECTED   = 0.88  # strong first-choice, minor doubt
    LIKELY     = 0.78  # probable starter
    ROTATION   = 0.65  # genuine competition for starting spot
    IMPACT_SUB = 0.55  # expected to come off the bench, gets minutes
    BENCH      = 0.35  # squad depth, unlikely to start
    FRINGE     = 0.15  # very unlikely to see meaningful minutes
    OUT        = 0.05  # injury / not in squad

# Unconfirmed intel is capped at LIKELY so a rumour never tanks a player's projection.
_UNCONFIRMED_FLOOR = P.LIKELY

# Probability that a player starts (gets 60+ min) in a given match.
# Default = 1.0 (no discount). Keys are lowercase substrings of the player name.
# Use P.XXX tier constants — never raw floats.
PLAYER_STARTER_PROB: dict[str, float] = {
    # ── Argentina ────────────────────────────────────────────────────────
    "messi":        P.EXPECTED,    # nailed knockout starter — group stage rotation was dead rubber vs Jordan
    "senesi":       P.BENCH,       # not expected to start for Argentina in knockouts
    "lautaro":      P.EXPECTED,    # Argentina #9 in knockouts — group rotation vs Jordan was misleading
    "enzo fern":    P.LIKELY,      # key Argentina CM in knockouts — MD3 rotation no longer relevant
    "soul":         P.IMPACT_SUB,  # Soulé — young, not yet a regular starter
    "lo celso":     P.BENCH,       # squad rotational player
    "barco":        P.BENCH,       # fringe squad
    "buend":        P.IMPACT_SUB,  # competes with De Paul / Mac Allister
    "romero":       P.BENCH,       # partially torn MCL (mid-April), World Cup in jeopardy
    "otamendi":     P.BENCH,       # 3rd-choice CB behind Romero and L. Martínez
    "lisandro":     P.EXPECTED,    # L. Martínez — Argentina first-choice CB in knockouts
    "nico paz":     P.BENCH,       # won't get minutes over Mac Allister / Enzo / De Paul

    # ── Germany ──────────────────────────────────────────────────────────
    "sané":         P.ROTATION,    # competes for wide role with Wirtz/Leweling
    "musiala":      P.EXPECTED,    # key Germany playmaker, starts every knockout game
    "havertz":      P.EXPECTED,    # back to #9 role in knockouts after MD3 rotation
    "wirtz":        P.EXPECTED,    # Germany's most important attacker in knockouts
    "goretzka":     P.IMPACT_SUB,  # deep rotation behind Musiala/Wirtz/Kimmich
    "leweling":     P.ROTATION,    # competes for wide role
    "lennart karl": P.BENCH,       # Freiburg MID — Germany squad but not guaranteed starter
    "raum":         P.LIKELY,      # Nathan Brown in contention for LB slot
    "undav":        P.ROTATION,    # was MD3 starter but Havertz likely returns for knockouts

    # ── England ──────────────────────────────────────────────────────────
    "saka":         P.EXPECTED,    # hamstring recovered; likely starter in knockouts
    "reece james":  P.LIKELY,      # recurring knee/hamstring issues — fitness managed
    "kane":         P.EXPECTED,    # England #9, starts every knockout game
    "bellingham":   P.NAILED,      # England's best player, nailed in knockouts
    "o'reilly":     P.BENCH,       # played vs Panama dead rubber, won't start knockouts
    "pickford":     P.NAILED,      # England #1 GK
    "mainoo":       P.ROTATION,    # competes with Rice/Bellingham
    "rogers":       P.BENCH,       # fringe squad player
    "gordon":       P.LIKELY,      # competing for wide role
    "eze":          P.ROTATION,    # competes with Palmer/Foden
    "palmer":       P.LIKELY,      # competes with Eze/Foden

    # ── Spain ────────────────────────────────────────────────────────────
    "yamal":        P.EXPECTED,    # 44.7% ownership; community expects him fit
    "williams":     P.LIKELY,      # Nico Williams, competing with Yamal/Olmo
    "olmo":         P.LIKELY,      # competes with Pedri when both fit
    "gavi":         P.LIKELY,      # returning from injury, competes with Zubimendi
    "merino":       P.BENCH,       # stress fracture in foot (Feb), targeting return — fitness risk
    "zubimendi":    P.LIKELY,      # rotates with Rodri
    "ferran":       P.IMPACT_SUB,  # super sub behind Williams/Yamal/Oyarzabal, rarely starts
    "cubar":        P.LIKELY,      # Cubarsi — competes with Laporte/García for CB slot
    "laporte":      P.LIKELY,      # rotation risk — Cubarsi pushing hard for his CB spot
    "llorente":     P.LIKELY,      # expected to start over Porro at Spain RB/wing
    "porro":        P.BENCH,       # Llorente preferred ahead of him in knockouts

    # ── France ───────────────────────────────────────────────────────────
    "dembel":       P.EXPECTED,    # Ousmane Dembélé — started MD1 & MD2, subbed ~85th min
    "cherki":       P.ROTATION,    # young, competes for wide/10 role
    "doué":         P.LIKELY,      # Désiré Doué — rotation

    # ── Brazil ───────────────────────────────────────────────────────────
    "neymar":        P.BENCH,       # did not play MD1 or MD2 — fitness very uncertain for MD3
    "casemiro":      P.ROTATION,    # avg 67 min in MD1+MD2 — being rotated off, not guaranteed full game
    "wesley":        P.LIKELY,      # Brazil RB rotation — competes with Vanderson/Danilo
    "cunha":         P.ROTATION,    # 3G in group stage in limited mins — pushing for knockout starts
    "igor thiago":   P.BENCH,       # minimal game time in MD2+MD3; not a genuine knockout starter

    # ── Norway ───────────────────────────────────────────────────────────
    "sorloth":      P.ROTATION,    # backup striker to Haaland
    "stig":         P.IMPACT_SUB,  # L. Østigård — Norway 3rd in group; tough France/Senegal fixtures
    "nusa":         P.ROTATION,    # young, rotation with Ødegaard/Aursnes

    # ── Portugal ─────────────────────────────────────────────────────────
    "ronaldo":      P.LIKELY,      # 41 years old in 2026; still likely starts but rotation risk

    # ── Belgium ──────────────────────────────────────────────────────────
    "lammens":      P.BENCH,       # Belgium #2 GK — Courtois is clear starter when fit
    "penders":      P.FRINGE,      # Belgium #3 GK
    "tielemans":    P.LIKELY,      # rotates in Belgium's evolving midfield
    "witsel":       P.IMPACT_SUB,  # 36 years old in 2026, squad veteran, not guaranteed XI
    "lukaku":       P.IMPACT_SUB,  # De Ketelaere expected to start as #9; Lukaku impact sub
    "de cuyper":    P.LIKELY,      # some competition for Belgium LB slot

    # ── Mexico ───────────────────────────────────────────────────────────
    "brian guti":   P.ROTATION,    # Brian Gutiérrez — heavy competition for starting spot

    # ── Australia ────────────────────────────────────────────────────────
    "beach":        P.BENCH,       # Patrick Beach — backup GK behind Ryan/Vukovic

    # ── New Zealand ───────────────────────────────────────────────────────
    "o. sail":      P.LIKELY,      # NZ #1 GK — save bonus vs Belgium overstated; model discount applied

    # ── Netherlands ──────────────────────────────────────────────────────
    "timber":       P.BENCH,       # groin injury, no game since March 14 — fitness uncertain
    "koopmeiners":  P.LIKELY,      # competes with Gravenberch/Reijnders
    "madueke":      P.ROTATION,    # wide rotation
    "schouten":     P.ROTATION,    # DM rotation
    "gakpo":        P.LIKELY,      # starts but faces Morocco R32 — strong defence, discounted ceiling
    "mem":          P.ROTATION,    # Memphis Depay — squad rotation, impact role in knockouts
    "malen":        P.LIKELY,      # key winger, knockouts starter after MD3 rotation

    # ── Ghana ────────────────────────────────────────────────────────────
    "kudus":        P.BENCH,       # quad injury Jan + hamstring setback Apr, WC participation in doubt

    # ── Morocco ──────────────────────────────────────────────────────────
    "hakimi":       P.EXPECTED,    # hamstring (PSG CL semi), expected to recover in time

    # ── Turkey ───────────────────────────────────────────────────────────
    "güler":        P.LIKELY,      # pulled hamstring (April), on track to recover for tournament

    # ── Croatia ──────────────────────────────────────────────────────────
    "modri":        P.EXPECTED,    # Modrić — Croatia's captain, starts every knockout game

    # ── Canada ───────────────────────────────────────────────────────────
    "davies":       P.IMPACT_SUB,  # ACL (March) + hamstring (May), day-by-day rehab

    # ── Colombia ─────────────────────────────────────────────────────────
    "mojica":       P.LIKELY,      # LB rotation risk — competes with Machado/Arias for left slot

    # ── Mexico ───────────────────────────────────────────────────────────
    "mateo ch":     P.FRINGE,      # Mateo Chávez — widely regarded as second-choice LB

    # ── Fantasy data artefacts / confirmed non-squad / poor value ────────
    "tagnaouti":    P.FRINGE,      # Morocco backup GK — Bounou is their clear #1
    "abunada":      P.FRINGE,      # Qatar GK — Qatar are among the weakest teams; minimal CS/save ceiling
    "mastil":       P.BENCH,       # Algeria GK — inflated by Jordan MD1 but faces Argentina MD2; poor value
    "dacosta":      P.OUT,         # not in Ecuador's actual WC squad
    "boulbina":     P.OUT,         # Algeria MID — not in WC squad
    "halhal":       P.OUT,         # Morocco DEF fringe — not a reliable starter
    "yaimar":       P.FRINGE,      # Ecuador DEF fringe — 0.5% ownership, unlikely starter
    "jayden":       P.FRINGE,      # Jayden Adams (South Africa) — low-qual team, unknown starter
    "sucic":        P.FRINGE,      # Petar Sučić (Croatia MID) — low ownership, fringe pick
    "vuskovic":     P.FRINGE,      # Luka Vusković (Croatia DEF) — low ownership, fringe pick

    # ── Iran ─────────────────────────────────────────────────────────────
    "beiranvand":   P.EXPECTED,    # Iran starter GK — noted for completeness

    # ── Egypt ────────────────────────────────────────────────────────────
    "shobeir":      P.EXPECTED,    # Mostafa Shobeir — Egypt #1 GK, started MD1 vs Belgium (90 min)

    # ── Uruguay ──────────────────────────────────────────────────────────
    "muslera":      P.EXPECTED,    # confirmed starter MD1 vs Saudi Arabia (90 min)
    "rochet":       P.BENCH,       # on bench MD1, did not play — Muslera is the #1

    # ── Ecuador ──────────────────────────────────────────────────────────
    "estupi":       P.BENCH,       # Estupiñán — 0 min MD1 vs Ivory Coast, not getting minutes

    # ── Switzerland ──────────────────────────────────────────────────────
    "widmer":       P.BENCH,       # Silvan Widmer — named sub, 0 min MD1 vs Canada

    # ── Argentina ────────────────────────────────────────────────────────
    "molina":       P.IMPACT_SUB,  # Nahuel Molina — came on as sub at HT in MD1 (45 min only)
    "di lollo":     P.OUT,          # Lautaro Di Lollo — won't feature in knockouts; blocks Argentina cap
}

# Intel not yet verified — applied as max(value, _UNCONFIRMED_FLOOR) so a rumour
# never tanks a player's projection harder than LIKELY (0.78).
PLAYER_STARTER_PROB_UNCONFIRMED: dict[str, float] = {
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
    "raphinha":   6.0,   # BRA — penalty + corner/FK taker, Brazil captain

    # ── Corner / FK takers ───────────────────────────────────────────────
    "kimmich":    5.0,   # GER — sole corner + FK taker; setpiece assists are primary ceiling-raiser as DEF
    "de bruyne":  1.5,   # BEL — primary FK delivery
    "degaard":    1.5,   # NOR — M. Ødegaard (Ø prefix skipped; "degaard" matches)
    "bellingham": 1.0,   # ENG
    "güler":      1.5,   # TUR — FK specialist
    "de paul":    1.0,   # ARG
    "pedri":      1.0,   # ESP
    "hakimi":     0.8,   # MAR — overlapping FK role
}

# Attacking-contribution multiplier for defensive/holding midfielders.
# Applied to the star-factor (sf) that drives goal/assist projections.
# CS and appearance points are unaffected — these players still earn those.
# Keys are lowercase substrings of the player name.
PLAYER_CDM_DISCOUNT: dict[str, float] = {
    "rodri":        0.25,  # Spain CDM — Ballon d'Or for defending, rarely scores
    "zubimendi":    0.30,  # Spain back-up DM
    "rice":         0.40,  # England DM — some attacking output but mainly defensive
    "casemiro":     0.35,  # aging Brazil CDM
    "tchouameni":   0.35,  # France DM
    "camavinga":    0.45,  # France CM, more box-to-box but still low goal return
    "amrabat":      0.30,  # Morocco DM
    "gravenberch":  0.45,  # Netherlands CM — runner, rarely scores
    "joão neves": 0.35,  # Portugal CDM — holding role, very low personal goal/assist ceiling
    "goretzka":     0.55,  # Germany box-to-box, some goals but limited
    "mac allister": 0.30,  # Argentina CDM — sits deep, low personal goal/assist ceiling
}

# Host nations receive a 10% boost to projected match points.
# The DC model is calibrated on neutral-venue history and doesn't account for
# home crowd lift, venue familiarity, or the extra motivation of hosting.
HOST_NATIONS: frozenset[str] = frozenset({"United States", "Mexico", "Canada"})
HOST_ADVANTAGE = 1.05

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

# Group stage complete — MD3 rotation and competitive bonus dicts cleared.
_MD3_ROTATION_TEAMS: dict[str, float] = {}
_MD3_COMPETITIVE_BONUS: dict[str, float] = {}


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
               t.name AS team_name, fp.ownership
        FROM   fantasy_players fp
        LEFT JOIN teams t ON fp.team_id = t.team_id
        WHERE  fp.position IS NOT NULL AND fp.price IS NOT NULL AND fp.price > 0
    """).fetchall()
    return [
        {"id": r[0], "name": r[1], "pos": r[2], "price": r[3],
         "team": r[4] or "Unknown", "ownership": float(r[5] or 0.0)}
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
                "r32_pct":   r["r32_pct"],
                "r16_pct":   r["r16_pct"],
                "qf_pct":    r["qf_pct"],
                "sf_pct":    r["sf_pct"],
                "final_pct": r["final_pct"],
                "win_pct":   r["win_pct"],
            }
            for r in results
        }
    except Exception:
        return {}


def _build_fixture_lambdas(
    dc: dict,
    matchday: int | None = None,
    matchdays: list[int] | None = None,
) -> dict[str, list[tuple[float, float]]]:
    """
    For each WC team compute (lambda_for, lambda_against) per match.
    matchdays=[1,2] → fixtures from those specific matchdays (DB lookup).
    matchday=1/2/3  → single matchday (DB lookup, kept for captain_picks).
    Neither         → all 3 group matches from WC2026_GROUPS.
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

    mds = matchdays if matchdays is not None else ([matchday] if matchday is not None else None)
    if mds is not None:
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute(
            f"""
            SELECT ht.name, at.name
            FROM   fixtures f
            JOIN   teams ht ON f.home_team_id = ht.team_id
            JOIN   teams at ON f.away_team_id = at.team_id
            WHERE  f.matchday IN ({','.join('?' * len(mds))})
            ORDER  BY f.kickoff_time
            """,
            mds,
        ).fetchall()
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
    matchdays: list[int] | None = None,
) -> list[dict]:
    """
    Monte Carlo projection using actual WC2026 group fixtures.

    matchdays=[1,2] → project only over those matchdays (e.g. MD1+MD2 squad build).
    matchday=1/2/3  → single matchday (captain picks / live advice).
    Neither         → all 3 group matches (full-tournament view).

    qual_probs: optional {canonical_team: {qf_pct, ...}} for Qualification Booster.
    Qual bonus is always included when matchday is None (covers both MD1+MD2 and full views).
    """
    rng = np.random.default_rng(42)

    team_params = dc.get("team_params", {})
    if team_params:
        mean_atk = float(np.mean([v["attack"]  for v in team_params.values()]))
        mean_def = float(np.mean([v["defense"] for v in team_params.values()]))
    else:
        mean_atk = mean_def = 1.0

    fixture_lambdas = _build_fixture_lambdas(dc, matchday=matchday, matchdays=matchdays)
    n_matches_default = (
        len(matchdays) if matchdays is not None
        else (1 if matchday is not None else 3)
    )
    fallback = [(mean_atk * mean_def, mean_atk * mean_def)] * n_matches_default

    # Pre-compute per-team average goals-against lambda.
    # Used to apply a defensive context discount for GK/DEF from high-conceding teams.
    team_avg_la: dict[str, float] = {
        tk: float(np.mean([la for _, la in matches]))
        for tk, matches in fixture_lambdas.items()
        if matches
    }

    # Pre-compute max ownership within each (team, position) group.
    # Used to detect genuine backup players vs starters.
    team_pos_max_own: dict[tuple, float] = {}
    team_gk_sum_own: dict[str, float] = {}
    for p in players:
        key = (p["team"], p["pos"])
        own_val = p.get("ownership", 0.0)
        team_pos_max_own[key] = max(team_pos_max_own.get(key, 0.0), own_val)
        if p["pos"] == "GK":
            team_gk_sum_own[p["team"]] = team_gk_sum_own.get(p["team"], 0.0) + own_val

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
        la_arr = np.array([la for _, la in matches], dtype=np.float32)
        team_sims[tk] = (gf, ga, la_arr)

    for p in players:
        tk  = _canonical(p["team"])
        pos = p["pos"]
        name_lower = p["name"].lower()

        # Price-based star factor: $4.5m -> 0.7, $10.5m -> 1.4
        sf = 0.7 + (max(45, min(105, p["price"])) - 45) / 60.0 * 0.7

        # Premium forward discount: group-stage audit showed FWDs priced £8.5m+
        # systematically underperformed (Mbappé -32.7, Haaland -28.9, Kane -20.1,
        # Lautaro -22.1). High-profile FWDs face tighter marking and rotation pressure.
        if pos == "FWD" and p["price"] >= 85:
            sf *= 0.60

        # CDM/holding-MID discount: reduces goal/assist projection for defensive mids
        for cdm_key, cdm_disc in PLAYER_CDM_DISCOUNT.items():
            if cdm_key in name_lower:
                sf *= cdm_disc
                break

        if tk in team_sims:
            gf_arr, ga_arr, la_vals = team_sims[tk]
        else:
            lam    = mean_atk * mean_def
            gf_arr = np.stack([rng.poisson(lam, n_sim)] * n_matches_default, axis=1)
            ga_arr = np.stack([rng.poisson(lam, n_sim)] * n_matches_default, axis=1)
            la_vals = np.full(n_matches_default, lam, dtype=np.float32)

        n_m = gf_arr.shape[1]  # actual matches for this team in the scope

        # Clean-sheet points
        cs_pts = (ga_arr == 0).astype(np.float32).sum(axis=1) * PT_CS.get(pos, 0.0)

        # Concede penalty: -1 per goal beyond the first (GK/DEF only)
        if pos in ("GK", "DEF"):
            concede_arr = np.maximum(0, ga_arr - 1).astype(np.float32)
            concede_pts = concede_arr.sum(axis=1) * PT_CONCEDE
            # Expected concede penalty saved per projected period (positive = pts saved)
            _raw_concede_saved = float(concede_arr.mean())  # goals beyond 1st, per match avg
        else:
            concede_pts = 0.0
            _raw_concede_saved = 0.0

        # Goal contribution
        gf_total = gf_arr.sum(axis=1).astype(np.float32)
        pt_g = gf_total * GOAL_SHARE.get(pos, 0.1) * sf * PT_GOAL.get(pos, 5.0)

        # Assist contribution
        pt_a = gf_total * ASSIST_RATIO * ASSIST_SHARE.get(pos, 0.1) * sf * PT_ASSIST

        # Saves (GK only): estimate shots on target = la * 2 per match.
        # Saves = shots - goals, clamped to 0. This gives non-zero save pts
        # during clean sheets (GKs still make saves even in 0-0 games).
        if pos == "GK":
            exp_saves = np.maximum(
                0.0, la_vals[np.newaxis, :] * 2.0 - ga_arr.astype(np.float32)
            )
            save_pts = (exp_saves / 3.0 * PT_SAVE3).sum(axis=1)
        else:
            save_pts = 0.0

        # Appearance + stat bonus scaled to actual match count
        appearance = n_m * PT_APPEARANCE
        stat_bonus = PT_STAT_BONUS.get(pos, 0.0) * (n_m / 3)

        match_avg = float((appearance + pt_g + pt_a + cs_pts + concede_pts + save_pts + stat_bonus).mean())

        # Defensive context discount for GK/DEF from high-conceding teams.
        # GKs get a steeper penalty than DEFs: a GK from a team expected to concede
        # 2+ goals/game is a bad fantasy pick (no CS, concede penalty stacks) whereas
        # a DEF on the same team can still contribute goal/assist points.
        if pos in ("GK", "DEF") and matchday is None:
            avg_la = team_avg_la.get(tk, 1.0)
            if avg_la > 0.75:
                if pos == "GK":
                    def_ctx = max(0.45, 1.0 - (avg_la - 0.75) * 0.55)
                else:
                    def_ctx = max(0.75, 1.0 - (avg_la - 0.75) * 0.25)
                match_avg *= def_ctx

        # Host nation advantage: USA/Mexico/Canada get 10% uplift on expected pts.
        if p["team"] in HOST_NATIONS:
            match_avg *= HOST_ADVANTAGE

        # Set-piece / penalty taker bonus (scaled to actual match count)
        for sp_key, sp_bonus in PLAYER_SETPIECE_BONUS.items():
            if sp_key in name_lower:
                match_avg += sp_bonus * (n_m / 3)
                break

        # Scouting bonus: +2pts per match where player scores >4pts AND <5% ownership
        if p.get("ownership", 0.0) < 5.0:
            if pos in ("GK", "DEF"):
                p_qualify = (ga_arr == 0).astype(np.float32).mean(axis=0)
            else:
                exp_gf = gf_arr.astype(np.float32).mean(axis=0)
                squad_sz = max(1, SQUAD_RULES.get(pos, 3))
                indiv_share = (
                    GOAL_SHARE.get(pos, 0.1) + ASSIST_RATIO * ASSIST_SHARE.get(pos, 0.1)
                ) / squad_sz
                lam_ga = exp_gf * indiv_share * sf
                p_qualify = 1 - np.exp(-lam_ga)
            match_avg += float(p_qualify.sum()) * 2.0

        # Knockout-stage projection: scale per-match rate by expected rounds played.
        # Activated when qual_probs contains r16_pct (i.e. group stage is complete).
        # KO_DIFFICULTY = 0.82: knockout opponents are top-32 quality, harder than
        # the mixed-strength group opponents simulated above.
        if qual_probs and matchday is None:
            qp = qual_probs.get(tk, {})
            if "r16_pct" in qp:
                skip_rounds: set[str] = set()
                for ko_key, skipped in PLAYER_KO_SKIP_ROUNDS.items():
                    if ko_key in name_lower:
                        skip_rounds = skipped
                        break
                _KO_ROUND_KEYS = ("r32_pct", "r16_pct", "qf_pct", "sf_pct", "final_pct")
                e_ko_matches = sum(
                    qp.get(r, 0.0) for r in _KO_ROUND_KEYS if r not in skip_rounds
                ) / 100.0
                projected = (match_avg / max(n_m, 1)) * 0.82 * e_ko_matches
            else:
                projected = match_avg + 2.0 * qp.get("qf_pct", 0.0) / 100.0
        else:
            projected = match_avg

        # Injury discount: scale down by available matches
        for inj_key, missed_mds in PLAYER_UNAVAILABLE.items():
            if inj_key in name_lower:
                if matchday is not None and matchday in missed_mds:
                    projected = 0.0   # unavailable this matchday
                elif matchday is None and missed_mds:
                    available = max(0, n_m - len(missed_mds))
                    projected = projected * (available / n_m) if n_m > 0 else 0.0
                break

        # Starter probability discount: confirmed overrides first, then unconfirmed (softcapped), then ownership-derived.
        explicit_prob = next(
            (prob for key, prob in PLAYER_STARTER_PROB.items() if key in name_lower),
            None,
        )
        if explicit_prob is None:
            unconf = next(
                (prob for key, prob in PLAYER_STARTER_PROB_UNCONFIRMED.items() if key in name_lower),
                None,
            )
            if unconf is not None:
                explicit_prob = max(unconf, _UNCONFIRMED_FLOOR)
        if explicit_prob is not None:
            projected *= explicit_prob
            match_avg *= explicit_prob
        else:
            own = p.get("ownership", 0.0)
            group_max = team_pos_max_own.get((p["team"], pos), 1.0)

            if pos == "GK":
                # GKs always use normalised ownership within the team.
                # (own / sum)^0.5 rewards a clear #1 (Neuer ≈ 0.94) while giving
                # genuinely competing GKs (Alisson vs Ederson) a meaningful haircut.
                # Backup GKs with near-zero ownership naturally project very low.
                gk_sum = team_gk_sum_own.get(p["team"], 0.0)
                if gk_sum > 0:
                    norm_prob = max(0.10, (own / gk_sum) ** 0.5)
                    projected *= norm_prob
                    match_avg *= norm_prob
            elif own < 1.0:
                # Outfield fringe: scale 0.40 → 0.85 over 0–1% ownership
                auto_prob = 0.40 + own * 0.45
                projected *= auto_prob
                match_avg *= auto_prob
            elif group_max > 2.0 and own < group_max * 0.30 and own < 3.0:
                # Relative outfield backup: clearly behind the group leader and <3% absolute
                rel = own / (group_max * 0.30)
                auto_prob = 0.50 + rel * 0.30
                projected *= auto_prob
                match_avg *= auto_prob

        # MD3-specific team-level adjustments (only when projecting MD3 in isolation)
        if matchdays == [3] or matchday == 3:
            if explicit_prob is None:
                # Rotation penalty for confirmed-1st teams — applies only to players
                # without an individual override (avoids double-penalising named stars)
                projected *= _MD3_ROTATION_TEAMS.get(tk, 1.0)
            # Competitive motivation bonus — all players benefit equally
            projected *= _MD3_COMPETITIVE_BONUS.get(tk, 1.0)

        p["projected_pts"] = round(projected, 2)
        p["pts_per_match"] = round(match_avg / max(n_m, 1), 2)

        # Qual bonus: expected +2 pts from Qualification Booster chip for next R32 match.
        # r16_pct = P(team wins their R32 match and advances to R16).
        if qual_probs and matchday is None:
            qp = qual_probs.get(tk, {})
            r16_p = qp.get("r16_pct", 0.0) / 100.0
            p["qual_bonus"] = round(2.0 * r16_p, 2)
        else:
            p["qual_bonus"] = 0.0

        # Concede saved: expected pts recovered if CS Shield negates concede penalty.
        # Scaled by knockout difficulty and starter probability (same as projected_pts).
        if pos in ("GK", "DEF") and _raw_concede_saved > 0:
            if qual_probs and matchday is None:
                qp = qual_probs.get(tk, {})
                if "r16_pct" in qp:
                    e_ko = (qp.get("r32_pct", 0.0) + qp.get("r16_pct", 0.0) +
                            qp.get("qf_pct", 0.0) + qp.get("sf_pct", 0.0) +
                            qp.get("final_pct", 0.0)) / 100.0
                    saved = _raw_concede_saved * 0.82 * e_ko
                else:
                    saved = _raw_concede_saved * n_m
            else:
                saved = _raw_concede_saved * n_m
            # Apply same starter-prob discount as projected_pts
            if explicit_prob is not None:
                saved *= explicit_prob
            p["concede_saved"] = round(saved, 2)
        else:
            p["concede_saved"] = 0.0

    return players


def optimise(budget: int = BUDGET_DEFAULT, predictor=None, booster: str | None = None,
             locked_player_ids: list[int] | None = None,
             locked_starter_ids: list[int] | None = None,
             matchdays: list[int] | None = None) -> dict:
    """
    Select the optimal 15-player squad with explicit starting XI (11) and bench (4).

    Bench slots earn BENCH_WEIGHT × their projected pts in the objective,
    so the solver picks quality backup options suitable for live substitutions.

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
    players    = _project_mc(players, dc, qual_probs=qual_probs, matchdays=matchdays)
    n          = len(players)

    pts    = np.array([p["projected_pts"] for p in players], dtype=float)
    prices = np.array([p["price"]         for p in players], dtype=float)
    pos_l  = [p["pos"]  for p in players]
    teams  = [p["team"] for p in players]

    # Variables: [x_0..x_{n-1}, s_0..s_{n-1}, c_0..c_{n-1}]
    # Objective: minimise −(sum(s_i*pts_i) + sum(c_i*pts_i))
    # Decompose: x_i gets BENCH_WEIGHT×pts, s_i gets (1−BENCH_WEIGHT)×pts, starter still totals pts.
    # Bench player earns BENCH_WEIGHT×pts, giving the solver an incentive to pick quality live subs.
    BENCH_WEIGHT = 0.60
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

    # Lock specified players into squad (x_i=1) or as starters (x_i=1, s_i=1)
    lb_arr = np.zeros(3 * n)
    ub_arr = np.ones(3 * n)
    locked_ids = set(locked_player_ids or [])
    locked_starter_set = set(locked_starter_ids or [])
    for i, player in enumerate(players):
        pid = player["id"]
        if pid in locked_ids or pid in locked_starter_set:
            lb_arr[i] = 1.0        # must be in squad
        if pid in locked_starter_set:
            lb_arr[n + i] = 1.0    # must be a starter

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

    model_trained_at = dc.get("trained_at", None)

    out: dict = {
        "squad":            squad,
        "starters":         starters,
        "bench":            bench,
        "total_pts":        round(sum(p["projected_pts"] for p in starters) + captain["projected_pts"], 1),
        "total_cost":       int(sum(p["price"] for p in squad)),
        "captain":          captain,
        "booster":          booster,
        "model_trained_at": model_trained_at,
    }

    if booster == "12th_man":
        squad_ids = {p["id"] for p in squad}
        # Sort by pts_per_match: the chip applies for one matchday, not the full group stage.
        external  = sorted(
            [p for p in players if p["id"] not in squad_ids],
            key=lambda p: p.get("pts_per_match", 0.0), reverse=True,
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

    elif booster == "clean_sheet_shield":
        gk_def = [p for p in starters if p["pos"] in ("GK", "DEF")]
        out["cs_shield_breakdown"] = [
            {"name": p["name"], "team": p["team"], "pos": p["pos"],
             "concede_saved": p.get("concede_saved", 0.0)}
            for p in sorted(gk_def, key=lambda x: x.get("concede_saved", 0.0), reverse=True)
        ]
        out["cs_shield_total"] = round(sum(p.get("concede_saved", 0.0) for p in gk_def), 2)

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



def get_projected_players(
    predictor=None,
    matchday: int | None = None,
    matchdays: list[int] | None = None,
) -> list[dict]:
    """All fantasy players with projected points — used to populate the squad builder.

    matchday=N      → project single matchday only (e.g. captain advice)
    matchdays=[1,2] → project over those matchdays only (e.g. projected vs actual)
    neither         → all 3 group matches (full group-stage view)
    """
    conn    = sqlite3.connect(DB_PATH)
    players = _load_players(conn)
    conn.close()
    if not players:
        return []
    dc         = _load_dc()
    qual_probs = _get_qual_probs(predictor)
    players    = _project_mc(players, dc, qual_probs=qual_probs, matchday=matchday, matchdays=matchdays)
    players.sort(key=lambda p: p.get("projected_pts", 0.0), reverse=True)
    return players
