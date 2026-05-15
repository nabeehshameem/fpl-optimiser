"""
chip_planner.py
Multi-gameweek chip timing advisor.

Scores each upcoming GW for each chip type and recommends the best window to
use each one. Runs the ML predictor for each GW in the horizon (form features
are frozen at last known results, which is correct — we don't have future data).

Chips supported:
  TC  – Triple Captain  (3× instead of 2×)
  BB  – Bench Boost     (bench players score normally)
  FH  – Free Hit        (temp full squad rebuild for one GW)
  WC  – Wildcard        (permanent rebuild; score by multi-GW gain if used now)
"""

import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "fpl.db"

ALL_CHIPS = ("tc", "bb", "fh", "wc")

# Eligibility filter re-used from optimiser constants
MIN_QUALIFYING_GAMES_3 = 2
MIN_CHANCE_OF_PLAYING = 75


def _get_current_gw() -> int:
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT gameweek_id FROM gameweeks WHERE is_next = 1 LIMIT 1"
    ).fetchone()
    conn.close()
    if row:
        return row[0]
    raise RuntimeError("No upcoming gameweek found in DB.")


def _eligible(pred: pd.DataFrame) -> pd.DataFrame:
    return pred[
        (pred["num_fixtures"] > 0)
        & (pred["qualifying_games_3"] >= MIN_QUALIFYING_GAMES_3)
        & (pred["chance_of_playing_next"] >= MIN_CHANCE_OF_PLAYING)
    ]


def _squad_pts(pred: pd.DataFrame, squad_ids: list) -> float:
    """Rough expected XI points for a given squad: top-11 by predicted_points."""
    sp = pred[pred["player_id"].isin(squad_ids)]
    return float(sp.nlargest(11, "predicted_points")["predicted_points"].sum())


def _optimal_pts(pred: pd.DataFrame) -> float:
    """Expected XI points of the ILP-optimal fresh squad for this GW."""
    from src.optimiser import SquadOptimiser
    try:
        result = SquadOptimiser().optimise(pred)
        return float(result["expected_points"])
    except Exception:
        # Fall back to naive top-11 if ILP fails (e.g., too few eligible players)
        return float(_eligible(pred).nlargest(11, "predicted_points")["predicted_points"].sum())


def plan_chips(
    current_squad: Optional[list],
    chips_available: Optional[list] = None,
    horizon: int = 5,
    current_gw: Optional[int] = None,
) -> dict:
    """
    Score each upcoming GW for each available chip and return recommendations.

    Args:
        current_squad:   list of 15 player_ids you currently own.
                         Required for BB, FH, WC; ignored for TC.
        chips_available: subset of ('tc', 'bb', 'fh', 'wc') to evaluate.
                         Defaults to all four.
        horizon:         number of GWs to look ahead (default 5).
        current_gw:      override the next-GW detected from DB.

    Returns:
        dict  chip_name -> {
            chip:            display name,
            recommended_gw:  int,
            reason:          str,
            scores_by_gw:    {gw: {...}},
        }
    """
    from src.ml_predictor import LightGBMPredictor

    chips_available = [c.lower() for c in (chips_available or ALL_CHIPS)]
    if current_gw is None:
        current_gw = _get_current_gw()

    squad_chips = {"bb", "fh", "wc"}
    needs_squad = bool(set(chips_available) & squad_chips)
    if needs_squad and (current_squad is None or len(current_squad) != 15):
        raise ValueError(
            "current_squad (15 player_ids) is required for BB / FH / WC chip planning."
        )

    predictor = LightGBMPredictor()
    predictor.load()

    gw_range = list(range(current_gw, current_gw + horizon))
    print(f"  Building predictions for GW{gw_range[0]}-GW{gw_range[-1]}...")
    gw_preds: dict[int, pd.DataFrame] = {}
    for gw in gw_range:
        gw_preds[gw] = predictor.predict_all(target_gw=gw)

    results: dict = {}

    # ── Triple Captain ────────────────────────────────────────────────────
    if "tc" in chips_available:
        scores: dict[int, dict] = {}
        for gw in gw_range:
            cands = _eligible(gw_preds[gw])
            if cands.empty:
                continue
            best = cands.nlargest(1, "predicted_points").iloc[0]
            # TC nets +1× best player's pts on top of normal 2× captaincy
            scores[gw] = {
                "gw": gw,
                "best_player": int(best["player_id"]),
                "predicted_pts": round(float(best["predicted_points"]), 2),
                "num_fixtures": int(best["num_fixtures"]),
                "tc_bonus_pts": round(float(best["predicted_points"]), 2),
            }
        if scores:
            best_gw = max(scores, key=lambda g: scores[g]["tc_bonus_pts"])
            s = scores[best_gw]
            results["tc"] = {
                "chip": "Triple Captain",
                "recommended_gw": best_gw,
                "reason": (
                    f"GW{best_gw}: best captain scores ~{s['predicted_pts']:.1f} pts "
                    f"({s['num_fixtures']} fixture{'s' if s['num_fixtures'] > 1 else ''}). "
                    f"TC nets you an extra {s['tc_bonus_pts']:.1f} pts vs normal captaincy."
                ),
                "scores_by_gw": scores,
            }

    # ── Bench Boost ───────────────────────────────────────────────────────
    if "bb" in chips_available and current_squad:
        scores = {}
        for gw in gw_range:
            sp = gw_preds[gw][gw_preds[gw]["player_id"].isin(current_squad)]
            if sp.empty:
                continue
            sorted_pts = sp["predicted_points"].sort_values(ascending=False).tolist()
            # With BB, bench players score 1× instead of ~0.1×, so gain ≈ 0.9× bench pts
            bench_pts = sum(sorted_pts[11:]) if len(sorted_pts) > 11 else 0.0
            bb_gain = round(bench_pts * 0.9, 2)
            scores[gw] = {
                "gw": gw,
                "total_squad_pts": round(sum(sorted_pts), 2),
                "bench_pts": round(bench_pts, 2),
                "bb_gain": bb_gain,
                "num_dgw_players": int((sp["num_fixtures"] >= 2).sum()),
            }
        if scores:
            best_gw = max(scores, key=lambda g: scores[g]["bb_gain"])
            s = scores[best_gw]
            results["bb"] = {
                "chip": "Bench Boost",
                "recommended_gw": best_gw,
                "reason": (
                    f"GW{best_gw}: bench expected to score {s['bench_pts']:.1f} pts "
                    f"(+{s['bb_gain']:.1f} net gain vs normal 0.1 weight). "
                    f"{s['num_dgw_players']} DGW player(s) in squad."
                ),
                "scores_by_gw": scores,
            }

    # ── Free Hit ──────────────────────────────────────────────────────────
    if "fh" in chips_available and current_squad:
        scores = {}
        for gw in gw_range:
            current_xi_pts = _squad_pts(gw_preds[gw], current_squad)
            optimal_xi_pts = _optimal_pts(gw_preds[gw])
            gain = round(optimal_xi_pts - current_xi_pts, 2)
            scores[gw] = {
                "gw": gw,
                "current_squad_pts": round(current_xi_pts, 2),
                "optimal_pts": round(optimal_xi_pts, 2),
                "fh_gain": gain,
                "num_current_dgw": int(
                    gw_preds[gw][gw_preds[gw]["player_id"].isin(current_squad)]["num_fixtures"]
                    .ge(2).sum()
                ),
            }
        if scores:
            best_gw = max(scores, key=lambda g: scores[g]["fh_gain"])
            s = scores[best_gw]
            results["fh"] = {
                "chip": "Free Hit",
                "recommended_gw": best_gw,
                "reason": (
                    f"GW{best_gw}: optimal squad scores {s['optimal_pts']:.1f} vs your "
                    f"current squad's {s['current_squad_pts']:.1f} "
                    f"(+{s['fh_gain']:.1f} gain). "
                    f"Your squad has {s['num_current_dgw']} DGW player(s) this GW."
                ),
                "scores_by_gw": scores,
            }

    # ── Wildcard ──────────────────────────────────────────────────────────
    # WC is a permanent rebuild, so the gain compounds over remaining GWs.
    # Score each possible activation GW by the cumulative gain from that point onward.
    if "wc" in chips_available and current_squad:
        scores = {}
        for start_gw in gw_range:
            remaining_gws = [g for g in gw_range if g >= start_gw]
            cumulative_gain = 0.0
            for gw in remaining_gws:
                current_xi_pts = _squad_pts(gw_preds[gw], current_squad)
                optimal_xi_pts = _optimal_pts(gw_preds[gw])
                cumulative_gain += max(0.0, optimal_xi_pts - current_xi_pts)
            scores[start_gw] = {
                "activate_gw": start_gw,
                "cumulative_gain": round(cumulative_gain, 2),
                "gws_covered": len(remaining_gws),
            }
        if scores:
            best_gw = max(scores, key=lambda g: scores[g]["cumulative_gain"])
            s = scores[best_gw]
            results["wc"] = {
                "chip": "Wildcard",
                "recommended_gw": best_gw,
                "reason": (
                    f"Activate in GW{best_gw} for a cumulative +{s['cumulative_gain']:.1f} pts "
                    f"over the next {s['gws_covered']} GW(s) vs keeping current squad."
                ),
                "scores_by_gw": scores,
            }

    return results


def print_chip_plan(plan: dict) -> None:
    """Pretty-print the chip plan returned by plan_chips()."""
    CHIP_ORDER = ["tc", "bb", "fh", "wc"]
    for key in CHIP_ORDER:
        if key not in plan:
            continue
        rec = plan[key]
        print(f"\n  {rec['chip'].upper()}")
        print(f"  Recommended GW: {rec['recommended_gw']}")
        print(f"  Reason: {rec['reason']}")
        print("  GW breakdown:")
        for gw, s in rec["scores_by_gw"].items():
            # Pick the most relevant score field for each chip
            score_field = {
                "tc": "tc_bonus_pts", "bb": "bb_gain",
                "fh": "fh_gain", "wc": "cumulative_gain",
            }.get(key)
            score = s.get(score_field, 0)
            marker = " <-- recommended" if gw == rec["recommended_gw"] else ""
            print(f"    GW{gw}: {score:+.1f} pts{marker}")
