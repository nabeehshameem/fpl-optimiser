"""
ucl/run_predictions.py
Generate scoreline predictions for all upcoming UCL fixtures and write
to predictions/ucl/matchday_{N}_predictions.json.

Works the same way as scripts/build_gw_tools.py for the FPL side:
  - Reads committed DC params from models/ucl_dc_params.json
  - Reads upcoming fixtures from data/ucl.db
  - Writes JSON to predictions/ucl/ (committed, served from Railway)

    python ucl/run_predictions.py               # all upcoming fixtures
    python ucl/run_predictions.py --round "Round of 16"
    python ucl/run_predictions.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

sys.stdout.reconfigure(encoding="utf-8")

from src.dc_match import predict_match  # noqa: E402

DB_PATH = PROJECT_ROOT / "data" / "ucl.db"
MODEL_PATH = PROJECT_ROOT / "models" / "ucl_dc_params.json"
EXPORT_DIR = PROJECT_ROOT / "predictions" / "ucl"


def _load_dc() -> dict:
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"UCL DC params not found: {MODEL_PATH}\n"
            "Run: python ucl/train_dc.py"
        )
    return json.loads(MODEL_PATH.read_text(encoding="utf-8"))


# A club's first UCL match is enough to give it a fitted rating, but one result
# shrunk hard toward average is not an established one. Below this many
# finished matches in the training data, a side's rating is reported as thin.
MIN_MATCHES = 6


def _match_counts() -> dict[str, int]:
    """Finished UCL matches per club, over the same rows train_dc.py fits on."""
    conn = sqlite3.connect(DB_PATH)
    try:
        rows = conn.execute("""
            SELECT th.short_name, ta.short_name
            FROM fixtures f
            JOIN teams th ON f.home_team_id = th.team_id
            JOIN teams ta ON f.away_team_id = ta.team_id
            WHERE f.status = 'FT'
              AND f.home_score IS NOT NULL AND f.away_score IS NOT NULL
        """).fetchall()
    finally:
        conn.close()
    counts: dict[str, int] = {}
    for h, a in rows:
        counts[h] = counts.get(h, 0) + 1
        counts[a] = counts.get(a, 0) + 1
    return counts


def _upcoming_fixtures(round_filter: str | None) -> list[dict]:
    if not DB_PATH.exists():
        raise RuntimeError(
            f"UCL database not found: {DB_PATH}\n"
            "Run: python ucl/init_db.py && python ucl/ingest_fixtures.py"
        )
    conn = sqlite3.connect(DB_PATH)
    try:
        where = "f.status NOT IN ('FT', 'AET', 'PEN')"
        params: tuple = ()
        if round_filter:
            where += " AND f.round_name = ?"
            params = (round_filter,)
        rows = conn.execute(f"""
            SELECT
                f.fixture_id,
                f.round_name,
                f.kickoff_utc,
                th.short_name AS home_sn,
                ta.short_name AS away_sn,
                th.name AS home_name,
                ta.name AS away_name
            FROM fixtures f
            JOIN teams th ON f.home_team_id = th.team_id
            JOIN teams ta ON f.away_team_id = ta.team_id
            WHERE {where}
            ORDER BY f.kickoff_utc
        """, params).fetchall()
    finally:
        conn.close()
    return [
        {
            "fixture_id": r[0],
            "round_name": r[1],
            "kickoff_utc": r[2],
            "home_sn": r[3],
            "away_sn": r[4],
            "home_name": r[5],
            "away_name": r[6],
        }
        for r in rows
    ]


def build_predictions(fixtures: list[dict], dc: dict,
                      counts: dict[str, int]) -> list[dict]:
    # predict_match falls back to league-average ratings for any team absent
    # from team_params. That fallback is invisible in its output, so a team
    # that has never played a UCL match under this model reads as an average
    # one. Surface it: a newly-qualified side's prediction is a prior, not a fit.
    rated = set(dc.get("team_params", {}))
    results = []
    for f in fixtures:
        pred = predict_match(f["home_sn"], f["away_sn"], dc)
        unrated = [sn for sn in (f["home_sn"], f["away_sn"]) if sn not in rated]
        thin = [sn for sn in (f["home_sn"], f["away_sn"])
                if counts.get(sn, 0) < MIN_MATCHES]
        results.append({
            "fixture_id": f["fixture_id"],
            "round": f["round_name"],
            "kickoff_utc": f["kickoff_utc"],
            "home": f["home_sn"],
            "home_name": f["home_name"],
            "away": f["away_sn"],
            "away_name": f["away_name"],
            "p_home": pred["p_home"],
            "p_draw": pred["p_draw"],
            "p_away": pred["p_away"],
            "xg_home": pred["xg_home"],
            "xg_away": pred["xg_away"],
            "top_scoreline": pred["top_scoreline"],
            "top_scoreline_pct": pred["top_scoreline_pct"],
            "home_cs_pct": pred["home_cs_pct"],
            "away_cs_pct": pred["away_cs_pct"],
            "cold_start": bool(unrated),
            "unrated_teams": unrated,
            "home_matches": counts.get(f["home_sn"], 0),
            "away_matches": counts.get(f["away_sn"], 0),
            # none: both sides unrated, so the output is the fallback constant
            # and says nothing about either team. partial: at least one side
            # rests on fewer than MIN_MATCHES results — rated, but not enough
            # to be shown with the same confidence as an established side.
            "confidence": ("none" if len(unrated) == 2
                           else "partial" if thin
                           else "rated"),
        })
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", default=None,
                    help="Filter to a single UCL round (e.g. 'Round of 16')")
    ap.add_argument("--season", type=int, default=None,
                    help="UCL season start year (informational; not yet used for DB routing)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    dc = _load_dc()
    fixtures = _upcoming_fixtures(args.round)

    if not fixtures:
        print("No upcoming UCL fixtures found.")
        return

    counts = _match_counts()
    predictions = build_predictions(fixtures, dc, counts)

    # Group by round for the output filename
    rounds = sorted({p["round"] for p in predictions if p["round"]})
    label = (
        args.round.lower().replace(" ", "_")
        if args.round
        else "all_upcoming"
    )
    slug = label.replace("/", "_").replace("-", "_")

    rated = set(dc.get("team_params", {}))
    seen = {sn for p in predictions for sn in (p["home"], p["away"])}
    unrated = sorted(seen - rated)
    thin = sorted(
        ({"team": sn, "matches": counts.get(sn, 0)}
         for sn in seen if counts.get(sn, 0) < MIN_MATCHES),
        key=lambda t: (t["matches"], t["team"]),
    )

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": "ucl_dc_params",
        "model_trained_at": dc.get("trained_at"),
        "rounds": rounds,
        "coverage": {
            "teams_total": len(seen),
            "teams_rated": len(seen & rated),
            "teams_unrated": unrated,
            "min_matches_established": MIN_MATCHES,
            "teams_established": len(seen) - len(thin),
            "teams_thin": thin,
            "cold_start_fixtures": sum(1 for p in predictions if p["cold_start"]),
            "by_confidence": {
                tier: sum(1 for p in predictions if p["confidence"] == tier)
                for tier in ("rated", "partial", "none")
            },
            "note": (
                "Ratings are fitted on completed UCL matches only. A club with "
                "no UCL history falls back to a league-average prior; a club "
                f"with fewer than {MIN_MATCHES} matches has a rating, but one "
                "shrunk heavily toward average. confidence='rated' means both "
                f"sides have {MIN_MATCHES}+ matches; 'partial' means at least "
                "one side is thin; 'none' means both are unrated and the numbers "
                "are the fallback constant — do not present those as predictions."
            ),
        },
        "fixtures": predictions,
    }

    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = EXPORT_DIR / f"{slug}_predictions.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"UCL predictions exported ({len(predictions)} fixtures) → {out}")


if __name__ == "__main__":
    main()
