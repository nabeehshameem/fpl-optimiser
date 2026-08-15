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


def build_predictions(fixtures: list[dict], dc: dict) -> list[dict]:
    results = []
    for f in fixtures:
        pred = predict_match(f["home_sn"], f["away_sn"], dc)
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
        })
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", default=None,
                    help="Filter to a single UCL round (e.g. 'Round of 16')")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    dc = _load_dc()
    fixtures = _upcoming_fixtures(args.round)

    if not fixtures:
        print("No upcoming UCL fixtures found.")
        return

    predictions = build_predictions(fixtures, dc)

    # Group by round for the output filename
    rounds = sorted({p["round"] for p in predictions if p["round"]})
    label = (
        args.round.lower().replace(" ", "_")
        if args.round
        else "all_upcoming"
    )
    slug = label.replace("/", "_").replace("-", "_")

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": "ucl_dc_params",
        "rounds": rounds,
        "fixtures": predictions,
    }

    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = EXPORT_DIR / f"{slug}_predictions.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"UCL predictions exported ({len(predictions)} fixtures) → {out}")


if __name__ == "__main__":
    main()
