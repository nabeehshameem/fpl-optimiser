"""
grade_retrospective.py

Walk-forward grading of the DC model over WC2026, reconstructed from git history.

For every played match (match_stats), this script:
  1. Finds the latest commit that touched wc/models/dc_params.json strictly
     BEFORE the match date (UTC midnight cutoff -> >=16h margin before any
     kickoff, so no result leakage is possible).
  2. Extracts BOTH dc_params.json AND score_predictor.py from that commit
     (so mid-tournament code changes are honoured, not just param changes).
  3. Runs predict() exactly as the live site would have at that time.
  4. Grades H/D/A probabilities and top scoreline against the 90-minute result.

Also grades the FROZEN pre-tournament model (last commit before June 11)
on every match, so we can answer: did in-tournament retraining help?

Outputs:
  wc/retrospective/per_match.csv
  wc/retrospective/summary.txt
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent   # repo root
DB_HEAD = REPO / "wc" / "data" / "wc.db"
OUT_DIR = REPO / "wc" / "retrospective"
OUT_DIR.mkdir(exist_ok=True)

TOURNAMENT_START = "2026-06-11"
N_GROUP_MATCHES = 72  # 48-team format: 72 group matches, then knockouts


def sh(*args: str) -> str:
    return subprocess.run(args, cwd=REPO, capture_output=True, text=True,
                          encoding="utf-8", check=True).stdout


# ── 1. Commit timeline for dc_params.json ────────────────────────────────────

def params_commits() -> list[tuple[datetime, str]]:
    """[(commit_utc_datetime, hash)] ascending, for commits touching dc_params.json."""
    raw = sh("git", "log", "--format=%H %aI", "--", "wc/models/dc_params.json")
    out = []
    for line in raw.strip().splitlines():
        h, iso = line.split()
        dt = datetime.fromisoformat(iso).astimezone(timezone.utc)
        out.append((dt, h))
    out.sort(key=lambda x: x[0])
    return out


def commit_before(commits: list[tuple[datetime, str]], match_date: str) -> tuple[datetime, str]:
    """Latest params commit strictly before match_date 00:00 UTC."""
    cutoff = datetime.fromisoformat(match_date + "T00:00:00+00:00")
    chosen = None
    for dt, h in commits:
        if dt < cutoff:
            chosen = (dt, h)
        else:
            break
    if chosen is None:
        raise RuntimeError(f"No params commit before {match_date}")
    return chosen


# ── 2. Load a historical model (code + params from one commit) ──────────────

_model_cache: dict[str, object] = {}


def load_model_at(commit: str):
    """Instantiate DCPredictor using score_predictor.py + dc_params.json from `commit`."""
    if commit in _model_cache:
        return _model_cache[commit]

    tmp = Path(tempfile.mkdtemp(prefix=f"dc_{commit[:8]}_"))
    (tmp / "wc" / "src").mkdir(parents=True)
    (tmp / "wc" / "models").mkdir(parents=True)
    (tmp / "wc" / "data").mkdir(parents=True)

    code = sh("git", "show", f"{commit}:wc/src/score_predictor.py")
    params = sh("git", "show", f"{commit}:wc/models/dc_params.json")
    (tmp / "wc" / "src" / "score_predictor.py").write_text(code, encoding="utf-8")
    (tmp / "wc" / "models" / "dc_params.json").write_text(params, encoding="utf-8")
    # point DB at HEAD db (only used for rank priors / display names)
    try:
        (tmp / "wc" / "data" / "wc.db").symlink_to(DB_HEAD)
    except OSError:
        shutil.copy2(DB_HEAD, tmp / "wc" / "data" / "wc.db")

    spec = importlib.util.spec_from_file_location(
        f"sp_{commit[:8]}", tmp / "wc" / "src" / "score_predictor.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]

    p = mod.DCPredictor()
    p.load()
    # canonical name -> team_id, from the params file itself
    name_to_id = {v: int(k) for k, v in json.loads(params).get("id_to_name", {}).items()}
    _model_cache[commit] = (p, mod, name_to_id)
    return _model_cache[commit]


# ── 3. Matches to grade ──────────────────────────────────────────────────────

def played_matches() -> list[dict]:
    conn = sqlite3.connect(DB_HEAD)
    rows = conn.execute("""
        SELECT match_date, home_team, away_team, home_score, away_score,
               went_to_et, home_score_90, away_score_90
        FROM match_stats
        WHERE home_score IS NOT NULL AND away_score IS NOT NULL
        ORDER BY match_date, stat_id
    """).fetchall()
    conn.close()
    out = []
    for i, (d, h, a, hg, ag, et, hg90, ag90) in enumerate(rows):
        g_h = hg90 if (et and hg90 is not None) else hg
        g_a = ag90 if (et and ag90 is not None) else ag
        out.append({
            "n": i + 1,
            "date": str(d)[:10],
            "home": h, "away": a,
            "hg": int(g_h), "ag": int(g_a),          # 90-min score (grading target)
            "hg_ft": int(hg), "ag_ft": int(ag),      # full result incl. ET
            "et": bool(et),
            "stage": "group" if i < N_GROUP_MATCHES else "knockout",
        })
    return out


# ── 4. Prediction + grading ──────────────────────────────────────────────────

# HEAD's alias table — name normalisation only, contains no predictive information
sys.path.insert(0, str(REPO))
from wc.src.score_predictor import _canonical as _canonical_head  # noqa: E402


def predict_match(model_tuple, home: str, away: str) -> dict | None:
    p, mod, name_to_id = model_tuple
    hk, ak = _canonical_head(home), _canonical_head(away)
    hid, aid = name_to_id.get(hk), name_to_id.get(ak)
    if hid is None or aid is None:
        return None
    try:
        r = p.predict(hid, aid)
    except TypeError:
        r = p.predict(hid, aid, False)  # older signature
    top = r["most_likely"][0]
    return {
        "p_h": r["win_pct"] / 100.0,
        "p_d": r["draw_pct"] / 100.0,
        "p_a": r["loss_pct"] / 100.0,
        "score_h": int(top[0]), "score_a": int(top[1]),
        "xg_h": r.get("home_xg"), "xg_a": r.get("away_xg"),
    }


def outcome(hg: int, ag: int) -> str:
    return "H" if hg > ag else ("A" if ag > hg else "D")


def grade(pred: dict, hg: int, ag: int) -> dict:
    res = outcome(hg, ag)
    y = {"H": 0, "D": 0, "A": 0}
    y[res] = 1
    probs = {"H": pred["p_h"], "D": pred["p_d"], "A": pred["p_a"]}
    brier = sum((probs[k] - y[k]) ** 2 for k in y)
    ll = -math.log(max(probs[res], 1e-12))
    pick = max(probs, key=probs.get)
    return {
        "result": res,
        "brier": brier,
        "logloss": ll,
        "pick": pick,
        "pick_correct": int(pick == res),
        "p_result": probs[res],
        "exact_score": int(pred["score_h"] == hg and pred["score_a"] == ag),
    }


# ── 5. Baselines ─────────────────────────────────────────────────────────────

def wc_base_rates() -> tuple[float, float, float]:
    """H/D/A rates from WC2018+WC2022 group+KO 90-min results in the matches table."""
    conn = sqlite3.connect(DB_HEAD)
    rows = conn.execute(
        "SELECT home_score, away_score FROM matches WHERE home_score IS NOT NULL"
    ).fetchall()
    conn.close()
    n = len(rows)
    h = sum(1 for a, b in rows if a > b) / n
    d = sum(1 for a, b in rows if a == b) / n
    return h, d, 1 - h - d


# ── 6. Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    commits = params_commits()
    frozen = commit_before(commits, TOURNAMENT_START)
    print(f"{len(commits)} params commits; frozen pre-tournament model = "
          f"{frozen[1][:8]} @ {frozen[0]:%Y-%m-%d %H:%M UTC}")

    matches = played_matches()
    print(f"{len(matches)} played matches to grade\n")

    bh, bd, ba = wc_base_rates()
    rows_out = []

    for m in matches:
        wf_dt, wf_hash = commit_before(commits, m["date"])
        wf_pred = predict_match(load_model_at(wf_hash), m["home"], m["away"])
        fr_pred = predict_match(load_model_at(frozen[1]), m["home"], m["away"])
        if wf_pred is None or fr_pred is None:
            print(f"  SKIP (team unmapped): {m['home']} vs {m['away']}")
            continue

        g_wf = grade(wf_pred, m["hg"], m["ag"])
        g_fr = grade(fr_pred, m["hg"], m["ag"])

        # baselines graded on same match
        res = g_wf["result"]
        y = {"H": 0, "D": 0, "A": 0}; y[res] = 1
        brier_uniform = sum((1/3 - y[k]) ** 2 for k in y)
        base = {"H": bh, "D": bd, "A": ba}
        brier_base = sum((base[k] - y[k]) ** 2 for k in y)

        rows_out.append({
            "n": m["n"], "date": m["date"], "stage": m["stage"],
            "home": m["home"], "away": m["away"],
            "score_90": f"{m['hg']}-{m['ag']}", "result": res, "et": int(m["et"]),
            "model_commit": wf_hash[:8],
            "commit_utc": f"{wf_dt:%Y-%m-%d %H:%M}",
            "p_h": round(wf_pred["p_h"], 4), "p_d": round(wf_pred["p_d"], 4),
            "p_a": round(wf_pred["p_a"], 4),
            "pred_score": f"{wf_pred['score_h']}-{wf_pred['score_a']}",
            "pick": g_wf["pick"], "pick_correct": g_wf["pick_correct"],
            "exact_score": g_wf["exact_score"],
            "brier": round(g_wf["brier"], 4), "logloss": round(g_wf["logloss"], 4),
            "p_result": round(g_wf["p_result"], 4),
            "frozen_brier": round(g_fr["brier"], 4),
            "frozen_pick_correct": g_fr["pick_correct"],
            "frozen_exact": g_fr["exact_score"],
            "brier_uniform": round(brier_uniform, 4),
            "brier_baserate": round(brier_base, 4),
        })

    # ── write per-match csv ──
    csv_path = OUT_DIR / "per_match.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)

    # ── summary ──
    def agg(rows, key):
        return float(np.mean([r[key] for r in rows]))

    def block(name, rows):
        n = len(rows)
        lines = [
            f"{name} (n={n})",
            f"  Walk-forward : Brier {agg(rows,'brier'):.4f} | logloss {agg(rows,'logloss'):.4f}"
            f" | outcome acc {agg(rows,'pick_correct'):.1%} | exact score {agg(rows,'exact_score'):.1%}",
            f"  Frozen 10-Jun: Brier {agg(rows,'frozen_brier'):.4f}"
            f" | outcome acc {agg(rows,'frozen_pick_correct'):.1%} | exact {agg(rows,'frozen_exact'):.1%}",
            f"  Baselines    : uniform Brier {agg(rows,'brier_uniform'):.4f}"
            f" | WC-base-rate Brier {agg(rows,'brier_baserate'):.4f}",
        ]
        return "\n".join(lines)

    grp = [r for r in rows_out if r["stage"] == "group"]
    ko = [r for r in rows_out if r["stage"] == "knockout"]

    # calibration: bucket every (prob, hit) pair across H/D/A
    pairs = []
    for r in rows_out:
        for k, pk in (("H", "p_h"), ("D", "p_d"), ("A", "p_a")):
            pairs.append((r[pk], 1 if r["result"] == k else 0))
    buckets = {}
    for p, hit in pairs:
        b = min(int(p * 10), 9)
        buckets.setdefault(b, []).append(hit)
    calib_lines = ["Calibration (predicted prob -> observed frequency):"]
    for b in sorted(buckets):
        hits = buckets[b]
        calib_lines.append(
            f"  {b*10:2d}-{b*10+10:3d}%  n={len(hits):3d}  observed {np.mean(hits):.1%}"
        )

    summary = "\n\n".join([
        f"WC2026 MODEL RETROSPECTIVE — walk-forward reconstruction from git history",
        f"Frozen pre-tournament model: commit {frozen[1][:8]} ({frozen[0]:%Y-%m-%d %H:%M} UTC)",
        block("ALL MATCHES", rows_out),
        block("GROUP STAGE", grp),
        block("KNOCKOUTS (90-min result)", ko),
        "\n".join(calib_lines),
    ])
    (OUT_DIR / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary)
    print(f"\nWrote {csv_path} and {OUT_DIR/'summary.txt'}")


if __name__ == "__main__":
    main()
