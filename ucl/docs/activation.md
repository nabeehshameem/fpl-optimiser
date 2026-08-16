# UCL Activation Runbook

Checklist to run before MD1 predictions go live. Steps are ordered by dependency;
[HUMAN] steps require the API key or manual verification.

---

## Pre-conditions

- All five test suites pass (run from the repo root):
  ```bash
  python ucl/scripts/test_ucl_ingest.py
  python ucl/scripts/test_ucl_train_dc.py
  python ucl/scripts/test_simulate_league_phase.py
  python ucl/scripts/test_simulate_bracket.py
  python ucl/scripts/test_ucl_api.py
  ```
- Bake-off complete and winner recorded in `ucl/docs/model_design.md` with numbers.
- API key in `.env` as `RAPIDAPI_KEY` (see `reference_rapidapi_key.md` in memory).

---

## Step 1 — Historical ingest [HUMAN]

Populate `data/ucl.db` with 2025/26 league-phase results for the bake-off, and
2026/27 fixtures for the live model.

```bash
# Initialise the DB schema (idempotent)
python ucl/init_db.py

# 2025/26 archive (for bake-off and DC training)
python ucl/ingest_fixtures.py --season 2025
python ucl/ingest_results.py --season 2025

# 2026/27 live fixtures
python ucl/ingest_fixtures.py --season 2026
```

Verify: `data/ucl.db` exists, `sqlite3 data/ucl.db "SELECT COUNT(*) FROM fixtures"` returns non-zero.

---

## Step 2 — Bake-off [HUMAN then Claude Code]

Run the three-model comparison on held-out 2025/26 league-phase data.

```bash
# Not yet written — ucl/run_bakeoff.py is a Phase 2 deliverable.
# Sequence once it exists:
python ucl/run_bakeoff.py --season 2025
```

Paste the output into `ucl/docs/model_design.md` under "Bake-off > Numbers".
Record the winner (`DC-xG`, `DC-goals`, or `Elo baseline`). The winner's params
file becomes the live model.

---

## Step 3 — Train the DC model on 2025/26 + any available 2026/27 data

```bash
# xG path (default, preferred if API provides xG)
python ucl/train_dc.py --season 2026

# Actual-goals path (for comparison)
python ucl/train_dc.py --season 2026 --use-actual-goals
```

The winner from the bake-off determines which flag to use (or neither for xG-default).
Output: `models/ucl_dc_params.json`.

Verify params are committed and not dirty:
```bash
git status -- models/ucl_dc_params.json
# must be clean
```

---

## Step 4 — Generate pre-MD1 predictions

```bash
# Scoreline predictions for all upcoming league-phase fixtures
python ucl/run_predictions.py

# Monte-Carlo qualification table (requires league-phase fixtures in DB)
python ucl/simulate_league_phase.py
```

Output:
- `predictions/ucl/all_upcoming_predictions.json`
- `predictions/ucl/league_phase_predictions.json`
- `predictions/ucl/league_phase_sim.json`

---

## Step 5 — Commit and push

```bash
git add predictions/ucl/ models/ucl_dc_params.json
git commit -m "UCL MD1: pre-match predictions + league phase sim"
git push origin main
git ls-remote origin HEAD  # verify SHA matches local HEAD
```

---

## Step 6 — Dry run [HUMAN]

Verify all UCL endpoints are live on Railway:

```bash
curl -s https://fpl-optimiser-production.up.railway.app/api/ucl/predictions/upcoming | python -m json.tool | head -20
curl -s https://fpl-optimiser-production.up.railway.app/api/ucl/league-phase/sim | python -m json.tool | head -10
```

Both must return 200 with `generated_at_utc` in the response.

---

## Post-MD1: bracket simulation

After the league phase concludes (16 matches per team, 36 teams = 96 + 48 = 144 fixtures total in the new format):

```bash
python ucl/simulate_bracket.py
git add predictions/ucl/bracket.json
git commit -m "UCL: bracket simulation — play-off + knockout win probabilities"
git push origin main
```

The `/api/ucl/bracket` endpoint serves this automatically.

---

## Ongoing: matchday refresh

After each matchday, run:
```bash
python ucl/ingest_results.py
python ucl/train_dc.py --season 2026   # optional: refit if >= 4 new results
python ucl/run_predictions.py
python ucl/simulate_league_phase.py
git add predictions/ucl/ models/ucl_dc_params.json
git commit -m "UCL MD{N}: post-match results + updated projections"
git push origin main
```

Retraining rule: only refit when you have ≥ 4 new results to add. A single
matchday outlier should not move parameters (same principle as the WC2026 lesson
— see `feedback_retraining_sample_size.md`).

---

## Rollback

If any step fails and MD1 is imminent:

1. Delete the bad JSON from `predictions/ucl/` and push.
2. The `/api/ucl/predictions/upcoming` endpoint returns 404 gracefully — no crash.
3. No UI component blocks on UCL data: the FPL and WC sections are independent.
