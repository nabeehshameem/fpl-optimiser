# UCL Model Design

## Purpose

Dixon-Coles predictions for the UEFA Champions League 2026/27 season. Same engine as the FPL and WC2026 verticals; parameters and data sources differ because the sample size is much smaller (~125–160 league-phase + knockout fixtures per season vs 380 PL matches).

---

## Architecture

```
API-Football (SportAPI7)
    ↓ ucl/ingest_fixtures.py       fixture schedule + results
    ↓ ucl/ingest_results.py        per-match stats + xG
    ↓ ucl/init_db.py               data/ucl.db
    ↓ ucl/train_dc.py              models/ucl_dc_params.json
    ↓ ucl/run_predictions.py       predictions/ucl/*_predictions.json
    ↓ ucl/simulate_league_phase.py predictions/ucl/league_phase_sim.json
    ↓ ucl/simulate_bracket.py      predictions/ucl/bracket.json
    ↓ src/ucl_api.py               /api/ucl/* (Railway, read-only JSON)
    ↓ themodelsays-web             UI
```

Rule 8 applies: Railway has no DB for UCL. Every endpoint serves a committed JSON file, not a live query.

---

## DC Model

### Response variable

xG is used where API-Football provides it; falls back to actual goals where xG is missing. The `--use-actual-goals` flag on `train_dc.py` overrides this for bake-off comparison.

### Regularisation

`REG_LAMBDA = 10.0` (vs 5.0 for FPL). The UCL sample is ~125–160 matches per season — about one-third the size of a PL season. Stronger regularisation prevents attack/defence parameters from collapsing to extreme values for teams with only 8 league-phase fixtures.

### Home advantage

UCL home advantage is lower than the PL. The prior in `x0` is `ln(1.10)` (~10% boost to home expected goals). The optimiser is free to shrink it; the real figure from 2024/25 UCL data is approximately 8–12%.

### Recency weighting

Exponential decay: `exp(-0.8 * days_ago / 365)`. A match played 12 months ago carries weight ≈ 0.45 relative to a match played today.

---

## Bake-off

**Gate: must pass before MD1 predictions are published.**

The bake-off compares three models on held-out UCL 2025/26 league-phase data:

| Model | Filename | Description |
|-------|----------|-------------|
| DC-xG | `ucl_dc_params_xg.json` | DC fitted on xG where available |
| DC-goals | `ucl_dc_params_goals.json` | DC fitted on actual goals only |
| Elo baseline | _(computed inline)_ | ELO ratings from 538-style prior |

**Evaluation:** W/D/L prediction accuracy on the 2025/26 league phase (all 96 fixtures). The winner is the model with the highest overall accuracy. Ties broken by top-scoreline accuracy (exact 0-0, 1-0, etc.).

**Status:** [PENDING — needs historical ingest. Sequence: [HUMAN] provides API key + runs `ucl/ingest_fixtures.py --season 2025` and `ucl/ingest_results.py --season 2025`, then bake-off runs here and the winner's short_name is recorded below.]

**Winner:** TBD

**Numbers:**
```
DC-xG:    ??/96 = ??%
DC-goals: ??/96 = ??%
Elo:      ??/96 = ??%
```

---

## Test suites

Run all before MD1:

```bash
python ucl/test_ucl_ingest.py
python ucl/test_ucl_train_dc.py
python ucl/test_simulate_league_phase.py
python ucl/test_simulate_bracket.py
python ucl/test_ucl_api.py
```

The ingest tests (I suite) cover DB schema and structure. The train tests (T suite) cover the fit() function with synthetic data. The simulation tests (L/B suites) cover the Monte Carlo engine with synthetic DB fixtures. The API tests (A suite) cover the ucl_api.py router with mocked file paths.

None of the above require the API key. The bake-off script (`ucl/run_bakeoff.py` — not yet written) requires the 2025/26 historical data and is [HUMAN]-gated.

---

## Activation runbook

See `ucl/docs/activation.md` (Phase 2 deliverable — not yet written).

Pre-MD1 checklist:
1. [ ] All five test suites pass
2. [ ] Bake-off complete, winner recorded above with numbers
3. [ ] `ucl/ingest_fixtures.py --season 2026` run, `data/ucl.db` committed
4. [ ] `ucl/train_dc.py` run on 2025/26 archive, `models/ucl_dc_params.json` committed
5. [ ] `ucl/run_predictions.py` generates `predictions/ucl/league_phase_predictions.json`
6. [ ] `ucl/simulate_league_phase.py` generates `predictions/ucl/league_phase_sim.json`
7. [ ] All six files committed and pushed before MD1 kickoff
8. [ ] [HUMAN] dry run: verify /api/ucl/predictions/league_phase and /api/ucl/league-phase/sim return 200 on Railway
