# BENCH_WEIGHT Evaluation

**Constant:** `BENCH_WEIGHT` in `src/optimiser.py` (appears in `optimise()` and `optimise_with_transfers()`)  
**Change:** 0.1 → 0.2 in commit `5f11918`  
**Evaluation date:** 2026-08-26  
**Verdict:** Revert to 0.1 — insufficient evidence to keep 0.2

---

## What the constant does

`BENCH_WEIGHT` is the multiplier applied to bench players' predicted points
in the ILP objective:

```
maximise: Σ pts_i·start_i + pts_i·cap_i + BENCH_WEIGHT·pts_i·(select_i − start_i)
```

A higher weight causes the optimiser to spend more budget on quality bench
players, at the potential expense of XI quality. A lower weight treats the
bench as near-free filler and concentrates budget on the starting eleven.

---

## Background

Commit `5f11918` raised the value from 0.1 to 0.2 with the comment:
*"0.2 reflects rotation value: bench players can be planned starters when XI
players face tough fixtures."* No backtest was attached.

---

## Backtest attempt

**Data source:** `data/fpl_2526.db` — full 2025/26 season archive  
**Predictions available:** naive_v1, GWs 35, 37, 38 only  
(No DC predictions were stored for 2025/26; DC was the deployed model.)

**Method:** For each GW with stored predictions:
1. Load naive_v1 predicted points + eligibility from archive snapshots.
2. Build a fresh 15-player squad with each weight (no carryover transfers).
3. Score each squad's starting XI + captain double against actual
   `player_gameweek_history` totals (auto-subs not simulated).

**Results:**

| GW | BW=0.1 | BW=0.2 | Diff | Note |
|----|--------|--------|------|------|
| 35 | — | — | — | ILP infeasible: archive DB has accumulated duplicate player rows that violate the 3-per-club constraint |
| 37 | 44 | 44 | 0 | Same squad, same score |
| 38 | — | — | — | Same infeasibility |

Only 1 of 3 GWs produced a valid comparison, and that GW showed no
difference between the two weights.

**Sample size:** 1 GW. Statistically meaningless.

---

## Why the archive produced infeasible results

The `players` table in the 2025/26 archive accumulates all players who
appeared during the season, including loans, mid-season arrivals, and
departures. Some "clubs" have up to 52 players in the table — far more than
FPL's typical 25-30 per club. The qualifying-games filter selects players
based on historical minutes, not squad membership at deadline time, which
can make the 3-per-club constraint infeasible for certain gameweeks.

---

## Verdict

The burden of proof is on the change, not the status quo. `BENCH_WEIGHT`
was raised from 0.1 to 0.2 without any supporting evaluation. With no
evidence that 0.2 outperforms 0.1 on held-out data, we revert.

**BENCH_WEIGHT reverted to 0.1** (commit following this file).

---

## When to revisit

Re-run `scripts/eval_bench_weight.py` once we have ≥ 10 graded 2026/27
gameweeks with DC predictions stored. The evaluation script is ready; it
just needs prediction data with enough coverage to avoid the archive's
infeasibility issues.

Suggested revisit: GW10 (approximately October 2026).
