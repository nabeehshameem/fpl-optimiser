# Transfer Horizon Decay Evaluation

**Constant:** `_DECAY = 0.85` in `src/optimiser.py` (`optimise_with_transfers`)  
**Status:** Pending backtest — value unchanged since initial implementation  
**Revisit:** GW10 2026/27 (October 2026)

---

## What this constant does

When `horizon > 1` GW is passed to `optimise_with_transfers`, the model
amortises a transfer hit across multiple gameweeks by scaling predicted
points:

```python
df["predicted_points"] *= sum(_DECAY ** k for k in range(horizon))
```

For `horizon=2`, this multiplies predicted points by `1 + 0.85 = 1.85`.
For `horizon=3`, by `1 + 0.85 + 0.72 = 2.57`.

The intent: a transfer hit costs 4 points now. If you gain G extra points
per GW for the next H gameweeks (discounted), the hit is worth it when
`sum(_DECAY^k * G for k in range(H)) > 4`.

The decay factor models the uncertainty that the new player will maintain
their form — future GW projections are worth less than this GW's.

---

## Rationale for 0.85

An 0.85 per-GW decay means a 15% discount per gameweek — roughly the
uncertainty in form maintenance. This was chosen to match intuition
(~"one gameweek of form is worth about 85% of the current one") without
calibration against actual player consistency data.

---

## How to evaluate

This constant is only relevant when `horizon > 1`, which is currently only
used in the transfer planner (not in the weekly lock). Evaluation requires:

1. Identify players where a `horizon=2` or `horizon=3` transfer was
   recommended vs. `horizon=1`.
2. Track whether the 2-3 GW gain actually materialised.
3. Calibrate _DECAY so that the model-recommended hits correspond to the
   actual observed gain.

Alternative approach: set _DECAY = 1.0 (no decay, pure expected value) and
compare to 0.85 on held-out data. If 0.85 better predicts the actual outcome
of planned vs. reactive transfers, it is vindicated.

**Note:** The current weekly lock always uses `horizon=1`, so this constant
has zero effect on the production model's picks. It only matters if the
transfer planner is used with a multi-GW horizon.
