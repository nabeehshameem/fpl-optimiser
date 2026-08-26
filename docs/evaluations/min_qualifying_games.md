# MIN_QUALIFYING_GAMES Evaluation

**Constants:** `MIN_QUALIFYING_GAMES_3 = 2`, `MIN_QUALIFYING_GAMES_5 = 2` in `src/optimiser.py`  
**Status:** Pending backtest — values unchanged since initial implementation  
**Revisit:** GW10 2026/27 (October 2026), when ≥ 10 GWs of player history are available

---

## What these constants do

Players are only considered eligible for squad selection if they have appeared
in at least `MIN_QUALIFYING_GAMES_3` of the last 3 gameweeks AND at least
`MIN_QUALIFYING_GAMES_5` of the last 5 gameweeks.

Both are set to 2, meaning a player must have played in ≥ 2 of the last 3 GWs
and ≥ 2 of the last 5 GWs. The intent is to exclude:

- Players who are highly rotation-prone (< 67% appearance rate in recent form)
- Players who have been absent through injury or suspension and may not be
  fit at deadline

The two-window check (short + medium) guards against players who played twice
in the last 5 but not recently (false positive on form) and players who played
two in a row after a long absence (still a rotation risk).

---

## Rationale for value 2

A threshold of 2/3 (~67%) was set by inspection rather than data. It is
intended to be tighter than 1/3 (too permissive) and looser than 3/3 (too
strict — excludes valid rotation-cover picks). The value has not been
backtested against actual season outcomes.

---

## How to evaluate

Run `scripts/eval_bench_weight.py`-style analysis varying the threshold
against held-out 2026/27 data:

1. For values in {1, 2, 3} (for both 3-GW and 5-GW windows):
   - Build optimal squads using DC predictions
   - Score against actual GW results
   - Compare: does a tighter filter improve average squad score by reducing
     rotation penalties, or does it over-restrict the eligible pool?

2. Also measure pool coverage: what % of top-30 xPts players are excluded
   at each threshold level?

Expected finding: the current value is a reasonable default. Any deviation
should show a statistically significant improvement over ≥ 10 GWs.
