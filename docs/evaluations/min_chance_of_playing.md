# MIN_CHANCE_OF_PLAYING Evaluation

**Constant:** `MIN_CHANCE_OF_PLAYING = 75` in `src/optimiser.py`  
**Status:** Pending backtest — value unchanged since initial implementation  
**Revisit:** GW10 2026/27 (October 2026)

---

## What this constant does

Players are excluded from squad selection if their FPL-reported
`chance_of_playing_next_round` is below 75%. FPL publishes this field as
a percentage (0, 25, 50, 75, or 100) based on manager availability reports.

The intent is to prevent the optimiser from selecting injured or suspended
players whose FPL score will likely be 1 pt (DNP) or near zero.

---

## Rationale for 75

The 75% threshold means:
- 100%: always included
- 75%: included (doubtful; includes some rotation doubt as well as injury)
- 50%: excluded (50/50 on availability — the expected value hit is large)
- 25%: excluded (very likely unavailable)
- 0%: excluded (suspended or confirmed out)

The 75 cut excludes the three lowest FPL availability tiers. This was set
by inspection: below 75% the expected value penalty from DNPs dominates
any predicted-points upside, but no backtest was run to confirm this.

---

## How to evaluate

Over 2026/27 GW1–38 data:

1. Collect all players with `chance_of_playing_next = 75` at deadline time.
2. Compute their actual GW minutes and points.
3. Compare against players at 100%:
   - Are 75%-chance players playing ≥ 60 min significantly less often?
   - Is their average GW contribution meaningfully lower?
4. If the DNP rate at 75% is < 15%, the threshold might be too strict (we
   are excluding players who mostly play). If > 40%, it may be too lenient.

A value of 50% should be tested as an alternative if 75% proves too strict.
