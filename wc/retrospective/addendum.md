# Addendum — the on-the-record calls, graded

*Appended after the tournament. The predictions above are unchanged, as promised.*

## Bronze final: France vs England — MISSED

We said: France 44% / draw 33% / England 23%, most likely 1–1, France 62.5%
to finish third after extra time and penalties.

What happened: **England won.** Our least likely 90-minute outcome. For the
record, the bookmakers leaned France too — this was a consensus miss, not a
contrarian one — but a miss is a miss, and it goes in the ledger at full
weight like everything else. One plausible read: third-place playoffs are
low-stakes exhibitions where motivation and heavy rotation swamp team
strength, exactly the kind of context a strength-based model prices worst.
We flagged the same blind spot after the group stage (minnows parking the
bus); "matches where the model's inputs stop describing the match" is now
officially our known weakness, twice documented.

## The final: Spain vs Argentina — Spain win, 1–0 AET

We said: Spain 43.7% / draw 38.9% / Argentina 17.4% on 90 minutes; Spain
67% to lift the trophy; most likely score 0–0.

What happened: **Spain lifted it, 1–0 after extra time.** At 90 minutes
the score was 0–0 — exactly what the model predicted as the most likely
scoreline, and confirmed as an exact score hit in the ledger. The WDL grade
at 90 minutes is a miss: the model's plurality call was a Spain win, not the
draw that materialised. The 38.9% draw probability was the second call, and
it was right in regulation. Ferran Torres scored from the bench at the 106th
minute and Spain claimed the trophy at the odds we gave them — 67%. The
headline call landed. The 90-minute WDL did not, and both facts are in the
ledger.

## Final tournament ledger

n=104 (all WC2026 matches graded on 90-minute results):

| | Walk-forward model | Frozen pre-tournament |
|---|---|---|
| All matches (n=104) | 59.6% WDL acc, 11.5% exact | 66.3% WDL acc, 9.6% exact |
| Group stage (n=72) | 62.5% WDL acc, 11.1% exact | 65.3% WDL acc, 11.1% exact |
| Knockouts (n=32) | 53.1% WDL acc, 12.5% exact | 68.8% WDL acc, 6.2% exact |

Adding the two final matches moved overall outcome accuracy from 60.8%
(n=102) to 59.6% (n=104): both new matches were 90-minute WDL misses. Exact
score accuracy improved 0.7pp to 11.5%, because the Final's 0–0 at 90
minutes was the model's top prediction — a correct call, delivered six
minutes late.

The frozen pre-tournament model (66.3%) continued to outperform the live
walk-forward versions (59.6%) across all 104 matches, and the gap widened
in knockouts (68.8% vs 53.1%). The finding that motivated this retrospective
holds: in-tournament retraining made the model worse, not better. The right
response is a full refit on the complete tournament dataset before the next
competition, not incremental updates mid-event.

---
*This page is now frozen. The model's next public record is the 2026/27 FPL
season — every squad hash-committed before the deadline at
themodelsays.com/fpl/methodology.*
