# FPL season runbook — 2026/27

The code is done. The season is an operations problem: 38 hard deadlines,
most landing ~4:00–5:00am Melbourne time (FPL deadlines are 90 minutes
before the first kickoff, typically UK Friday/Saturday evening). This
document exists so no step depends on being awake.

## The core rule

**Automate the lock. Never race a deadline manually.**
The repo already has the pattern: the WC pipeline auto-commits and pushes
("Data: pipeline update …") on a schedule. Reuse it. A scheduled job
(cron on the desktop, or a Railway worker) runs the pre-deadline sequence;
you verify in the morning; manual execution is the *fallback*, not the plan.

## Weekly cadence (all times relative to deadline D)

### D − 24h — refresh (automated)
    python scripts/ingest_bootstrap.py     # prices, injuries, flags
    python scripts/ingest_fixtures.py
    python scripts/run_predictions.py      # DC predictions for target GW
Failure here is non-fatal: 24h of slack to fix by hand.

### D − 10h — LOCK (automated; the one that matters)
    python scripts/lock_model_squad.py
    git add predictions/fpl/gwNN.json && git commit -m "GW NN lock"
    git push
Then the job verifies its own push (`git ls-remote origin main`) and alerts
on mismatch — email, Telegram, anything that wakes you.
Why D−10h and not later: prices change in the FPL overnight window; locking
~10h out accepts pennies of price drift in exchange for never missing the
public timestamp. The drift is a documented approximation, not a bug.

### Morning after deadline — human verification (2 minutes, non-negotiable)
    git ls-remote origin main              # push happened
    # open the commit on GitHub            # timestamp < deadline, hash file only
If the lock did NOT publish before the deadline → see failure playbook rule 1.

### GW end + ~12h — grade (semi-automated, no deadline pressure)
Wait for FPL to mark the GW `finished` (bonus points settle hours after the
last whistle — grade off the flag, never off full-time scores).
    python scripts/ingest_gameweek_results.py   # or the history backfill
    python scripts/grade_model_gw.py
    git add predictions/fpl/gwNN_result.json && git commit && git push
    git ls-remote origin main
Cards and receipts warm themselves on first request; nothing to run.

### Weekly content beat (any time before next deadline)
`/api/fpl/model/season` → the running record → one TMS clip. The script is
the numbers; the numbers are already public.

## Failure playbook

1. **Lock not pushed before the deadline.** The squad may still be in the
   ledger with a local timestamp — that is NOT the public record. The
   honesty rule, decided now so it never gets decided under pressure: the
   gameweek is marked **UNVERIFIED** in the public season table, with one
   plain sentence saying the commitment missed the deadline. It still gets
   graded and counted. One unverified week explained honestly costs almost
   nothing; one quietly backfilled timestamp, discovered, ends the project.
2. **Lock job fails at D−10h** (no predictions / DC params missing / DB
   error). The alert fires; there are 10 hours; run the D−24h steps by hand,
   then the lock. This is what the loud, specific error messages were for.
3. **FPL API down at grading time.** Wait. Grading has no deadline; the
   receipts backfill means users lose nothing.
4. **Double gameweeks (winter).** Before the first DGW: verify the results
   ingest SUMS a player's fixtures into the (player_id, gameweek_id) row
   rather than overwriting. This is the standing item from the grader
   review; it is cheap in October and a graded-wrong week in December.
5. **Predictor change mid-season.** Only on evidence, only in a commit with
   the evaluation attached, announced on the methodology page. The bake-off
   harness is the referee, same as pre-season.

## One-time setup checklist (before GW1)

- [ ] curl the two FPL endpoints, real 2025/26 team ID (field names + no auth)
- [ ] Railway: PUBLIC_BASE_URL, CARDS_DIR volume; confirm boot after 83da319
- [ ] season_rollover.py --label 2526 once 26/27 API is live
- [ ] train_dc.py refit against data/fpl_2526.db
- [ ] Schedule the D−24h and D−10h jobs; test the failure alert by breaking
      one on purpose
- [ ] Dry-run the full week once against the new DB: predict → lock →
      push → (fake finish) → grade on a test branch, then delete the branch
- [ ] Publish methodology page; link it from the /r/ share pages
