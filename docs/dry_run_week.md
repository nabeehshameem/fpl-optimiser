# Dry-run week — safely, without touching the real ledger

The lock and grade jobs write to an APPEND-ONLY ledger and publish a
commitment hash. Rehearsing against the real database would consume GW1:
the real lock would afterwards refuse ("already locked"), and the published
hash would commit the model to a squad built at rehearsal time.

Point everything at copies instead. All three variables must be set.

## 1. Make the scratch copies (PowerShell)

    cd C:\Users\GGPC\Projects\fpl-optimiser
    Copy-Item data\fpl.db data\fpl_dryrun.db
    New-Item -ItemType Directory -Force -Path C:\temp\fpl-dryrun | Out-Null
    git checkout -b dryrun ; git push -u origin dryrun

## 2. Set the overrides for THIS SHELL ONLY

    $env:FPL_DB_PATH   = "data\fpl_dryrun.db"
    $env:FPL_EXPORT_DIR= "C:\temp\fpl-dryrun"
    $env:GIT_BRANCH    = "dryrun"

Do NOT put these in the Task Scheduler actions or in user environment
variables — the real jobs must use the defaults.

## 3. Rehearse the week

    python scripts/weekly_ops.py refresh
    python scripts/weekly_ops.py lock

Check, in order:
  * C:\temp\fpl-dryrun\gw01.json exists and contains ONLY
    {gameweek, locked_at_utc, deadline_utc, squad_hash} — no player details
  * the commit landed on the dryrun branch: git ls-remote origin dryrun
  * healthchecks.io shows a ping for the lock check
  * data\fpl.db is UNCHANGED: the real ledger must have no GW1 row —
    python -c "import sqlite3;print(sqlite3.connect('data/fpl.db').execute('SELECT COUNT(*) FROM model_squad_log').fetchone())"
    must print (0,)

## 4. Rehearse grading

Grading refuses until FPL marks the gameweek finished, so force it on the
COPY only:

    python -c "import sqlite3;c=sqlite3.connect('data/fpl_dryrun.db');c.execute('UPDATE gameweeks SET finished=1 WHERE gameweek_id=1');c.commit()"
    python scripts/weekly_ops.py grade

Expect: auto-subs and captaincy applied, gw01_result.json written to the
scratch directory, hash in the result matching the commitment from step 3.
(Player results will be empty pre-season, so points will be zero — that is
fine. What is being tested is the pipeline, not the score.)

## 5. Rehearse it UNATTENDED — the part that matters

Schedule a one-off Task Scheduler task, with the three variables set in the
task's own Action, running `weekly_ops.py lock` five minutes from now, then
put the machine to sleep. This is the only test of the two failure modes
nothing else covers: whether the machine wakes, and whether git can
authenticate without a logged-in interactive session.

## 6. Tear down

    Remove-Item data\fpl_dryrun.db
    git checkout main ; git push origin --delete dryrun ; git branch -D dryrun

Then confirm the real ledger is still clean (step 3's last check) before
the deadline.
