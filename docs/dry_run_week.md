# Dry-run week — safely, without touching the real ledger

The lock and grade jobs write to an APPEND-ONLY ledger and publish a
commitment hash. Rehearsing against the real database would consume GW1:
the real lock would afterwards refuse ("already locked"), and the published
hash would commit the model to a squad built at rehearsal time.

Point everything at copies instead. All three variables must be set.

## 1. Make the scratch copies (PowerShell)

    cd C:\Users\GGPC\Projects\fpl-optimiser
    Copy-Item data\fpl.db data\fpl_dryrun.db
    New-Item -ItemType Directory -Force -Path predictions\fpl-dryrun | Out-Null
    git checkout -b dryrun ; git push -u origin dryrun

## 2. Set the overrides for THIS SHELL ONLY

    $env:FPL_DB_PATH   = "data\fpl_dryrun.db"
    $env:FPL_EXPORT_DIR= "predictions\fpl-dryrun"
    $env:GIT_BRANCH    = "dryrun"

FPL_EXPORT_DIR must be a subdirectory of the project root so git can stage
the output files. C:\temp or any external path will be rejected.

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
    python -c "
    import sqlite3; c=sqlite3.connect('data/fpl.db')
    tables = {r[0] for r in c.execute(\"SELECT name FROM sqlite_master WHERE type='table'\")}
    if 'model_squad_log' not in tables:
        print('CLEAN: model_squad_log does not exist (no lock ever run on real DB)')
    else:
        print(c.execute('SELECT COUNT(*) FROM model_squad_log').fetchone())
    "
    must print CLEAN or (0,)

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

### Exact steps

**a. Open Task Scheduler — not schtasks.** Search "Task Scheduler" in Start.
Click "Create Task" (not "Create Basic Task") from the right pane.

**b. General tab:**
- Name: `FPL_DryRun_Lock`
- Security options: check "Run whether user is logged on or not"
- Check "Run with highest privileges"

**c. Triggers tab → New:**
- Begin the task: On a schedule
- One time
- Set time to NOW + 5 minutes (check the clock and add 5)
- Enabled: checked

**d. Actions tab → New:**
- Action: Start a program
- Program/script: `powershell.exe`
- Add arguments (paste this as one line):

```
-NonInteractive -Command "$env:FPL_DB_PATH='data\fpl_dryrun.db'; $env:FPL_EXPORT_DIR='predictions\fpl-dryrun'; $env:GIT_BRANCH='dryrun'; Set-Location 'C:\Users\GGPC\Projects\fpl-optimiser'; python scripts\weekly_ops.py lock >> logs\dryrun_lock.txt 2>&1"
```

- Start in: `C:\Users\GGPC\Projects\fpl-optimiser`

**e. Conditions tab:**
- Uncheck "Start the task only if the computer is on AC power" (so it runs on battery after wake)
- Check "Wake the computer to run this task"

**f. Settings tab:**
- Check "Run task as soon as possible after a scheduled start is missed"

**g. Click OK.** Enter your Windows password when prompted.

**h. Put the machine to sleep immediately:**
```powershell
rundll32.exe powrprof.dll,SetSuspendState 0,1,0
```

**i. After wake, check:**
```powershell
Get-Content logs\dryrun_lock.txt
```
Expect: lock output including "Exported predictions\fpl-dryrun\gw01.json"

```powershell
Get-Content predictions\fpl-dryrun\gw01.json
```
Must contain ONLY `{gameweek, locked_at_utc, deadline_utc, squad_hash}` — no player names.

```powershell
git ls-remote origin dryrun
git log --oneline -3 origin/dryrun
```
Commit must exist and be NEWER than the sleep time, proving git push happened outside an interactive session.

**What success looks like:**
- `gw01.json` contains only the hash (no squad details)
- The commit timestamp is after you went to sleep
- `data\fpl.db` still shows 0 rows in `model_squad_log` (real DB untouched)
- Task Scheduler shows "Last Run Result: 0x0" (success)

**j. Delete the task after a successful run:**

Task Scheduler → right-click `FPL_DryRun_Lock` → Delete. Don't leave a
stale scheduled lock task pointing at a dryrun DB.

## 6. Tear down

    Remove-Item data\fpl_dryrun.db
    git checkout main ; git push origin --delete dryrun ; git branch -D dryrun

Then confirm the real ledger is still clean (step 3's last check) before
the deadline.
