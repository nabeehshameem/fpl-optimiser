# UCL Data Sources — Audit

*Written 2026-08-16. Re-audit before the 2026/27 UCL group phase (expected Oct 2026).*

## Decision summary

**Primary source: API-Football via RapidAPI** (we already have the key: `RAPIDAPI_KEY` in `.env`).
It covers everything we need for B1–B5: fixtures, results, standings, live scores, and
the per-match stats (xG, shots, possession) that the DC model needs for post-match
parameter updates.

**Secondary source: UEFA Fantasy Football API** (gaming.uefa.com) — for the UCL
Fantasy vertical (B4). It exposes player data, team mappings, and per-player
fantasy points in a structure closely analogous to the FPL bootstrap endpoint.

---

## Candidate sources

### 1. API-Football (RapidAPI — host `api-football-v1.p.rapidapi.com`)

**What it has**
- Fixtures and results for all 36 UCL league-phase matches + all knockout rounds
- Live scores and match events (goals, cards, substitutions) with sub-minute latency
- Per-match statistics: shots (on/off target), possession %, passes, xG where available
- Lineups and squad data — useful for cold-start priors on newly promoted clubs
- Historical UCL data back to 2015/16 for model training

**Rate limits / cost**
- Free tier: 100 requests/day — enough for polling but not for live ingest
- Pro tier (~$10/month): 7,500 requests/day — comfortable for full season with polling
  every 5 minutes during live matches
- The `RAPIDAPI_KEY` in `.env` is already activated on this host (SportAPI7 plan)

**Endpoints used**
- `GET /fixtures?league=2&season=2026` — all UCL fixtures for the season; use for
  the fixture grid and the DC prediction pipeline
- `GET /fixtures?id={fixture_id}` — single fixture with live stats; poll during
  matches for live score updates
- `GET /standings?league=2&season=2026` — league-phase table (36-team format)
- `GET /statistics?fixture={id}` — per-match stats after full time; feeds DC
  parameter retraining

**Key consideration — team short_name mapping (Rule 3)**
API-Football uses its own numeric team IDs. Before any DB write, map to `short_name`
via a teams lookup table built from `GET /teams?league=2&season=2026` at the start
of each season. Never join on API-Football team IDs across seasons.

**Verdict: PRIMARY. Use for all fixture/result/stats ingest (B1).**

---

### 2. UEFA Fantasy Football API (gaming.uefa.com)

**What it has**
- Bootstrap-style endpoint: all registered UCL players with team, position, price,
  and ownership — mirrors FPL's `bootstrap-static` very closely
- Per-gameweek player points and fixture difficulty ratings
- Live bonus point calculations during matches

**Rate limits / cost**
- Undocumented public API — no authentication required for read endpoints
- Same pattern as the FPL API: poll no more than once per minute to stay under
  unofficial rate limits; scrape headers for `Retry-After` if you hit 429s

**Key endpoints (unofficial, may change)**
- `https://gaming.uefa.com/en/uclfantasy/services/feeds/participants/participants_2_2026.json`
  — full player list with stats for the 2026 UCL season (season ID changes each year)
- `https://gaming.uefa.com/en/uclfantasy/services/feeds/fixtures/fixtures_2_2026.json`
  — all UCL fixtures with venue and kickoff times

**Key consideration**
These are unofficial endpoints scraped from the UEFA Fantasy UI. They have changed
without notice between seasons. Before B4, verify the URLs using the browser DevTools
Network tab on https://gaming.uefa.com/en/uclfantasy and capture the actual request
URLs for the 2026/27 season. Document the season ID in `ucl/docs/api_notes.md` once confirmed.

**Verdict: SECONDARY for B4 (UCL fantasy player data). Not needed for B1–B3.**

---

### 3. football-data.org

**What it has**
- UCL fixtures, results, and standings
- Free tier (tier 1): limited to selected competitions; UCL available on tier 2 ($5/month)
- No per-match statistics (shots, xG, etc.)

**Verdict: SKIP. API-Football covers the same data plus statistics we need for DC
retraining. Adding a second source for the same data creates a sync problem.**

---

### 4. OpenFootball (GitHub: openfootball/champions-league)

**What it has**
- Historical UCL results in a plain-text format, back to 1955
- No API — static files that update a few days after each match
- No statistics, no lineups, no live data

**Use case: historical parameter seeding only (B1 cold start).**
If we decide we want pre-2015 UCL data to deepen the DC training set, this is the
source. It is unlikely to change the 2026/27 season parameters materially — post-WC
form has more predictive power than UCL results from 2009.

**Verdict: ARCHIVE ONLY. Do not build an ingest pipeline for this.**

---

### 5. understat.com

**What it has**
- xG data for UCL matches (and major European leagues)
- No official API — data is embedded in JavaScript on match pages (requires scraping)
- Terms of service are ambiguous on programmatic access

**Verdict: SKIP for now.** API-Football's `statistics` endpoint includes xG for
most top-flight European fixtures. If API-Football xG coverage proves thin for
UCL, revisit understat as a supplementary source for model validation only.

---

## Data flow for B1–B4

```
API-Football                    UEFA Fantasy API
    │                                 │
    ├─ fixtures/results/stats ──┐     └─ player data ──┐
    │                           │                       │
    ▼                           ▼                       ▼
ucl/ingest_fixtures.py    ucl/ingest_results.py   ucl/ingest_fantasy_players.py
    │                           │
    └──────────┬────────────────┘
               │
           data/ucl.db  (separate DB from fpl.db — Rule 3: different team IDs)
               │
    ┌──────────┴───────────────┐
    │                          │
ucl/train_dc.py          ucl/run_predictions.py
    │                          │
    └──────────┬───────────────┘
               │
        predictions/ucl/gwNN_predictions.json   (same JSON-on-disk pattern as FPL)
               │
         src/ucl_api.py  →  GET /api/ucl/gw/{gw}
                             GET /api/ucl/standings
                             GET /api/ucl/bracket
```

## Implementation order for B1–B6

| Phase | Task | Depends on |
|-------|------|-----------|
| B1 | `ucl/ingest_fixtures.py` — fetch and store UCL fixtures from API-Football | `.env` RAPIDAPI_KEY ✓ |
| B1 | `ucl/ingest_results.py` — post-match result + stats ingest | B1 fixtures |
| B2 | `ucl/train_dc.py` — DC ratings from UCL history (reuse `src/dc_match.py`) | B1 results |
| B3 | `ucl/run_predictions.py` — scoreline matrix for upcoming fixtures | B2 DC params |
| B3 | `src/ucl_api.py` — API endpoints; `GET /api/ucl/gw/{gw}`, standings, bracket | B3 predictions |
| B4 | `ucl/ingest_fantasy_players.py` — UEFA Fantasy player data | UEFA API confirmed |
| B5 | League-phase table simulation — 36-team format with 36 fixtures | B3 DC params |
| B6 | Knockout bracket — 8-team PO round + R16 through Final | B5 |

## Action items before B1 starts

1. **Verify RAPIDAPI_KEY plan covers `/statistics` endpoint** — the free tier excludes
   match statistics; confirm the SportAPI7 plan includes it before building the
   retraining pipeline.
2. **Capture UEFA Fantasy season ID** for 2026/27 — open gaming.uefa.com in a browser,
   check the Network tab, note the season ID in the fixture/participant URLs.
3. **Decide DB strategy** — recommend `data/ucl.db` as a separate file from `fpl.db`.
   UCL team IDs from API-Football will conflict with FPL team IDs (same short_names,
   different numeric IDs), and keeping them separate makes Rule 3 enforcement easier.
