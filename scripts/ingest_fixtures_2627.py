"""
Ingest 2026/27 FPL fixtures, teams, and player prices from the FPL API.

Creates/updates:
  - upcoming_fixtures table  (2026/27 fixtures, used by optimizer)
  - Refreshes teams table with 2026/27 team list
  - Refreshes players table with 2026/27 prices and rosters
  - Inserts 2026/27 gameweeks into gameweeks table (offset by 1000 to avoid clash)
"""

import sys, json, sqlite3, urllib.request
sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "fpl.db"

BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
FIXTURES_URL  = "https://fantasy.premierleague.com/api/fixtures/"

POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
SEASON = "2627"
GW_OFFSET = 1000  # 2026/27 GWs stored as 1001-1038


def fetch(url: str) -> dict | list:
    req = urllib.request.Request(url, headers={"User-Agent": "fpl-optimiser/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def setup_upcoming_fixtures_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS upcoming_fixtures (
            fixture_id      INTEGER PRIMARY KEY,
            season          TEXT    NOT NULL DEFAULT '2627',
            gameweek_id     INTEGER NOT NULL,
            home_team_id    INTEGER NOT NULL,
            away_team_id    INTEGER NOT NULL,
            kickoff_time    TEXT,
            finished        INTEGER DEFAULT 0,
            home_score      INTEGER,
            away_score      INTEGER
        )
    """)
    conn.commit()


def add_season_column_to_gameweeks(conn: sqlite3.Connection) -> None:
    cols = [r[1] for r in conn.execute("PRAGMA table_info(gameweeks)").fetchall()]
    if "season" not in cols:
        conn.execute("ALTER TABLE gameweeks ADD COLUMN season TEXT DEFAULT '2526'")
        conn.execute("UPDATE gameweeks SET season = '2526' WHERE season IS NULL")
        conn.commit()


def main() -> None:
    print(f"Fetching FPL bootstrap data from API...")
    boot = fetch(BOOTSTRAP_URL)

    print(f"Fetching FPL fixtures from API...")
    fixtures_raw = fetch(FIXTURES_URL)

    conn = sqlite3.connect(DB_PATH)
    setup_upcoming_fixtures_table(conn)
    add_season_column_to_gameweeks(conn)

    # ── Teams ──────────────────────────────────────────────────────────────────
    api_teams = boot.get("teams", [])
    print(f"\nUpserting {len(api_teams)} teams...")
    for t in api_teams:
        conn.execute("""
            INSERT INTO teams (team_id, name, short_name)
            VALUES (?, ?, ?)
            ON CONFLICT(team_id) DO UPDATE SET
              name       = excluded.name,
              short_name = excluded.short_name
        """, (t["id"], t["name"], t["short_name"]))
    conn.commit()
    print(f"  Teams: {[t['short_name'] for t in api_teams]}")

    # ── Gameweeks ──────────────────────────────────────────────────────────────
    api_events = boot.get("events", [])
    print(f"\nUpserting {len(api_events)} 2026/27 gameweeks (IDs {GW_OFFSET+1}–{GW_OFFSET+len(api_events)})...")
    for ev in api_events:
        gw_id = ev["id"] + GW_OFFSET
        conn.execute("""
            INSERT INTO gameweeks (gameweek_id, deadline_time, is_current, is_next, finished, average_score, season)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(gameweek_id) DO UPDATE SET
              deadline_time = excluded.deadline_time,
              is_current    = excluded.is_current,
              is_next       = excluded.is_next,
              finished      = excluded.finished,
              average_score = excluded.average_score,
              season        = excluded.season
        """, (
            gw_id,
            ev.get("deadline_time"),
            int(ev.get("is_current", False)),
            int(ev.get("is_next", False)),
            int(ev.get("finished", False)),
            ev.get("average_entry_score", 0),
            SEASON,
        ))
    conn.commit()

    # ── Players ────────────────────────────────────────────────────────────────
    api_elements = boot.get("elements", [])
    print(f"\nUpserting {len(api_elements)} players...")
    for el in api_elements:
        pos_code = el.get("element_type", 3)
        conn.execute("""
            INSERT INTO players (
                player_id, first_name, second_name, web_name,
                team_id, position, current_cost, last_updated,
                corners_order, freekicks_order, penalties_order
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'), ?, ?, ?)
            ON CONFLICT(player_id) DO UPDATE SET
              first_name       = excluded.first_name,
              second_name      = excluded.second_name,
              web_name         = excluded.web_name,
              team_id          = excluded.team_id,
              position         = excluded.position,
              current_cost     = excluded.current_cost,
              last_updated     = excluded.last_updated,
              corners_order    = excluded.corners_order,
              freekicks_order  = excluded.freekicks_order,
              penalties_order  = excluded.penalties_order
        """, (
            el["id"],
            el.get("first_name", ""),
            el.get("second_name", ""),
            el.get("web_name", ""),
            el.get("team", 0),
            pos_code,
            el.get("now_cost", 0),
            el.get("corners_and_indirect_freekicks_order"),
            el.get("direct_freekicks_order"),
            el.get("penalties_order"),
        ))
    conn.commit()

    # Count by position
    for pos_code, pos_name in POSITION_MAP.items():
        cnt = sum(1 for el in api_elements if el.get("element_type") == pos_code)
        print(f"  {pos_name}: {cnt}")

    # ── Fixtures ───────────────────────────────────────────────────────────────
    print(f"\nIngesting {len(fixtures_raw)} fixtures into upcoming_fixtures...")
    conn.execute("DELETE FROM upcoming_fixtures WHERE season = '2627'")

    gw_counts: dict[int, int] = {}
    for fx in fixtures_raw:
        gw = fx.get("event")
        if not gw:
            continue   # no GW assigned yet (BGW/DGW edge cases)
        gw_id = gw + GW_OFFSET
        gw_counts[gw] = gw_counts.get(gw, 0) + 1
        conn.execute("""
            INSERT OR REPLACE INTO upcoming_fixtures
              (fixture_id, season, gameweek_id, home_team_id, away_team_id,
               kickoff_time, finished, home_score, away_score)
            VALUES (?, '2627', ?, ?, ?, ?, ?, ?, ?)
        """, (
            fx["id"],
            gw_id,
            fx["team_h"],
            fx["team_a"],
            fx.get("kickoff_time"),
            int(fx.get("finished", False)),
            fx.get("team_h_score"),
            fx.get("team_a_score"),
        ))
    conn.commit()

    print(f"  Stored {sum(gw_counts.values())} fixtures across {len(gw_counts)} gameweeks")

    # Show GW1 fixtures
    gw1_id = 1 + GW_OFFSET
    gw1 = conn.execute("""
        SELECT th.name, ta.name, uf.kickoff_time
        FROM   upcoming_fixtures uf
        JOIN   teams th ON uf.home_team_id = th.team_id
        JOIN   teams ta ON uf.away_team_id = ta.team_id
        WHERE  uf.gameweek_id = ?
        ORDER  BY uf.kickoff_time
    """, (gw1_id,)).fetchall()

    print(f"\nGW1 fixtures ({len(gw1)} matches):")
    for home, away, ko in gw1:
        ko_short = str(ko)[:16] if ko else "TBC"
        print(f"  {home:<25} vs  {away:<25}  [{ko_short}]")

    conn.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
