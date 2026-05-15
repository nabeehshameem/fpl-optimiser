"""
mini_league.py
Mini-league tracker: standings, GW rank history, chip usage.

Uses the public FPL classic leagues API (no authentication required).
Data is cached in the local SQLite DB; run ingest_mini_league.py to refresh.
"""

import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "fpl.db"

FPL_BASE = "https://fantasy.premierleague.com/api"
REQUEST_DELAY = 0.5  # seconds between API calls

CHIP_SHORT = {
    "wildcard":  "WC",
    "freehit":   "FH",
    "bboost":    "BB",
    "3xc":       "TC",
}

# ---------- Schema ----------

_SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS mini_leagues (
        league_id    INTEGER PRIMARY KEY,
        name         TEXT,
        last_updated TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS league_standings (
        league_id    INTEGER NOT NULL,
        entry_id     INTEGER NOT NULL,
        entry_name   TEXT,
        player_name  TEXT,
        rank         INTEGER,
        last_rank    INTEGER,
        total_points INTEGER,
        event_total  INTEGER,
        fetched_gw   INTEGER,
        PRIMARY KEY (league_id, entry_id),
        FOREIGN KEY (league_id) REFERENCES mini_leagues(league_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS league_entry_history (
        entry_id               INTEGER NOT NULL,
        gameweek_id            INTEGER NOT NULL,
        points                 INTEGER,
        total_points           INTEGER,
        overall_rank           INTEGER,
        bank                   INTEGER,
        value                  INTEGER,
        event_transfers        INTEGER,
        event_transfers_cost   INTEGER,
        points_on_bench        INTEGER,
        PRIMARY KEY (entry_id, gameweek_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS league_entry_chips (
        entry_id     INTEGER NOT NULL,
        chip_name    TEXT NOT NULL,
        gameweek_id  INTEGER,
        PRIMARY KEY (entry_id, chip_name)
    )
    """,
]


def _ensure_schema() -> None:
    conn = sqlite3.connect(DB_PATH)
    for stmt in _SCHEMA:
        conn.execute(stmt)
    conn.commit()
    conn.close()


# ---------- API helpers ----------

def _get(url: str) -> dict:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_league_standings(league_id: int) -> dict:
    """Fetch all standings pages for a classic league."""
    url = f"{FPL_BASE}/leagues-classic/{league_id}/standings/"
    data = _get(url)
    results = list(data["standings"]["results"])
    page = 1
    while data["standings"].get("has_next"):
        page += 1
        time.sleep(REQUEST_DELAY)
        data = _get(f"{url}?page_standings={page}")
        results.extend(data["standings"]["results"])
    return {"league": data["league"], "results": results}


def fetch_entry_history(entry_id: int) -> dict:
    return _get(f"{FPL_BASE}/entry/{entry_id}/history/")


# ---------- Ingest ----------

def ingest_league(league_id: int) -> None:
    """Fetch standings + all entry histories and cache in DB."""
    _ensure_schema()

    print(f"  Fetching standings for league {league_id}...")
    league_data = fetch_league_standings(league_id)
    league_name = league_data["league"]["name"]
    entries = league_data["results"]
    print(f"  League: {league_name!r}  ({len(entries)} entries)")

    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT gameweek_id FROM gameweeks WHERE is_current = 1 LIMIT 1"
    ).fetchone()
    if not row:
        row = conn.execute(
            "SELECT gameweek_id FROM gameweeks WHERE is_next = 1 LIMIT 1"
        ).fetchone()
    current_gw = row[0] if row else None
    conn.close()

    now_iso = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT OR REPLACE INTO mini_leagues (league_id, name, last_updated) VALUES (?, ?, ?)",
        (league_id, league_name, now_iso),
    )
    for e in entries:
        conn.execute(
            """
            INSERT OR REPLACE INTO league_standings
              (league_id, entry_id, entry_name, player_name, rank, last_rank,
               total_points, event_total, fetched_gw)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (league_id, e["entry"], e["entry_name"], e["player_name"],
             e["rank"], e["last_rank"], e["total"], e["event_total"], current_gw),
        )
    conn.commit()
    conn.close()

    print(f"  Fetching history for {len(entries)} entries...")
    for i, e in enumerate(entries):
        entry_id = e["entry"]
        time.sleep(REQUEST_DELAY)
        hist = fetch_entry_history(entry_id)

        conn = sqlite3.connect(DB_PATH)
        for gw_row in hist.get("current", []):
            conn.execute(
                """
                INSERT OR REPLACE INTO league_entry_history
                  (entry_id, gameweek_id, points, total_points, overall_rank,
                   bank, value, event_transfers, event_transfers_cost, points_on_bench)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (entry_id, gw_row["event"], gw_row["points"], gw_row["total_points"],
                 gw_row.get("overall_rank"), gw_row.get("bank"), gw_row.get("value"),
                 gw_row.get("event_transfers"), gw_row.get("event_transfers_cost"),
                 gw_row.get("points_on_bench")),
            )
        conn.execute("DELETE FROM league_entry_chips WHERE entry_id = ?", (entry_id,))
        for chip in hist.get("chips", []):
            conn.execute(
                "INSERT OR IGNORE INTO league_entry_chips (entry_id, chip_name, gameweek_id) VALUES (?, ?, ?)",
                (entry_id, chip["name"], chip["event"]),
            )
        conn.commit()
        conn.close()

        if (i + 1) % 10 == 0 or (i + 1) == len(entries):
            print(f"    {i+1}/{len(entries)} done")

    print(f"  Ingest complete.")


# ---------- Analysis ----------

def _current_gw_from_db() -> Optional[int]:
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("SELECT gameweek_id FROM gameweeks WHERE is_current = 1 LIMIT 1").fetchone()
    if not row:
        row = conn.execute("SELECT gameweek_id FROM gameweeks WHERE is_next = 1 LIMIT 1").fetchone()
    conn.close()
    return row[0] if row else None


def league_summary(league_id: int, current_gw: Optional[int] = None) -> pd.DataFrame:
    """
    Standings table with rank move, GW points, transfers made this GW,
    chip played this GW, and all chips used so far.
    """
    conn = sqlite3.connect(DB_PATH)
    standings = pd.read_sql_query(
        "SELECT * FROM league_standings WHERE league_id = ? ORDER BY rank",
        conn, params=(league_id,),
    )
    if standings.empty:
        conn.close()
        raise ValueError(
            f"No standings found for league {league_id}. "
            "Run: python scripts/ingest_mini_league.py --league {league_id}"
        )

    if current_gw is None:
        current_gw = current_gw or standings["fetched_gw"].max()

    entry_ids = standings["entry_id"].tolist()
    ph = ",".join("?" * len(entry_ids))

    gw_hist = pd.read_sql_query(
        f"""
        SELECT entry_id, event_transfers, event_transfers_cost, points_on_bench
        FROM league_entry_history
        WHERE entry_id IN ({ph}) AND gameweek_id = ?
        """,
        conn, params=entry_ids + [current_gw],
    )
    chips = pd.read_sql_query(
        f"SELECT entry_id, chip_name, gameweek_id FROM league_entry_chips WHERE entry_id IN ({ph})",
        conn, params=entry_ids,
    )
    conn.close()

    chips["chip_short"] = chips["chip_name"].map(CHIP_SHORT).fillna(chips["chip_name"].str.upper())
    chips_this_gw = (
        chips[chips["gameweek_id"] == current_gw]
        .groupby("entry_id")["chip_short"]
        .apply("+".join)
        .reset_index()
        .rename(columns={"chip_short": "chip_gw"})
    )
    chips_all = (
        chips.sort_values("gameweek_id")
        .groupby("entry_id")
        .apply(lambda g: ", ".join(f"{r['chip_short']} GW{r['gameweek_id']}" for _, r in g.iterrows()))
        .reset_index()
        .rename(columns={0: "chips_used"})
    )

    df = (
        standings
        .merge(gw_hist, on="entry_id", how="left")
        .merge(chips_this_gw, on="entry_id", how="left")
        .merge(chips_all, on="entry_id", how="left")
    )

    df["move"] = (df["last_rank"] - df["rank"]).apply(
        lambda x: f"+{int(x)}" if x > 0 else (str(int(x)) if x < 0 else "=")
    )

    result = df[[
        "rank", "entry_name", "player_name", "total_points", "event_total",
        "move", "event_transfers", "event_transfers_cost",
        "points_on_bench", "chip_gw", "chips_used",
    ]].copy()
    result.columns = [
        "#", "Team", "Manager", "Total", "GW",
        "Move", "Xfers", "Hit", "Bench", "Chip", "All chips",
    ]
    result["Chip"] = result["Chip"].fillna("")
    result["All chips"] = result["All chips"].fillna("none")
    result["Xfers"] = result["Xfers"].fillna(0).astype(int)
    result["Hit"] = result["Hit"].fillna(0).astype(int)
    result["Bench"] = result["Bench"].fillna(0).astype(int)
    return result.reset_index(drop=True)


def rank_history_table(league_id: int, recent_gws: int = 6) -> pd.DataFrame:
    """
    Within-league rank per GW for each entry (computed from total_points).
    Rows = teams, columns = GW rank.
    """
    conn = sqlite3.connect(DB_PATH)
    standings = pd.read_sql_query(
        "SELECT entry_id, entry_name FROM league_standings WHERE league_id = ? ORDER BY rank",
        conn, params=(league_id,),
    )
    if standings.empty:
        conn.close()
        raise ValueError(f"No standings for league {league_id}.")

    entry_ids = standings["entry_id"].tolist()
    ph = ",".join("?" * len(entry_ids))
    hist = pd.read_sql_query(
        f"SELECT entry_id, gameweek_id, total_points FROM league_entry_history WHERE entry_id IN ({ph})",
        conn, params=entry_ids,
    )
    conn.close()

    gws = sorted(hist["gameweek_id"].unique())[-recent_gws:]
    hist = hist[hist["gameweek_id"].isin(gws)].copy()
    hist["league_rank"] = hist.groupby("gameweek_id")["total_points"].rank(
        method="min", ascending=False
    ).astype(int)

    pivot = hist.pivot(index="entry_id", columns="gameweek_id", values="league_rank")
    pivot.columns = [f"GW{g}" for g in pivot.columns]
    result = standings.merge(pivot, on="entry_id", how="left").drop(columns=["entry_id"])
    result = result.rename(columns={"entry_name": "Team"})
    return result.reset_index(drop=True)


def chips_remaining(league_id: int) -> pd.DataFrame:
    """Who still has which chips left (has not used them yet)."""
    ALL_CHIPS = [("wildcard", "WC"), ("freehit", "FH"), ("bboost", "BB"), ("3xc", "TC")]

    conn = sqlite3.connect(DB_PATH)
    standings = pd.read_sql_query(
        "SELECT entry_id, entry_name, player_name FROM league_standings WHERE league_id = ? ORDER BY rank",
        conn, params=(league_id,),
    )
    entry_ids = standings["entry_id"].tolist()
    ph = ",".join("?" * len(entry_ids))
    used = pd.read_sql_query(
        f"SELECT entry_id, chip_name FROM league_entry_chips WHERE entry_id IN ({ph})",
        conn, params=entry_ids,
    )
    conn.close()

    used_by_entry = used.groupby("entry_id")["chip_name"].apply(set).to_dict()
    rows = []
    for _, row in standings.iterrows():
        used_chips = used_by_entry.get(row["entry_id"], set())
        remaining = [short for name, short in ALL_CHIPS if name not in used_chips]
        rows.append({
            "Team": row["entry_name"],
            "Manager": row["player_name"],
            "Remaining": ", ".join(remaining) if remaining else "none",
        })
    return pd.DataFrame(rows)
