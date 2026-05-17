"""
morning_brief.py
Daily FPL content brief for analytics creators.

Each morning, pulls:
  - FBRef Premier League standard stats (xG, xAG, goals, assists)
  - FPL DB for ownership, price, and transfer trends

Then outputs:
  1. xG leaders            — most dangerous players this season
  2. xG overperformers     — goals >> xG (luck-driven; potential sell/avoid signal)
  3. xG underperformers    — xG >> goals (unlucky; potential buy signal)
  4. xA leaders            — creative hub rankings
  5. Price movers          — buying / selling pressure from ownership delta
  6. 3-5 video idea cards  — title, angle, hook, and supporting data

Output:
  - Printed to console
  - Saved to data/briefs/YYYY-MM-DD.json
  - Optional email: set env vars BRIEF_EMAIL_TO, BRIEF_SMTP_HOST,
    BRIEF_SMTP_USER, BRIEF_SMTP_PASS, then pass --email

Dependencies (pip install if missing):
  pip install lxml          # strongly recommended for FBRef HTML parsing
  pip install requests pandas

Usage:
  python scripts/morning_brief.py
  python scripts/morning_brief.py --no-fbref        # skip FBRef (faster, no xG)
  python scripts/morning_brief.py --email           # send brief by email
  python scripts/morning_brief.py --top 10          # show top-N per section
  python scripts/morning_brief.py --min-matches 10  # stricter appearance filter

Automation (run daily via Windows Task Scheduler):
  Action: python "C:\\path\\to\\fpl-optimiser\\scripts\\morning_brief.py"
  Trigger: Daily, 06:00 AM
"""

import argparse
import json
import os
import smtplib
import sqlite3
import sys
import time
from datetime import date, datetime
from email.mime.text import MIMEText
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DB_PATH    = PROJECT_ROOT / "data" / "fpl.db"
BRIEFS_DIR = PROJECT_ROOT / "data" / "briefs"

# Understat embeds player stats as JSON in the league page — no bot protection.
# Season key = year the season starts (2024 = 2024/25).
UNDERSTAT_URL = "https://understat.com/league/EPL/{season}"
UNDERSTAT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept-Language": "en-GB,en;q=0.9",
}
POSITION_NAMES = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


# ── Understat scraper ─────────────────────────────────────────────────────────

def _current_season() -> int:
    """Return the Understat season key (year the season started)."""
    today = date.today()
    return today.year if today.month >= 8 else today.year - 1


UNDERSTAT_API_URL = "https://understat.com/main/getPlayersStats/"


def _fetch_understat_xg(min_matches: int = 5, season: int | None = None) -> pd.DataFrame:
    """
    Fetch Premier League player xG / xA stats from Understat's AJAX endpoint.

    Uses a session to first visit the league page (cookie handshake), then
    POSTs to the data endpoint. No API key needed.
    Returns one row per player with columns:
      name, team, position, matches, minutes, goals, assists, xg, xa, npxg,
      xg_diff, xa_rank
    """
    if season is None:
        season = _current_season()

    session = requests.Session()
    league_url = UNDERSTAT_URL.format(season=season)
    session.get(league_url, headers=UNDERSTAT_HEADERS, timeout=30)

    resp = session.post(
        UNDERSTAT_API_URL,
        data={"league": "EPL", "season": str(season)},
        headers={
            **UNDERSTAT_HEADERS,
            "X-Requested-With": "XMLHttpRequest",
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            "Referer": league_url,
        },
        timeout=30,
    )
    resp.raise_for_status()

    payload = resp.json()
    if not payload.get("success") or "players" not in payload:
        raise ValueError(f"Unexpected Understat response: {resp.text[:200]}")

    rows = []
    for p in payload["players"]:
        try:
            rows.append({
                "name":     p["player_name"],
                "team":     p["team_title"],
                "position": p.get("position", ""),
                "matches":  int(p.get("games", 0)),
                "minutes":  int(p.get("time", 0)),
                "goals":    float(p.get("goals", 0)),
                "assists":  float(p.get("assists", 0)),
                "xg":       float(p.get("xG", 0)),
                "xa":       float(p.get("xA", 0)),
                "npxg":     float(p.get("npxG", 0)),
            })
        except (KeyError, ValueError):
            continue

    df = pd.DataFrame(rows)
    df = df[df["matches"] >= min_matches].reset_index(drop=True)
    df["xg_diff"] = df["goals"] - df["xg"]
    df["xa_rank"] = df["xa"].rank(ascending=False).astype(int)
    return df[["name", "team", "position", "matches", "minutes",
               "goals", "assists", "xg", "xa", "npxg", "xg_diff", "xa_rank"]].copy()


# ── FPL DB helpers ─────────────────────────────────────────────────────────────

def _fpl_snapshot() -> pd.DataFrame:
    """Current player info + latest ownership snapshot from DB."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        """
        SELECT p.player_id, p.web_name, p.position, p.current_cost,
               t.short_name AS team,
               s.selected_by_percent
        FROM players p
        JOIN teams t ON p.team_id = t.team_id
        LEFT JOIN player_snapshots s
            ON s.player_id = p.player_id
            AND s.snapshot_id = (
                SELECT MAX(snapshot_id)
                FROM player_snapshots
                WHERE player_id = p.player_id
            )
        """,
        conn,
    )
    conn.close()

    df["selected_by_percent"] = (
        pd.to_numeric(df["selected_by_percent"], errors="coerce").fillna(0.0)
    )
    df["position"] = df["position"].map(POSITION_NAMES)
    df["price"] = (df["current_cost"] / 10).round(1)
    return df


def _price_movers(threshold: float = 0.3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (rising, falling) DataFrames based on ownership delta between
    the two most recent snapshots.
    """
    conn = sqlite3.connect(DB_PATH)
    snaps = pd.read_sql_query(
        """
        SELECT player_id, selected_by_percent,
               ROW_NUMBER() OVER (
                   PARTITION BY player_id ORDER BY snapshot_id DESC
               ) AS rn
        FROM player_snapshots
        """,
        conn,
    )
    players = pd.read_sql_query(
        """
        SELECT p.player_id, p.web_name, p.position, p.current_cost,
               t.short_name AS team
        FROM players p JOIN teams t ON p.team_id = t.team_id
        """,
        conn,
    )
    conn.close()

    snaps["selected_by_percent"] = (
        pd.to_numeric(snaps["selected_by_percent"], errors="coerce").fillna(0.0)
    )
    latest = snaps[snaps["rn"] == 1][["player_id", "selected_by_percent"]]
    prev = (
        snaps[snaps["rn"] == 2][["player_id", "selected_by_percent"]]
        .rename(columns={"selected_by_percent": "prev_own"})
    )

    df = latest.merge(prev, on="player_id", how="left")
    df["prev_own"] = df["prev_own"].fillna(df["selected_by_percent"])
    df["delta"] = df["selected_by_percent"] - df["prev_own"]
    df = df.merge(players, on="player_id", how="left")
    df["position"] = df["position"].map(POSITION_NAMES)
    df["price"] = (df["current_cost"] / 10).round(1)
    df["own%"] = df["selected_by_percent"].round(1)
    df["d_own"] = df["delta"].round(2)

    keep = ["web_name", "team", "position", "price", "own%", "d_own"]
    rising  = df[df["delta"] >=  threshold].nlargest(5,  "delta")[keep].copy()
    falling = df[df["delta"] <= -threshold].nsmallest(5, "delta")[keep].copy()
    return rising, falling


# ── Video idea generator ───────────────────────────────────────────────────────

def _generate_ideas(
    xg_leaders: pd.DataFrame,
    overperformers: pd.DataFrame,
    underperformers: pd.DataFrame,
    xa_leaders: pd.DataFrame,
    price_risers: pd.DataFrame,
    fpl_df: pd.DataFrame,
) -> list[dict]:
    ideas: list[dict] = []

    # Idea 1 — xG underperformer (regression to mean; buy signal)
    if not underperformers.empty:
        p = underperformers.iloc[0]
        deficit = abs(float(p["xg_diff"]))
        ideas.append({
            "title": (
                f"The Most UNLUCKY Player in the PL Right Now "
                f"— {p['name']} xG Deep Dive"
            ),
            "angle": "xG underperformer — goals WILL come; buy before everyone else does",
            "hook": (
                f"The data says {p['name']} should have scored {deficit:.1f} more goals "
                f"than he has. This can't continue. Here's why he's about to explode."
            ),
            "data": (
                f"{p['name']} ({p.get('team','?')}) | "
                f"xG: {p['xg']:.1f} | Goals: {p['goals']:.0f} | "
                f"Deficit: {deficit:.1f} goals"
            ),
        })

    # Idea 2 — xG overperformer (sell / avoid)
    if not overperformers.empty:
        p = overperformers.iloc[0]
        surplus = float(p["xg_diff"])
        ideas.append({
            "title": (
                f"Is {p['name']} Due a GOAL DROUGHT? "
                f"The xG Data Says Yes"
            ),
            "angle": "Finishing luck can't continue — sell signal before the rest catch on",
            "hook": (
                f"{p['name']} has scored {surplus:.1f} more goals than xG predicts. "
                f"History says this corrects itself. Is a drought coming?"
            ),
            "data": (
                f"{p['name']} ({p.get('team','?')}) | "
                f"Goals: {p['goals']:.0f} | xG: {p['xg']:.1f} | "
                f"Surplus: {surplus:.1f} goals"
            ),
        })

    # Idea 3 — xG leader (dangerous player, own if price is right)
    if not xg_leaders.empty:
        p = xg_leaders.iloc[0]
        ideas.append({
            "title": (
                f"The MOST DANGEROUS Player in the Premier League "
                f"— Why {p['name']} is a MUST-OWN"
            ),
            "angle": "League xG leader — consistent goal threat regardless of current returns",
            "hook": (
                f"No one in the Premier League is generating better chances than "
                f"{p['name']}. {p['xg']:.1f} xG this season. The goals are coming."
            ),
            "data": (
                f"{p['name']} ({p.get('team','?')}) | "
                f"xG: {p['xg']:.1f} | xA: {p.get('xa', 0):.1f} | "
                f"Goals: {p['goals']:.0f}"
            ),
        })

    # Idea 4 — xA leader (assist machine, creative hub)
    if not xa_leaders.empty:
        p = xa_leaders.iloc[0]
        ideas.append({
            "title": (
                f"The BEST Assist Provider You're Probably Not Owning "
                f"— {p['name']} xA Analysis"
            ),
            "angle": "Creative hub ranked #1 in PL by xA — clean sheets and assists stacking up",
            "hook": (
                f"{p['name']} ranks #1 in the PL for expected assists with {p['xa']:.1f} xA. "
                f"Here's why his underlying numbers make him elite FPL value."
            ),
            "data": (
                f"{p['name']} ({p.get('team','?')}) | "
                f"xA: {p['xa']:.1f} | Assists: {p['assists']:.0f} | "
                f"xG: {p['xg']:.1f}"
            ),
        })

    # Idea 5 — price risers (FOMO / deadline urgency)
    if not price_risers.empty:
        names = " & ".join(price_risers["web_name"].head(2).tolist())
        deltas = price_risers.head(3)[["web_name", "d_own"]].to_string(index=False)
        ideas.append({
            "title": f"BUY NOW Before the Price Rise: {names} Are Flying In",
            "angle": "Transfer rush -> price rise imminent; buy before it costs you",
            "hook": (
                f"Thousands of managers are buying these players right now. "
                f"Miss the window and you'll pay an extra 0.1m. Here's who to target."
            ),
            "data": f"Ownership changes:\n{deltas}",
        })

    return ideas


# ── Output helpers ─────────────────────────────────────────────────────────────

def _header(title: str) -> None:
    bar = "=" * 60
    print(f"\n{bar}\n  {title}\n{bar}")


def _print_table(df: pd.DataFrame, cols: list[str], top_n: int) -> None:
    available = [c for c in cols if c in df.columns]
    print(df[available].head(top_n).to_string(index=False))


def _save_brief(brief: dict) -> Path:
    BRIEFS_DIR.mkdir(parents=True, exist_ok=True)
    path = BRIEFS_DIR / f"{date.today()}.json"
    path.write_text(json.dumps(brief, indent=2, default=str), encoding="utf-8")
    return path


def _send_email(brief: dict) -> None:
    to_addr  = os.environ.get("BRIEF_EMAIL_TO", "")
    host     = os.environ.get("BRIEF_SMTP_HOST", "smtp.gmail.com")
    port     = int(os.environ.get("BRIEF_SMTP_PORT", "587"))
    user     = os.environ.get("BRIEF_SMTP_USER", "")
    password = os.environ.get("BRIEF_SMTP_PASS", "")

    if not all([to_addr, user, password]):
        print(
            "  [email] Set BRIEF_EMAIL_TO, BRIEF_SMTP_USER, BRIEF_SMTP_PASS "
            "env vars to enable email delivery."
        )
        return

    lines = [f"FPL Morning Brief — {date.today()}\n"]
    for i, idea in enumerate(brief.get("video_ideas", []), 1):
        lines += [
            f"{i}. {idea['title']}",
            f"   Angle: {idea['angle']}",
            f"   Hook:  {idea['hook']}",
            f"   Data:  {idea['data']}",
            "",
        ]
    body = "\n".join(lines)

    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = f"FPL Brief {date.today()}"
    msg["From"]    = user
    msg["To"]      = to_addr

    with smtplib.SMTP(host, port) as smtp:
        smtp.ehlo()
        smtp.starttls()
        smtp.login(user, password)
        smtp.send_message(msg)
    print(f"  Email sent to {to_addr}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="FPL daily content brief")
    parser.add_argument("--no-fbref", action="store_true",
                        help="Skip Understat xG scrape (faster; no xG sections)")
    parser.add_argument("--email", action="store_true",
                        help="Send brief by email (requires BRIEF_* env vars)")
    parser.add_argument("--top", type=int, default=8,
                        help="Top-N rows per section (default 8)")
    parser.add_argument("--min-matches", type=int, default=5,
                        help="Min PL appearances for FBRef inclusion (default 5)")
    args = parser.parse_args()

    print(f"\nFPL Morning Brief — {date.today()}")
    brief: dict = {
        "generated": datetime.now().isoformat(),
        "sections": {},
        "video_ideas": [],
    }

    # ── FBRef ────────────────────────────────────────────────────────────
    fbref_df = pd.DataFrame()
    if not args.no_fbref:
        season = _current_season()
        print(f"\nFetching Understat xG stats ({season}/{str(season+1)[-2:]} season)...")
        try:
            fbref_df = _fetch_understat_xg(min_matches=args.min_matches, season=season)
            print(f"  {len(fbref_df)} players loaded.")
        except Exception as exc:
            print(f"  [warn] Understat fetch failed: {exc}")
            print("  Continuing without xG data.")

    xg_leaders = xa_leaders = overperformers = underperformers = pd.DataFrame()

    if not fbref_df.empty:
        xg_leaders      = fbref_df.nlargest(args.top, "xg").reset_index(drop=True)
        xa_leaders      = fbref_df.nlargest(args.top, "xa").reset_index(drop=True)
        overperformers  = fbref_df.nlargest(args.top, "xg_diff").reset_index(drop=True)
        underperformers = fbref_df.nsmallest(args.top, "xg_diff").reset_index(drop=True)

        _header(f"xG LEADERS  (top {args.top} most dangerous players)")
        _print_table(xg_leaders, ["name", "team", "position", "matches", "goals", "xg", "xa"], args.top)

        _header(f"xG OVERPERFORMERS  (goals >> xG — potential sell / avoid signal, top {args.top})")
        print("  Positive xg_diff = scored more than expected. Luck-driven; may regress.\n")
        _print_table(overperformers, ["name", "team", "goals", "xg", "xg_diff"], args.top)

        _header(f"xG UNDERPERFORMERS  (xG >> goals — unlucky buyers, top {args.top})")
        print("  Negative xg_diff = deserves more goals than scored. Regression likely.\n")
        _print_table(underperformers, ["name", "team", "goals", "xg", "xg_diff"], args.top)

        _header(f"xA LEADERS  (most creative players, top {args.top})")
        _print_table(xa_leaders, ["name", "team", "position", "matches", "assists", "xa"], args.top)

        brief["sections"]["xg_leaders"]      = xg_leaders.head(10).to_dict("records")
        brief["sections"]["xa_leaders"]      = xa_leaders.head(10).to_dict("records")
        brief["sections"]["overperformers"]  = overperformers.head(10).to_dict("records")
        brief["sections"]["underperformers"] = underperformers.head(10).to_dict("records")

    # ── FPL price movers ─────────────────────────────────────────────────
    fpl_df = pd.DataFrame()
    price_risers = price_fallers = pd.DataFrame()

    try:
        fpl_df = _fpl_snapshot()
        price_risers, price_fallers = _price_movers()

        if not price_risers.empty:
            _header("PRICE RISERS  (buying pressure — act before the rise)")
            _print_table(price_risers, ["web_name", "team", "position", "price", "own%", "d_own"], args.top)
            brief["sections"]["price_risers"] = price_risers.to_dict("records")

        if not price_fallers.empty:
            _header("PRICE FALLERS  (selling pressure — may drop soon)")
            _print_table(price_fallers, ["web_name", "team", "position", "price", "own%", "d_own"], args.top)
            brief["sections"]["price_fallers"] = price_fallers.to_dict("records")

        if price_risers.empty and price_fallers.empty:
            print("\n  [info] Price movers: need at least 2 snapshot runs. "
                  "Run ingest_bootstrap.py again tomorrow.")

    except Exception as exc:
        print(f"\n  [warn] FPL DB error: {exc}")

    # ── Video ideas ──────────────────────────────────────────────────────
    ideas = _generate_ideas(
        xg_leaders, overperformers, underperformers, xa_leaders, price_risers, fpl_df
    )
    _header(f"VIDEO IDEAS  ({len(ideas)} suggestions)")
    for i, idea in enumerate(ideas, 1):
        print(f"\n  {i}. {idea['title']}")
        print(f"     Angle: {idea['angle']}")
        print(f"     Hook:  {idea['hook']}")
        print(f"     Data:  {idea['data']}")

    brief["video_ideas"] = ideas

    # ── Persist & deliver ────────────────────────────────────────────────
    saved = _save_brief(brief)
    print(f"\n  Brief saved: {saved}")

    if args.email:
        _send_email(brief)

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
