"""
test_fpl_receipts.py

Tests for src/fpl_receipts.py with the FPL API fully mocked at the
_fetch_json chokepoint. Run: python scripts/test_fpl_receipts.py

R1  Ungraded GW -> 409, and NO fetch is attempted
R2  Happy path: net = gross - hits on both sides; winner correct; row cached
R3  Immutability: second request performs ZERO fetches (mock raises if called)
R4  Backfill: requesting GW2 with GW1 graded-but-uncached fetches both;
    H2H covers the full graded season, not just the requested GW
R5  FPL 404 -> clean 404 "team not found"
R6  Draw counted as draw, not a win for either side
"""

from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import src.fpl_receipts as fr  # noqa: E402


def build_fixture(tmp: Path, graded: dict[int, int]) -> tuple[Path, Path]:
    """Returns (export_dir, receipts_db_path). Writes minimal result JSONs."""
    export_dir = tmp / "fpl"
    export_dir.mkdir(exist_ok=True)
    for gw, net in graded.items():
        (export_dir / f"gw{gw:02d}_result.json").write_text(json.dumps({
            "gameweek": gw, "net_points": net,
        }))
    receipts_db = tmp / "receipts.db"
    return export_dir, receipts_db


class MockFPL:
    """Programmable stand-in for _fetch_json with call accounting."""

    def __init__(self, entry_name="Test XI", gw_points=None, status=200,
                 shape="ok"):
        self.shape = shape
        self.entry_name = entry_name
        self.gw_points = gw_points or {}   # {gw: (gross, hits)}
        self.status = status
        self.calls = []

    def __call__(self, url: str) -> dict:
        self.calls.append(url)
        if self.status == 404:
            raise HTTPException(404, "FPL team not found.")
        if url.endswith("/picks/"):
            gw = int(url.rstrip("/").split("/")[-2])
            if self.shape == "renamed":       # FPL renames the fields
                return {"entry_history": {"total_points": 61, "transfers_cost": 4}}
            if self.shape == "missing":       # entry_history gone entirely
                return {"picks": [], "active_chip": None}
            gross, hits = self.gw_points[gw]
            return {"entry_history": {"points": gross,
                                      "event_transfers_cost": hits}}
        return {"name": self.entry_name}


def client(export_dir: Path, receipts_db: Path, mock: MockFPL) -> TestClient:
    fr.EXPORT_DIR = export_dir
    fr.RECEIPTS_DB_PATH = receipts_db
    fr._fetch_json = mock
    app = FastAPI()
    app.include_router(fr.router)
    return TestClient(app)


def check(label, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {label}" + (f"  ({detail})" if detail else ""))
    return cond


def main():
    ok = True

    # R1: ungraded GW
    mock = MockFPL()
    export_dir, receipts_db = build_fixture(Path(tempfile.mkdtemp()), {1: 60})
    c = client(export_dir, receipts_db, mock)
    r = c.get("/api/fpl/receipt/2/12345")
    ok &= check("R1 ungraded GW -> 409", r.status_code == 409, str(r.status_code))
    ok &= check("R1 no fetch attempted", mock.calls == [], f"calls={mock.calls}")

    # R2: happy path — user 70 gross, 8 hits -> 62 net vs model 60 -> user wins
    export_dir, receipts_db = build_fixture(Path(tempfile.mkdtemp()), {1: 60})
    mock = MockFPL(gw_points={1: (70, 8)})
    c = client(export_dir, receipts_db, mock)
    j = c.get("/api/fpl/receipt/1/12345").json()
    ok &= check("R2 user net = gross - hits", j["user"]["points_net"] == 62,
                json.dumps(j["user"]))
    ok &= check("R2 winner user (62 > 60)", j["winner"] == "user")
    ok &= check("R2 team name captured", j["user"]["team_name"] == "Test XI")
    ok &= check("R2 not from cache on first hit", j["from_cache"] is False)

    # R3: immutability — same request again; ANY fetch is a failure
    class ExplodingFetch:
        def __call__(self, url):
            raise AssertionError(f"refetch attempted: {url}")
    fr._fetch_json = ExplodingFetch()
    j = c.get("/api/fpl/receipt/1/12345").json()
    ok &= check("R3 cached: zero fetches on repeat", j["from_cache"] is True
                and j["user"]["points_net"] == 62)

    # R4: backfill — GW1+GW2 graded; fresh team requests GW2 only
    export_dir, receipts_db = build_fixture(Path(tempfile.mkdtemp()), {1: 60, 2: 55})
    mock = MockFPL(gw_points={1: (50, 0), 2: (58, 0)})
    c = client(export_dir, receipts_db, mock)
    j = c.get("/api/fpl/receipt/2/777").json()
    picks_calls = [u for u in mock.calls if u.endswith("/picks/")]
    ok &= check("R4 both graded GWs fetched", len(picks_calls) == 2,
                f"{picks_calls}")
    ok &= check("R4 H2H spans full season", j["h2h_season"] == {
        "user": 1, "model": 1, "draws": 0}, json.dumps(j["h2h_season"]))
    ok &= check("R4 requested GW correct", j["winner"] == "user"
                and j["user"]["points_net"] == 58)

    # R5: FPL 404
    export_dir, receipts_db = build_fixture(Path(tempfile.mkdtemp()), {1: 60})
    c = client(export_dir, receipts_db, MockFPL(status=404))
    r = c.get("/api/fpl/receipt/1/99999999")
    ok &= check("R5 FPL 404 -> 404", r.status_code == 404, str(r.status_code))

    # R6: draw
    export_dir, receipts_db = build_fixture(Path(tempfile.mkdtemp()), {1: 60})
    c = client(export_dir, receipts_db, MockFPL(gw_points={1: (60, 0)}))
    j = c.get("/api/fpl/receipt/1/555").json()
    ok &= check("R6 draw is a draw", j["winner"] == "draw"
                and j["h2h_season"]["draws"] == 1)

    # R7/R8: a changed FPL response shape must FAIL LOUDLY, never publish a 0.
    for shape, label in [("renamed", "renamed fields"),
                         ("missing", "no entry_history")]:
        exp, rdb = build_fixture(Path(tempfile.mkdtemp()), {1: 60})
        c = client(exp, rdb, MockFPL(gw_points={1: (70, 8)}, shape=shape))
        r = c.get("/api/fpl/receipt/1/12345")
        body = r.text
        ok &= check(f"R7 {label} -> 502, not a silent zero",
                    r.status_code == 502 and "shape has changed" in body,
                    f"{r.status_code}: {body[:70]}")

    # R8: a cache written under the previous schema (no season column) must be
    # rebuilt rather than served — its rows' season is unknowable, and FPL may
    # reassign team ids between seasons.
    exp, rdb = build_fixture(Path(tempfile.mkdtemp()), {1: 60})
    old = sqlite3.connect(rdb)
    old.execute("""CREATE TABLE receipts (gameweek_id INTEGER NOT NULL,
        team_id INTEGER NOT NULL, fetched_at TEXT NOT NULL, team_name TEXT,
        points_gross INTEGER NOT NULL, hit_points INTEGER NOT NULL,
        points_net INTEGER NOT NULL, PRIMARY KEY (gameweek_id, team_id))""")
    old.execute("INSERT INTO receipts VALUES (1,12345,'t','Stale XI',999,0,999)")
    old.commit(); old.close()

    c = client(exp, rdb, MockFPL(gw_points={1: (70, 8)}))
    j = c.get("/api/fpl/receipt/1/12345").json()
    ok &= check("R8 pre-season-column cache discarded, not served",
                j["user"]["points_net"] == 62, f"got {j['user']['points_net']}")
    cols = [r[1] for r in sqlite3.connect(rdb).execute(
        "PRAGMA table_info(receipts)")]
    ok &= check("R8 rebuilt table carries the season column", "season" in cols,
                str(cols))

    print("\n" + ("ALL PASS" if ok else "FAILURES PRESENT"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
