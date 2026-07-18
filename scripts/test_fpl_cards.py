"""
test_fpl_cards.py

Tests for the card renderer and share wrapper. Run:
  python scripts/test_fpl_cards.py            # verify against goldens
  python scripts/test_fpl_cards.py --update   # regenerate goldens (after a
                                              # Pillow/FreeType upgrade —
                                              # eyeball renders first!)

C1  Both formats render at exact contract dimensions
C2  Pixel-hash matches committed goldens (deterministic rendering)
C3  User-win and model-win renders differ (winner highlight is real)
C4  Endpoint disk cache: second request renders NOTHING (renderer explodes)
C5  fmt validation -> 422
W1  /r/ page carries absolute og:image URL and the receipt numbers
W2  XSS: hostile team name arrives escaped, never raw
"""

from __future__ import annotations

import hashlib
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import src.card_renderer as cr  # noqa: E402
import src.fpl_cards as fc  # noqa: E402

GOLDEN_DIR = PROJECT_ROOT / "tests" / "goldens"


def fixture_receipt(winner="model", name="Nabeeh's XI"):
    u, m = (74, 61) if winner == "user" else (61, 74) if winner == "model" else (60, 60)
    return {
        "gameweek": 7,
        "user": {"team_id": 123, "team_name": name, "points_gross": u + 4,
                 "hit_points": 4, "points_net": u},
        "model": {"net_points": m},
        "winner": winner,
        "h2h_season": {"user": 3, "model": 5, "draws": 1},
        "from_cache": True,
    }


def pixel_hash(img) -> str:
    return hashlib.sha256(img.tobytes()).hexdigest()


def check(label, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {label}" + (f"  ({detail})" if detail else ""))
    return cond


def main():
    update = "--update" in sys.argv
    ok = True

    # C1 + C2
    for fmt, (w, h) in cr.FORMATS.items():
        img = cr.render_card(fixture_receipt(), fmt=fmt)
        ok &= check(f"C1 {fmt} dimensions {w}x{h}", img.size == (w, h),
                    str(img.size))
        golden = GOLDEN_DIR / f"card_{fmt}.sha256"
        ph = pixel_hash(img)
        if update:
            GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
            golden.write_text(ph)
            img.save(GOLDEN_DIR / f"card_{fmt}.png")   # for eyeballing
            print(f"[GOLD] {fmt} golden updated: {ph[:16]}…")
        else:
            if not golden.exists():
                ok &= check(f"C2 {fmt} golden exists", False,
                            "run with --update once, eyeball, commit")
            else:
                ok &= check(f"C2 {fmt} matches golden",
                            golden.read_text().strip() == ph, ph[:16])

    # C3: the highlight must actually change the pixels
    a = pixel_hash(cr.render_card(fixture_receipt("user")))
    b = pixel_hash(cr.render_card(fixture_receipt("model")))
    ok &= check("C3 winner highlight changes render", a != b)

    # C4/C5: endpoint cache + validation
    fc.CARDS_DIR = Path(tempfile.mkdtemp())
    fc.get_receipt = lambda gw, team_id: fixture_receipt()
    app = FastAPI()
    app.include_router(fc.router)
    c = TestClient(app)

    r = c.get("/card/7/123.png?fmt=og")
    ok &= check("C4 first request renders PNG", r.status_code == 200
                and r.headers["content-type"] == "image/png"
                and "immutable" in r.headers["cache-control"])

    def explode(*a, **k):
        raise AssertionError("re-render attempted on cached card")
    fc.render_card = explode
    fc.get_receipt = explode
    r = c.get("/card/7/123.png?fmt=og")
    ok &= check("C4 second request served from disk", r.status_code == 200)

    r = c.get("/card/7/123.png?fmt=gif")
    ok &= check("C5 bad fmt -> 422", r.status_code == 422, str(r.status_code))

    # W1/W2: share page
    fc.get_receipt = lambda gw, team_id: fixture_receipt(
        name='<script>alert(1)</script>"pwn')
    r = c.get("/r/7/123")
    body = r.text
    ok &= check("W1 absolute og:image present",
                'og:image" content="https://' in body
                and "/card/7/123.png?fmt=og" in body)
    ok &= check("W1 numbers in page", "74" in body and "61" in body)
    ok &= check("W2 hostile name escaped", "<script>" not in body
                and "&lt;script&gt;" in body)

    print("\n" + ("ALL PASS" if ok else "FAILURES PRESENT"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
