"""
fpl_cards.py

The visible layer:

  GET /card/{gw}/{team_id}.png?fmt=og|story
      Pillow render of the receipt. Generated on first request, written to
      CARDS_DIR, served from disk forever after — receipts for graded
      gameweeks are immutable, so the cache never invalidates.

  GET /r/{gw}/{team_id}
      Server-rendered HTML share page. THIS is the URL people post: crawlers
      read its og: tags (they never execute the SPA's JS), unfurl the card
      PNG, and humans who click land on a page with the numbers and a CTA.

Security note that is easy to miss: FPL team names are arbitrary
user-controlled strings. They are HTML-escaped everywhere they appear in the
wrapper — an unescaped team name is stored XSS served from our domain.

PUBLIC_BASE_URL env var must be the API's public origin (og:image must be an
absolute URL or unfurlers ignore it).
"""

from __future__ import annotations

import html
import io
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

from src.card_renderer import FORMATS, render_card
from src.fpl_receipts import receipt as get_receipt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CARDS_DIR = Path(os.environ.get("CARDS_DIR", PROJECT_ROOT / "cards"))
PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL",
                                 "https://api.themodelsays.com").rstrip("/")

router = APIRouter(tags=["fpl-cards"])

_IMMUTABLE = "public, max-age=31536000, immutable"


@router.get("/card/{gw}/{team_id}.png")
def card_png(gw: int, team_id: int, fmt: str = "og") -> FileResponse:
    if fmt not in FORMATS:
        raise HTTPException(422, f"fmt must be one of {sorted(FORMATS)}")
    CARDS_DIR.mkdir(parents=True, exist_ok=True)
    path = CARDS_DIR / f"{gw}_{team_id}_{fmt}.png"
    if not path.exists():
        r = get_receipt(gw, team_id)          # raises 404/409/502 upstream
        img = render_card(r, fmt=fmt)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(buf.getvalue())
        tmp.rename(path)                      # atomic: never serve half a PNG
    return FileResponse(path, media_type="image/png",
                        headers={"Cache-Control": _IMMUTABLE})


_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<meta property="og:title" content="{title}">
<meta property="og:description" content="{desc}">
<meta property="og:image" content="{img}">
<meta property="og:type" content="website">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:image" content="{img}">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
 body{{background:#0d1b2a;color:#eceff4;font-family:system-ui,sans-serif;
      display:flex;flex-direction:column;align-items:center;gap:24px;
      padding:40px 16px;text-align:center}}
 img{{max-width:min(92vw,720px);border-radius:12px}}
 a.cta{{background:#2ecc71;color:#0d1b2a;font-weight:700;padding:14px 28px;
      border-radius:8px;text-decoration:none}}
 p{{color:#8c9eb2;max-width:60ch}}
</style>
</head>
<body>
<h1>{heading}</h1>
<img src="{img}" alt="Gameweek {gw} receipt: {alt}">
<p>The model's squad is hash-committed before every deadline and revealed
after — verify it yourself on the methodology page.</p>
<a class="cta" href="https://themodelsays.com/fpl?team={team_id}">Get your own receipt</a>
</body>
</html>"""


@router.get("/r/{gw}/{team_id}", response_class=HTMLResponse)
def share_page(gw: int, team_id: int) -> HTMLResponse:
    r = get_receipt(gw, team_id)
    # Raw here; escaped exactly ONCE at the .format boundary below.
    name = r["user"].get("team_name") or f"Team {team_id}"
    u, m = r["user"]["points_net"], r["model"]["net_points"]
    h2h = r["h2h_season"]
    verdict = {"user": f"{name} beat The Model",
               "model": f"The Model beat {name}",
               "draw": f"{name} drew with The Model"}[r["winner"]]
    title = f"GW{gw}: The Model {m} – {name} {u}"
    desc = (f"{verdict}. Season: {h2h['user']}–{h2h['model']} "
            f"({h2h['draws']} drawn). Every model squad hash-committed "
            f"before the deadline.")
    img = f"{PUBLIC_BASE_URL}/card/{gw}/{team_id}.png?fmt=og"
    page = _PAGE.format(title=html.escape(title, quote=True),
                        desc=html.escape(desc, quote=True),
                        img=img, heading=html.escape(verdict, quote=True),
                        gw=gw, alt=html.escape(f"{name} {u}, The Model {m}",
                                               quote=True),
                        team_id=team_id)
    return HTMLResponse(page)
