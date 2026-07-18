"""
card_renderer.py

Renders the "Beat the Model" receipt card as a PIL Image. One layout function
parameterised by (width, height) — the OG unfurl (1200x630) and the
story/screenshot format (1080x1920) are the same code path, per the Stage 4
design contract.

Determinism: fonts are pinned in assets/fonts/ (DejaVu — Bitstream Vera
licence, freely redistributable). Given the same Pillow + FreeType, renders
are byte-identical, which is what the golden-file tests rely on. Goldens are
environment-pinned: regenerate with test_fpl_cards.py --update after any
Pillow/FreeType upgrade, and eyeball the new renders before committing.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FONT_DIR = PROJECT_ROOT / "assets" / "fonts"
FONT_REG = FONT_DIR / "DejaVuSans.ttf"
FONT_BOLD = FONT_DIR / "DejaVuSans-Bold.ttf"

# palette
BG = (13, 27, 42)          # deep navy
PANEL = (22, 40, 60)
FG = (236, 239, 244)       # off-white
MUTED = (140, 158, 178)
WIN = (46, 204, 113)       # green
LOSE = (155, 170, 186)
DRAW = (241, 196, 15)      # amber

FORMATS = {"og": (1200, 630), "story": (1080, 1920)}


def _font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size)


def _center_text(d: ImageDraw.ImageDraw, xy, text, font, fill):
    x, y = xy
    l, t, r, b = d.textbbox((0, 0), text, font=font)
    d.text((x - (r - l) / 2 - l, y - (b - t) / 2 - t), text, font=font, fill=fill)


def render_card(receipt: dict, fmt: str = "og") -> Image.Image:
    """receipt: the /api/fpl/receipt response dict."""
    if fmt not in FORMATS:
        raise ValueError(f"fmt must be one of {sorted(FORMATS)}")
    w, h = FORMATS[fmt]
    img = Image.new("RGB", (w, h), BG)
    d = ImageDraw.Draw(img)

    u = receipt["user"]
    m_net = receipt["model"]["net_points"]
    u_net = u["points_net"]
    winner = receipt["winner"]
    h2h = receipt["h2h_season"]
    gw = receipt["gameweek"]
    name = (u.get("team_name") or f"Team {u['team_id']}")[:24]

    s = min(w, h) / 630.0            # scale unit
    portrait = h > w
    cx = w / 2

    f_head = _font(FONT_BOLD, int(34 * s))
    f_gw = _font(FONT_REG, int(24 * s))
    f_name = _font(FONT_REG, int(26 * s))
    f_score = _font(FONT_BOLD, int(120 * s))
    f_vs = _font(FONT_REG, int(30 * s))
    f_h2h = _font(FONT_BOLD, int(26 * s))
    f_foot = _font(FONT_REG, int(20 * s))

    # header
    top = h * (0.10 if portrait else 0.09)
    _center_text(d, (cx, top), "THE MODEL SAYS", f_head, FG)
    _center_text(d, (cx, top + 46 * s), f"GAMEWEEK {gw} RECEIPT", f_gw, MUTED)

    # score blocks
    def block(center_x, center_y, label, score, won):
        colour = WIN if won else (DRAW if winner == "draw" else LOSE)
        _center_text(d, (center_x, center_y - 90 * s), label, f_name,
                     FG if won else MUTED)
        _center_text(d, (center_x, center_y), str(score), f_score, colour)
        if won:
            _center_text(d, (center_x, center_y + 92 * s), "WINNER", f_h2h, WIN)

    if portrait:
        block(cx, h * 0.36, name, u_net, winner == "user")
        _center_text(d, (cx, h * 0.505), "vs", f_vs, MUTED)
        block(cx, h * 0.65, "THE MODEL", m_net, winner == "model")
        strip_y = h * 0.83
    else:
        block(w * 0.28, h * 0.52, name, u_net, winner == "user")
        _center_text(d, (cx, h * 0.52), "vs", f_vs, MUTED)
        block(w * 0.72, h * 0.52, "THE MODEL", m_net, winner == "model")
        strip_y = h * 0.83

    # season strip
    strip = (f"SEASON  ·  YOU {h2h['user']} — {h2h['model']} MODEL"
             + (f"  ·  {h2h['draws']} DRAWN" if h2h["draws"] else ""))
    _center_text(d, (cx, strip_y), strip, f_h2h, FG)

    # footer
    _center_text(d, (cx, h * 0.93), "themodelsays.com/fpl", f_foot, MUTED)
    return img
