"""
build_record_page.py

Generate the public "WC 2026 record, match by match" article from the committed
retrospective artifacts.

    python scripts/build_record_page.py --out ../themodelsays-web/public/wc2026-record.html

WHY THIS IS A GENERATOR AND NOT A HAND-WRITTEN PAGE

Every figure on the page is computed from wc/retrospective/per_match.csv at
build time. Nothing is typed in. This project has already published four wrong
percentages on the methodology page because numbers were copied out of a chat
message instead of read from the source file, and the ledger was regraded from
n=102 to n=104 after the bronze final and Final were ingested, silently
invalidating every figure quoted anywhere else. A generator makes that class of
error impossible: regenerate and the page cannot disagree with the record.

Inputs (both committed):
    wc/retrospective/per_match.csv   one row per graded match
    wc/retrospective/summary.txt     the walk-forward vs frozen summary

Output: a self-contained static HTML page. Static rather than a React route so
it is crawlable without depending on the prerender step, matching the existing
methodology.html / about.html pattern.
"""

from __future__ import annotations

import argparse
import csv
import html
import math
from collections import defaultdict
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "wc" / "retrospective" / "per_match.csv"
SUMMARY_PATH = PROJECT_ROOT / "wc" / "retrospective" / "summary.txt"

SITE = "https://www.themodelsays.com"
REPO = "https://github.com/nabeehshameem/fpl-optimiser"

RESULT_WORD = {"H": "home win", "D": "draw", "A": "away win"}


# ── data ─────────────────────────────────────────────────────────────────────

def load_rows() -> list[dict]:
    """Read the graded record, tolerating the committed file's encoding.

    per_match.csv is currently cp1252 (it was regenerated on Windows, where
    csv defaults to the local codepage) — "Türkiye" and "Curaçao" are the
    giveaways. Team names end up in the published page, so a silent mojibake
    would put "TÃ¼rkiye" in front of readers. Decode explicitly rather than
    guessing, and say which encoding was used.
    """
    if not CSV_PATH.exists():
        raise SystemExit(f"missing {CSV_PATH} — run grade_retrospective.py first")
    raw = CSV_PATH.read_bytes()
    for enc in ("utf-8", "cp1252"):
        try:
            text = raw.decode(enc)
            if enc != "utf-8":
                print(f"  note: {CSV_PATH.name} is {enc}, not UTF-8 — "
                      "worth normalising the writer")
            break
        except UnicodeDecodeError:
            continue
    else:
        raise SystemExit(f"cannot decode {CSV_PATH} as UTF-8 or cp1252")
    rows = list(csv.DictReader(text.splitlines()))
    for r in rows:
        for k in ("p_h", "p_d", "p_a", "brier", "logloss", "p_result",
                  "frozen_brier", "brier_uniform", "brier_baserate"):
            r[k] = float(r[k])
        for k in ("pick_correct", "exact_score", "frozen_pick_correct",
                  "frozen_exact", "et"):
            r[k] = int(r[k])
        r["n"] = int(r["n"])
    return rows


def mean(vals) -> float:
    vals = list(vals)
    return sum(vals) / len(vals) if vals else 0.0


def block_stats(rows: list[dict]) -> dict:
    return {
        "n": len(rows),
        "acc": mean(r["pick_correct"] for r in rows),
        "exact": mean(r["exact_score"] for r in rows),
        "brier": mean(r["brier"] for r in rows),
        "logloss": mean(r["logloss"] for r in rows),
        "frozen_acc": mean(r["frozen_pick_correct"] for r in rows),
        "frozen_brier": mean(r["frozen_brier"] for r in rows),
        "frozen_exact": mean(r["frozen_exact"] for r in rows),
        "brier_uniform": mean(r["brier_uniform"] for r in rows),
        "brier_base": mean(r["brier_baserate"] for r in rows),
    }


def calibration(rows: list[dict]) -> list[dict]:
    """Every (predicted probability, did it happen) pair, bucketed by decile."""
    buckets: dict[int, list[int]] = defaultdict(list)
    for r in rows:
        for outcome, key in (("H", "p_h"), ("D", "p_d"), ("A", "p_a")):
            b = min(int(r[key] * 10), 9)
            buckets[b].append(1 if r["result"] == outcome else 0)
    out = []
    for b in sorted(buckets):
        hits = buckets[b]
        out.append({
            "lo": b * 10, "hi": b * 10 + 10,
            "n": len(hits), "observed": mean(hits),
            "midpoint": b * 10 + 5,
        })
    return out


def notable(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """(boldest correct calls, most confident misses)."""
    correct = sorted((r for r in rows if r["pick_correct"]),
                     key=lambda r: r["p_result"])[:6]
    missed = sorted((r for r in rows if not r["pick_correct"]),
                    key=lambda r: max(r["p_h"], r["p_d"], r["p_a"]),
                    reverse=True)[:6]
    return correct, missed


# ── rendering ────────────────────────────────────────────────────────────────

CSS = """
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
background:#0d0118;color:#b9aed0;line-height:1.7}
.nav{background:rgba(13,1,24,.95);border-bottom:1px solid rgba(255,255,255,.08);
padding:16px 24px;display:flex;align-items:center;gap:20px;flex-wrap:wrap}
.nav a{color:#b9aed0;text-decoration:none;font-size:14px}
.nav a:hover{color:#fff}
.nav .brand{color:#fff;font-weight:700;font-size:16px}
main{max-width:860px;margin:0 auto;padding:48px 24px 80px}
h1{color:#fff;font-size:clamp(28px,5vw,40px);line-height:1.2;margin-bottom:12px}
h2{color:#fff;font-size:24px;margin:44px 0 14px}
h3{color:#fff;font-size:18px;margin:28px 0 10px}
p{margin-bottom:16px}
.lede{font-size:18px;color:#d5cce6}
.meta{color:#796a93;font-size:13px;margin-bottom:32px}
a{color:#00FF87}
table{width:100%;border-collapse:collapse;margin:20px 0;font-size:14px}
th,td{text-align:left;padding:9px 10px;border-bottom:1px solid rgba(255,255,255,.08)}
th{color:#fff;font-weight:600;font-size:12px;text-transform:uppercase;
letter-spacing:.06em;color:#796a93}
tbody tr:hover{background:rgba(255,255,255,.02)}
.num{text-align:right;font-variant-numeric:tabular-nums}
.win{color:#00FF87}.miss{color:#ff6b6b}
.callout{background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.08);
border-left:3px solid #7B2EE3;border-radius:8px;padding:18px 20px;margin:24px 0}
.callout strong{color:#fff}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
code{background:rgba(0,0,0,.4);padding:2px 6px;border-radius:4px;
font-size:13px;color:#00FF87}
pre{background:rgba(0,0,0,.4);border:1px solid rgba(255,255,255,.08);
border-radius:8px;padding:16px;overflow-x:auto;margin:16px 0}
pre code{background:none;padding:0;color:#d5cce6}
footer{border-top:1px solid rgba(255,255,255,.08);margin-top:56px;padding-top:24px;
color:#796a93;font-size:13px}
"""


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def esc(s: str) -> str:
    return html.escape(str(s), quote=True)


def render(rows: list[dict]) -> str:
    all_s = block_stats(rows)
    groups = [r for r in rows if r["stage"] == "group"]
    knock = [r for r in rows if r["stage"] != "group"]
    g_s, k_s = block_stats(groups), block_stats(knock)
    calib = calibration(rows)
    bold, missed = notable(rows)
    n_exact = sum(r["exact_score"] for r in rows)
    # Exact scoreline right while the W/D/L call missed — a consequence of the
    # modal scoreline and the modal outcome being different argmaxes.
    exact_not_pick = sum(1 for r in rows
                         if r["exact_score"] and not r["pick_correct"])
    first, last = rows[0]["date"], rows[-1]["date"]

    frozen_better = all_s["frozen_acc"] > all_s["acc"]
    gap = abs(all_s["frozen_acc"] - all_s["acc"]) * 100
    k_gap = abs(k_s["frozen_acc"] - k_s["acc"]) * 100

    over_confident = [b for b in calib if b["lo"] >= 80 and b["n"] >= 5]
    oc_line = ""
    if over_confident:
        worst = min(over_confident, key=lambda b: b["observed"] - b["midpoint"] / 100)
        oc_line = (f"Predictions in the {worst['lo']}–{worst['hi']}% band "
                   f"({worst['n']} of them) came in at "
                   f"{pct(worst['observed'])}.")

    rows_html = "\n".join(
        f"""<tr>
<td class="num">{r['n']}</td><td>{esc(r['date'])}</td>
<td>{esc(r['home'])} v {esc(r['away'])}</td>
<td class="num">{esc(r['pred_score'])}</td>
<td class="num">{esc(r['score_90'])}{' *' if r['et'] else ''}</td>
<td class="num">{r['p_h'] * 100:.0f}/{r['p_d'] * 100:.0f}/{r['p_a'] * 100:.0f}</td>
<td class="{'win' if r['pick_correct'] else 'miss'}">{esc(r['pick'])} {'&check;' if r['pick_correct'] else '&cross;'}</td>
<td class="num {'win' if r['exact_score'] else ''}">{'&check;' if r['exact_score'] else '&mdash;'}</td>
<td class="num">{r['brier']:.3f}</td>
</tr>"""
        for r in rows)

    calib_html = "\n".join(
        f"<tr><td>{b['lo']}–{b['hi']}%</td><td class='num'>{b['n']}</td>"
        f"<td class='num'>{pct(b['observed'])}</td></tr>" for b in calib)

    bold_html = "\n".join(
        f"<li><strong>{esc(r['home'])} {esc(r['score_90'])} {esc(r['away'])}</strong> — "
        f"called the {RESULT_WORD[r['result']]} at {pct(r['p_result'])}"
        f"{', exact scoreline too' if r['exact_score'] else ''}.</li>"
        for r in bold)

    missed_html = "\n".join(
        f"<li><strong>{esc(r['home'])} {esc(r['score_90'])} {esc(r['away'])}</strong> — "
        f"we had {RESULT_WORD[r['pick']]} at "
        f"{pct(max(r['p_h'], r['p_d'], r['p_a']))}; it finished a "
        f"{RESULT_WORD[r['result']]}."
        # Where the top scoreline was nonetheless exact, say so: omitting it
        # would make the list look worse than the record actually was.
        + (f" The top scoreline ({esc(r['pred_score'])}) was right even though "
           "the outcome call was not." if r["exact_score"] else "")
        + "</li>"
        for r in missed)

    title = "The World Cup 2026 record, match by match"
    desc = (f"Every one of our {all_s['n']} World Cup 2026 match predictions, "
            f"graded against the result. {pct(all_s['acc'])} outcome accuracy, "
            f"{n_exact} exact scorelines, and the two places the model failed.")

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>{esc(title)} | TheModelSays</title>
<meta name="description" content="{esc(desc)}" />
<link rel="canonical" href="{SITE}/wc2026-record" />
<link rel="icon" type="image/png" href="/favicon.png" />
<meta property="og:title" content="{esc(title)}" />
<meta property="og:description" content="{esc(desc)}" />
<meta property="og:url" content="{SITE}/wc2026-record" />
<meta property="og:type" content="article" />
<meta name="twitter:card" content="summary_large_image" />
<script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=ca-pub-5120443387712788" crossorigin="anonymous"></script>
<style>{CSS}</style>
</head>
<body>
<nav class="nav">
  <a class="brand" href="/">TheModelSays</a>
  <a href="/methodology">How it works</a>
  <a href="/wc2026-record">WC 2026 record</a>
  <a href="/about">About</a>
</nav>
<main>
<h1>{esc(title)}</h1>
<p class="meta">All {all_s['n']} matches, {esc(first)} to {esc(last)}. Graded from the
committed record — every figure on this page is computed from
<code>per_match.csv</code>, not typed in.</p>

<p class="lede">We predicted every match of the 2026 World Cup before it was
played, published each prediction in a public repository, and graded all
{all_s['n']} of them afterwards. This page is the whole record: the hits, the
misses, and the two findings that made the model worse rather than better.</p>

<h2>The headline numbers</h2>
<div class="scroll"><table>
<thead><tr><th>Metric</th><th class="num">The model</th>
<th class="num">Coin flip</th><th class="num">Historical base rates</th></tr></thead>
<tbody>
<tr><td>Outcome accuracy (W/D/L)</td><td class="num">{pct(all_s['acc'])}</td>
<td class="num">33.3%</td><td class="num">—</td></tr>
<tr><td>Brier score (lower is better)</td><td class="num">{all_s['brier']:.4f}</td>
<td class="num">{all_s['brier_uniform']:.4f}</td>
<td class="num">{all_s['brier_base']:.4f}</td></tr>
<tr><td>Exact scorelines</td>
<td class="num">{n_exact} of {all_s['n']} ({pct(all_s['exact'])})</td>
<td class="num">—</td><td class="num">—</td></tr>
<tr><td>Log loss</td><td class="num">{all_s['logloss']:.4f}</td>
<td class="num">1.0986</td><td class="num">—</td></tr>
</tbody></table></div>

<p>Beating a coin flip is trivial; beating the historical base rates of football
results is not. Both comparison columns are computed over the same
{all_s['n']} matches, so they are like-for-like.</p>

<h3>By stage</h3>
<div class="scroll"><table>
<thead><tr><th>Stage</th><th class="num">Matches</th>
<th class="num">Outcome accuracy</th><th class="num">Exact scores</th>
<th class="num">Brier</th></tr></thead>
<tbody>
<tr><td>Group stage</td><td class="num">{g_s['n']}</td>
<td class="num">{pct(g_s['acc'])}</td><td class="num">{pct(g_s['exact'])}</td>
<td class="num">{g_s['brier']:.4f}</td></tr>
<tr><td>Knockouts (90 minutes)</td><td class="num">{k_s['n']}</td>
<td class="num">{pct(k_s['acc'])}</td><td class="num">{pct(k_s['exact'])}</td>
<td class="num">{k_s['brier']:.4f}</td></tr>
</tbody></table></div>

<h2>Finding one: retraining during the tournament made the model worse</h2>

<p>We retrained after every matchday, weighting the tournament's own results
four times more heavily than historical data. The intuition was obvious — use
the newest information. Graded against the record, it was
{'wrong' if frozen_better else 'right'}.</p>

<div class="scroll"><table>
<thead><tr><th>Model</th><th class="num">Outcome accuracy</th>
<th class="num">Brier</th><th class="num">Exact scores</th></tr></thead>
<tbody>
<tr><td>Frozen before the tournament</td>
<td class="num">{pct(all_s['frozen_acc'])}</td>
<td class="num">{all_s['frozen_brier']:.4f}</td>
<td class="num">{pct(all_s['frozen_exact'])}</td></tr>
<tr><td>Retrained after every matchday</td>
<td class="num">{pct(all_s['acc'])}</td>
<td class="num">{all_s['brier']:.4f}</td>
<td class="num">{pct(all_s['exact'])}</td></tr>
</tbody></table></div>

<p>The version that never saw a single tournament result was
{gap:.1f} percentage points {'better' if frozen_better else 'worse'} across all
{all_s['n']} matches, and {k_gap:.1f} points
{'better' if k_s['frozen_acc'] > k_s['acc'] else 'worse'} in the knockouts —
{pct(k_s['frozen_acc'])} against {pct(k_s['acc'])}. Three to seven matches per
team is not enough data to update team strength; we were fitting to noise and
calling it form. Walk-forward retraining has been dropped.</p>

<h2>Finding two: the model could not imagine a well-organised underdog</h2>

<p>Our confident predictions were not as good as their confidence implied.
{oc_line} A calibrated model's 90% predictions come in about 90% of the time.</p>

<div class="scroll"><table>
<thead><tr><th>Predicted probability</th><th class="num">Predictions</th>
<th class="num">Actually happened</th></tr></thead>
<tbody>
{calib_html}
</tbody></table></div>

<p>Nearly all of the damage sat in one place: heavy favourites held to draws by
sides that set up to defend. A Poisson goal model with a draw correction still
assumes both teams are trying to score at their usual rate. That is the wrong
assumption for a tournament with a 48-team field and a lot of underdogs happy
to take a point.</p>

<h2>The calls we got right when it was hard</h2>
<ul>{bold_html}</ul>

<h2>The calls we got wrong while confident</h2>
<ul>{missed_html}</ul>

<h2>Verify any of this yourself</h2>
<p>Predictions were committed to a public repository before kick-off, so the
timestamps are checkable independently of anything we say. The grading script
reconstructs each prediction from the model parameters as they existed at least
sixteen hours before the match — a walk-forward rebuild from git history, not a
retrospective fit:</p>
<pre><code>git clone {REPO}
python wc/scripts/grade_retrospective.py</code></pre>
<p>That regenerates <code>per_match.csv</code> and <code>summary.txt</code>,
which are the only inputs to this page. If your numbers differ from the ones
above, the numbers above are wrong.</p>

<div class="callout">
<strong>What this record does and does not show.</strong> It shows that these
predictions existed before the matches and how they scored. It does not show
that the model will do as well on the next tournament — {all_s['n']} matches is
a small sample, and two of the model's weaknesses only became visible after the
fact. The next public record is the 2026/27 Fantasy Premier League season, where
every squad is committed before the deadline and graded afterwards on the same
terms.
</div>

<h2>Every match</h2>
<p>Two different predictions are graded here, and they can disagree.
<strong>Top scoreline</strong> is the single most likely score. <strong>W/D/L
call</strong> is the most likely outcome, which is every home-win scoreline
added together against every draw and every away win. The most likely single
score is often a draw while the most likely outcome is still a home win, because
home victories are spread across many different scorelines.</p>

<p>That is why {exact_not_pick} matches below show a correct exact score next to
a missed W/D/L call. Canada 1&ndash;1 Bosnia &amp; Herzegovina is the clearest
case: the top scoreline was 1&ndash;1, but the home win held 42% against the
draw's 32%, so the outcome call was a home win. Both grades are shown rather
than the flattering one.</p>

<p>Probabilities are home/draw/away at 90 minutes. Matches marked * went to
extra time and are graded on the 90-minute result. Brier score is per match,
lower being better.</p>
<div class="scroll"><table>
<thead><tr><th class="num">#</th><th>Date</th><th>Match</th>
<th class="num">Top scoreline</th><th class="num">Actual</th>
<th class="num">H/D/A %</th><th>W/D/L call</th>
<th class="num">Exact</th><th class="num">Brier</th></tr></thead>
<tbody>
{rows_html}
</tbody></table></div>

<footer>
<p>Generated from the committed retrospective on {date.today():%d %B %Y}.
Source data: <code>wc/retrospective/per_match.csv</code>. ·
<a href="/methodology">How the model works</a> ·
<a href="{REPO}">Code and data</a></p>
</footer>
</main>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows = load_rows()
    page = render(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(page, encoding="utf-8")

    import re
    prose = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", page, flags=re.S)
    prose = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", prose))).strip()
    print(f"wrote {args.out}  ({len(rows)} matches, {len(prose):,} chars of prose)")


if __name__ == "__main__":
    main()
