# themodelsays-web — frontend handoff

This is the production frontend for **themodelsays.com**. It was scaffolded from a design prototype built in Claude (Vite + React 18). The landing page is fully designed and converted to a proper ES-module React component. From here you need to:

1. Verify it runs locally
2. Wire it to the live Railway backend
3. Build the `/app` and `/track-record` routes
4. Deploy to Vercel + attach the domain

---

## Project structure

```
.
├── index.html                       — Vite entry (head + meta + OG tags)
├── package.json                     — react 18 + vite + @vitejs/plugin-react
├── vite.config.js
├── vercel.json                      — SPA rewrites + asset cache headers
├── .env.example                     — copy to .env.local
├── public/
│   └── assets/                      — logo.png, logo-mark.png (favicon)
└── src/
    ├── main.jsx                     — ReactDOM.createRoot entry
    ├── App.jsx                      — currently just renders the landing page
    ├── index.css                    — body bg / font reset
    ├── data.js                      — FPL sample data (presentational only)
    ├── lib/
    │   └── api.js                   — typed client for the Railway API
    └── components/
        └── LandingV4Final.jsx       — the landing page (820 lines, self-contained)
```

The landing page was originally built with CDN React + in-browser Babel + `window` globals. It's been auto-converted to a proper ES module: `import React from 'react'`, `import { FPL_DATA, TEAM_COLORS } from '../data.js'`, default export. **Do not refactor it just to refactor it** — it works, it matches the approved design, leave it alone unless changing the design.

---

## Backend context (critical — don't redo this)

- The backend already exists. It's a **FastAPI app deployed to Railway**, not Fly.io and not Streamlit. Older planning docs in the design project (`deploy/` folder, references to `fly.toml` / `streamlit_app.py`) are superseded — ignore them.
- The Railway URL goes in `VITE_API_URL`. No trailing slash.
- The endpoint the landing page needs to call is **`POST /api/wc/predict`** (World Cup match score predictor). A pre-built helper is in `src/lib/api.js` — `predictWorldCupMatch({ home, away, stage })`.
- **Confirm the response shape** against the actual Railway endpoint and adjust the helper if needed. The `lib/api.js` comments assume `{ home_score, away_score, win_prob, draw_prob, ... }` — verify.
- **CORS:** the Railway FastAPI app must allow the Vercel preview domain, the production domain (`themodelsays.com` + `www.themodelsays.com`), and `http://localhost:5173` for local dev. If you don't already have CORS middleware configured, add:
  ```py
  from fastapi.middleware.cors import CORSMiddleware
  app.add_middleware(
      CORSMiddleware,
      allow_origins=[
          "http://localhost:5173",
          "https://themodelsays.com",
          "https://www.themodelsays.com",
          "https://themodelsays-web.vercel.app",  # adjust to actual Vercel project name
      ],
      allow_methods=["*"],
      allow_headers=["*"],
  )
  ```
  Patterns like `https://*.vercel.app` need `allow_origin_regex` instead — set that if you want preview-deploy URLs to work.

---

## Setup (local)

```bash
npm install
cp .env.example .env.local
# edit .env.local — paste your Railway URL
npm run dev
```

Opens at http://localhost:5173. You should see the landing page, exactly as designed. Confirm:

- Logo + brand mark render (top-left of nav, footer)
- Animated rotating quote in the hero (`"Take Haaland (C)."` etc.)
- Pitch graphic with the optimal XV
- Four feature cards with mini-widgets
- Accuracy bar chart
- Footer with TikTok / YouTube / Instagram links (replace `href`s with real profile URLs in `src/components/LandingV4Final.jsx` near the `SOCIALS` array)

---

## What's NOT built yet

The landing page links to **`/app`** and **`/track-record`** — both currently 404. You need to decide:

1. **Build them as React routes** in this repo (add `react-router-dom`, create `src/pages/App.jsx` + `src/pages/TrackRecord.jsx`). Recommended — keeps everything in one Vercel deployment.
2. **Or** keep them as separate sub-deployments and rewrite paths in `vercel.json`.

The original prototype had separate `app/index.html` + `pages/track-record.html` files. If you have those designs handy, port them into React components under `src/pages/` and add a router. If not, they need design + build work — flag it back.

### World Cup score predictor — currently illustration only

The "World Cup 2026" feature card on the landing page shows a `<WidgetBracket />` — that's **static sample data** (England beat Argentina in the final). If you want an **interactive widget** where a visitor picks two teams and gets a live prediction from `/api/wc/predict`, that's new work. Suggested approach:

- Add a new component `src/components/WorldCupPredictor.jsx`
- Two `<select>`s for home/away team
- Button → calls `predictWorldCupMatch(...)` from `lib/api.js`
- Render the result inline
- Either replace `<WidgetBracket />` in the feature card, or add it as a new section on the page

Ask the user where they want it before building. Don't silently swap out the static bracket — the design currently leans on it for visual completeness.

---

## Deploy to Vercel

1. Push this repo to GitHub.
2. vercel.com → New Project → import the repo.
3. Framework preset: **Vite** (auto-detected).
4. Environment Variables → add `VITE_API_URL` = your Railway URL.
5. Deploy. You get `themodelsays-web.vercel.app` immediately.
6. Project Settings → Domains → add `themodelsays.com` + `www.themodelsays.com`. Vercel walks you through the DNS records (an A record for the apex and a CNAME for `www`, or full nameserver delegation if you prefer).
7. Add the production domain to the backend's CORS allowlist (see Backend context above) and redeploy Railway.

---

## Common gotchas

- **`Failed to fetch` / CORS errors in browser console**: the backend isn't allowing the frontend's origin. Fix CORS on Railway, not in the frontend.
- **Env vars not picked up**: Vite only exposes variables prefixed with `VITE_`. Restart `npm run dev` after editing `.env.local`.
- **Logo not rendering**: assets live in `/public/assets/`. In code they're referenced as `/assets/logo.png` (absolute, root-relative). Don't change to relative paths — they'll break in nested routes.
- **Fonts**: loaded from Google Fonts in `index.html`. If you self-host later, update both the `<link>` and the `fontFamily` declarations inside `LandingV4Final.jsx`.
- **TypeScript**: not set up. If you want it, `npm install -D typescript @types/react @types/react-dom` and rename files. Not required.

---

## Iterating with the design project

The design source of truth lives in the Claude Design project (the one with `variations/v4-final.jsx`). If the user makes design changes there and wants them reflected here:

- Small tweaks → port the edit by hand (it's React JSX on both sides).
- Big restructure → re-run the conversion: the deltas to apply are (1) add `import React from 'react'` + `import { FPL_DATA, TEAM_COLORS } from '../data.js'` at the top, (2) replace `window.FPL_DATA` / `window.TEAM_COLORS` with the imported names, (3) replace `src="assets/logo.png"` with `src="/assets/logo.png"`, (4) replace internal `href` placeholders, (5) remove the trailing `window.LandingV4Final = LandingV4Final;` assignment, (6) add `export default LandingV4Final;`.

---

## Open questions to bring back to the user

1. Should `/app` and `/track-record` be built in this repo (recommended) or kept as separate deployments?
2. The World Cup feature card: keep the static bracket illustration, or replace with a live `/api/wc/predict` interactive widget? (Or add the live widget as a new section?)
3. Social handles in the footer — the placeholder is `@themodelsays` on all three platforms. Confirm the real handles before launch.
4. Analytics — Plausible / Fathom / Vercel Analytics? Nothing is wired up yet.
5. Email capture / waitlist — not in scope of the landing page as designed. Add if needed.
