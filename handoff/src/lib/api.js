// API client for the FPL Optimiser backend (Railway-hosted FastAPI).
//
// The backend URL is read from VITE_API_URL at build time. In dev, set it in
// `.env.local`; in production (Vercel), set it in Project Settings → Environment
// Variables.
//
// All endpoints documented in the backend repo: api.py / FastAPI auto-docs at
// `${VITE_API_URL}/docs`.

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

class ApiError extends Error {
  constructor(message, status, body) {
    super(message);
    this.status = status;
    this.body = body;
  }
}

async function request(path, { method = 'GET', body, signal } = {}) {
  const res = await fetch(`${API_URL}${path}`, {
    method,
    headers: body ? { 'Content-Type': 'application/json' } : undefined,
    body: body ? JSON.stringify(body) : undefined,
    signal,
  });
  const text = await res.text();
  const data = text ? JSON.parse(text) : null;
  if (!res.ok) {
    throw new ApiError(
      data?.detail || res.statusText || 'Request failed',
      res.status,
      data
    );
  }
  return data;
}

// ── World Cup ─────────────────────────────────────────────────────
// POST /api/wc/predict
// body: { home_team, away_team, home_advantage? }
// Returns PredictResponse — see api.py for full shape.
export function predictWorldCupMatch({ home_team, away_team, home_advantage = false }) {
  return request('/api/wc/predict', {
    method: 'POST',
    body: { home_team, away_team, home_advantage },
  });
}

export const wc = {
  teams:     ()              => request('/api/wc/teams'),
  simulate:  (n = 10000)    => request(`/api/wc/simulate?n_sim=${n}`),
  bracket:   ()              => request('/api/wc/bracket'),
  standings: ()              => request('/api/wc/standings'),
  captains:  (top = 10, md)  => request(`/api/wc/fantasy/captains?top_n=${top}${md != null ? `&matchday=${md}` : ''}`),
  players:   ()              => request('/api/wc/fantasy/players'),
  optimise:  (body)          => request('/api/wc/fantasy/optimise', { method: 'POST', body }),
};

// ── FPL "Beat the Model" (data served from committed JSON files) ───
// /api/fpl/model/gw/{gw}     — commitment + reveal for a locked gameweek
// /api/fpl/model/season       — full season record
export const fpl = {
  modelGw:     (gw)  => request(`/api/fpl/model/gw/${gw}`),
  modelSeason: ()    => request('/api/fpl/model/season'),
};

export { ApiError, API_URL };
