import React from 'react';
import { wc, fpl } from '../lib/api.js';

const v = {
  bg:        '#0d0118',
  bg2:       '#15032a',
  bg3:       '#1f0a3d',
  surface:   'rgba(255,255,255,0.03)',
  surfaceHi: 'rgba(255,255,255,0.06)',
  border:    'rgba(255,255,255,0.08)',
  borderHi:  'rgba(255,255,255,0.16)',
  text:      '#ffffff',
  textDim:   '#b9aed0',
  textVeryDim: '#796a93',
  electric:  '#00FF87',
  green:     '#02EFFF',
  pink:      '#FF2882',
  purple:    '#7B2EE3',
  amber:     '#FFB020',
  red:       '#ff5577',
};

const display = 'Space Grotesk, sans-serif';
const mono    = 'JetBrains Mono, monospace';

const POS_ORDER = ['GK', 'DEF', 'MID', 'FWD'];

// GW1 deadline
const GW1 = new Date('2026-08-21T17:00:00Z');

function useCountdown(target) {
  const [diff, setDiff] = React.useState(target - Date.now());
  React.useEffect(() => {
    const t = setInterval(() => setDiff(target - Date.now()), 1000);
    return () => clearInterval(t);
  }, [target]);
  if (diff <= 0) return null;
  const d = Math.floor(diff / 86400000);
  const h = Math.floor((diff % 86400000) / 3600000);
  const m = Math.floor((diff % 3600000) / 60000);
  const s = Math.floor((diff % 60000) / 1000);
  return { d, h, m, s };
}

// ── Captain picks ──────────────────────────────────────────────────

function CaptainPicks() {
  const [data, setData] = React.useState(null);
  const [err,  setErr]  = React.useState(false);

  React.useEffect(() => {
    wc.captains(8).then(setData).catch(() => setErr(true));
  }, []);

  if (err) return (
    <Section title="Captain picks" tag="LIVE MODEL">
      <div style={{ color: v.textVeryDim, fontFamily: mono, fontSize: 13, padding: '24px 0' }}>
        Could not load captain data. Check backend.
      </div>
    </Section>
  );

  if (!data) return (
    <Section title="Captain picks" tag="LIVE MODEL">
      <div style={{ color: v.textVeryDim, fontFamily: mono, fontSize: 13, padding: '24px 0' }}>
        Loading…
      </div>
    </Section>
  );

  const picks = (data.picks || []).slice(0, 8);
  const max   = picks[0]?.projected_pts || 1;

  return (
    <Section title="Captain picks" tag="LIVE MODEL"
      sub="Expected points this gameweek. The model's top recommended armband holder is marked (C).">
      <div style={{ display: 'flex', flexDirection: 'column', gap: 0 }}>
        {picks.map((p, i) => {
          const pct = (p.projected_pts / max) * 100;
          return (
            <div key={p.id} style={{
              display: 'grid', gridTemplateColumns: '28px 120px 1fr 64px',
              alignItems: 'center', gap: 16,
              padding: '14px 0',
              borderBottom: i < picks.length - 1 ? `1px solid ${v.border}` : 'none',
            }}>
              <span style={{ color: v.textVeryDim, fontFamily: mono, fontSize: 12, fontWeight: 700 }}>
                {String(i + 1).padStart(2, '0')}
              </span>
              <div>
                <div style={{ color: v.text, fontFamily: display, fontSize: 15, fontWeight: 700, display: 'flex', alignItems: 'center', gap: 8 }}>
                  {p.name}
                  {i === 0 && (
                    <span style={{ color: v.electric, fontFamily: mono, fontSize: 10, fontWeight: 800,
                      background: 'rgba(0,255,135,0.15)', border: `1px solid rgba(0,255,135,0.35)`,
                      borderRadius: 4, padding: '2px 6px', letterSpacing: '0.04em' }}>(C)</span>
                  )}
                </div>
                <div style={{ color: v.textVeryDim, fontFamily: mono, fontSize: 11, marginTop: 2 }}>
                  {p.team} · {p.pos} · ${p.price_m}m
                </div>
              </div>
              <div style={{ height: 6, background: v.border, borderRadius: 99, overflow: 'hidden' }}>
                <div style={{
                  width: `${pct}%`, height: '100%', borderRadius: 99,
                  background: i === 0 ? v.electric : i <= 2 ? v.green : v.purple,
                  boxShadow: i === 0 ? `0 0 10px ${v.electric}60` : 'none',
                }} />
              </div>
              <div style={{ textAlign: 'right', fontFamily: mono, fontSize: 14, fontWeight: 700,
                color: i === 0 ? v.electric : v.textDim }}>
                {p.projected_pts.toFixed(1)} <span style={{ fontSize: 10, color: v.textVeryDim }}>pts</span>
              </div>
            </div>
          );
        })}
      </div>
    </Section>
  );
}

// ── Squad optimiser ────────────────────────────────────────────────

const BOOSTERS = [
  { value: null,                       label: 'No booster' },
  { value: '12th_man',                 label: '12th Man' },
  { value: 'max_captain',              label: 'Max Captain' },
  { value: 'qualification_booster',    label: 'Qualification Booster' },
  { value: 'clean_sheet_shield',       label: 'Clean Sheet Shield' },
];

function SquadOptimiser() {
  const [budget,  setBudget]  = React.useState(1000);
  const [booster, setBooster] = React.useState(null);
  const [result,  setResult]  = React.useState(null);
  const [loading, setLoading] = React.useState(false);
  const [err,     setErr]     = React.useState(null);

  function run() {
    setLoading(true); setErr(null); setResult(null);
    wc.optimise({ budget: Number(budget), booster: booster || null })
      .then(r => { setResult(r); setLoading(false); })
      .catch(e => { setErr(e.message || 'Optimisation failed'); setLoading(false); });
  }

  return (
    <Section title="Squad optimiser" tag="LIVE MODEL"
      sub="Build the model's optimal squad under the WC2026 fantasy rules. FPL squad optimiser goes live GW1.">
      <div style={{ display: 'flex', gap: 16, alignItems: 'flex-end', flexWrap: 'wrap', marginBottom: 28 }}>
        <label style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          <span style={{ color: v.textVeryDim, fontFamily: mono, fontSize: 11, fontWeight: 700, letterSpacing: '0.06em', textTransform: 'uppercase' }}>Budget</span>
          <select value={budget} onChange={e => setBudget(e.target.value)} style={inputStyle}>
            <option value={800}>$80m</option>
            <option value={900}>$90m</option>
            <option value={1000}>$100m (default)</option>
            <option value={1100}>$110m</option>
            <option value={1200}>$120m</option>
          </select>
        </label>
        <label style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          <span style={{ color: v.textVeryDim, fontFamily: mono, fontSize: 11, fontWeight: 700, letterSpacing: '0.06em', textTransform: 'uppercase' }}>Booster</span>
          <select value={booster || ''} onChange={e => setBooster(e.target.value || null)} style={inputStyle}>
            {BOOSTERS.map(b => <option key={b.value || 'none'} value={b.value || ''}>{b.label}</option>)}
          </select>
        </label>
        <button onClick={run} disabled={loading} style={btnStyle(loading)}>
          {loading ? 'Optimising…' : 'Optimise squad →'}
        </button>
      </div>

      {err && (
        <div style={{ color: v.red, fontFamily: mono, fontSize: 13, padding: '12px 16px',
          background: 'rgba(255,85,119,0.1)', border: `1px solid rgba(255,85,119,0.25)`, borderRadius: 8 }}>
          {err}
        </div>
      )}

      {result && <OptimiseResult result={result} />}
    </Section>
  );
}

const inputStyle = {
  background: 'rgba(255,255,255,0.06)', border: `1px solid ${v.borderHi}`,
  color: v.text, borderRadius: 8, padding: '10px 14px',
  fontFamily: mono, fontSize: 13, cursor: 'pointer', minWidth: 160,
};

function btnStyle(disabled) {
  return {
    background: disabled ? 'rgba(0,255,135,0.3)' : v.electric,
    color: v.bg, border: 0, borderRadius: 999,
    padding: '11px 22px', fontSize: 13, fontWeight: 700, cursor: disabled ? 'not-allowed' : 'pointer',
    fontFamily: display, letterSpacing: '0.02em', textTransform: 'uppercase',
    boxShadow: disabled ? 'none' : `0 4px 18px rgba(0,255,135,0.28)`,
    transition: 'all 0.15s ease',
  };
}

function OptimiseResult({ result }) {
  const { starters = [], bench = [], captain, total_pts, total_cost_m, booster: bstr } = result;
  const byPos = Object.fromEntries(POS_ORDER.map(pos => [pos, []]));
  starters.forEach(p => byPos[p.pos]?.push(p));

  return (
    <div style={{ marginTop: 8 }}>
      <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap', marginBottom: 20 }}>
        <Stat label="Expected pts" value={total_pts.toFixed(1)} accent />
        <Stat label="Total cost"   value={`$${total_cost_m}m`} />
        <Stat label="Captain"      value={captain?.name || '—'} />
        {bstr && <Stat label="Booster" value={bstr.replace(/_/g, ' ')} />}
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        {POS_ORDER.filter(pos => byPos[pos].length > 0).map(pos => (
          <div key={pos} style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
            <span style={{ color: v.textVeryDim, fontFamily: mono, fontSize: 10, fontWeight: 700,
              letterSpacing: '0.08em', width: 36, flexShrink: 0 }}>{pos}</span>
            {byPos[pos].map(p => <PlayerChip key={p.id} p={p} isCapt={captain?.id === p.id} />)}
          </div>
        ))}
        {bench.length > 0 && (
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center', marginTop: 8,
            paddingTop: 8, borderTop: `1px dashed ${v.border}` }}>
            <span style={{ color: v.textVeryDim, fontFamily: mono, fontSize: 10, fontWeight: 700,
              letterSpacing: '0.08em', width: 36, flexShrink: 0 }}>BEN</span>
            {bench.map(p => <PlayerChip key={p.id} p={p} dim />)}
          </div>
        )}
      </div>
    </div>
  );
}

function PlayerChip({ p, isCapt, dim }) {
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 6,
      background: dim ? 'rgba(255,255,255,0.03)' : 'rgba(255,255,255,0.06)',
      border: `1px solid ${isCapt ? 'rgba(0,255,135,0.4)' : v.border}`,
      borderRadius: 8, padding: '6px 10px',
    }}>
      <span style={{ color: dim ? v.textVeryDim : v.text, fontFamily: display, fontSize: 12, fontWeight: 600 }}>
        {p.name}
      </span>
      {isCapt && (
        <span style={{ color: v.electric, fontFamily: mono, fontSize: 10, fontWeight: 800 }}>(C)</span>
      )}
      <span style={{ color: v.textVeryDim, fontFamily: mono, fontSize: 10 }}>{p.projected_pts.toFixed(1)}</span>
    </div>
  );
}

function Stat({ label, value, accent }) {
  return (
    <div>
      <div style={{ color: v.textVeryDim, fontFamily: mono, fontSize: 10, fontWeight: 700,
        letterSpacing: '0.08em', textTransform: 'uppercase', marginBottom: 4 }}>{label}</div>
      <div style={{ color: accent ? v.electric : v.text, fontFamily: display, fontSize: 22, fontWeight: 700, letterSpacing: '-0.02em' }}>
        {value}
      </div>
    </div>
  );
}

// ── FPL GW1 countdown ─────────────────────────────────────────────

function FplCountdown() {
  const cd = useCountdown(GW1.getTime());
  const [gw1Data, setGw1Data] = React.useState(null);

  React.useEffect(() => {
    fpl.modelGw(1)
      .then(setGw1Data)
      .catch(() => setGw1Data(false));
  }, []);

  return (
    <Section title="FPL 2026/27" tag="COMING SOON"
      sub="The model locks its GW1 squad before the deadline. Once locked, the squad hash is committed — no editing after the fact.">

      {(!gw1Data) && (
        <div style={{
          display: 'flex', gap: 24, marginBottom: 28, flexWrap: 'wrap',
        }}>
          {cd ? (
            [['Days', cd.d], ['Hours', cd.h], ['Mins', cd.m], ['Secs', cd.s]].map(([label, val]) => (
              <div key={label} style={{ textAlign: 'center' }}>
                <div style={{ color: v.electric, fontFamily: mono, fontSize: 36, fontWeight: 700, lineHeight: 1,
                  letterSpacing: '-0.02em', fontVariantNumeric: 'tabular-nums' }}>
                  {String(val).padStart(2, '0')}
                </div>
                <div style={{ color: v.textVeryDim, fontFamily: mono, fontSize: 11, fontWeight: 700,
                  letterSpacing: '0.08em', textTransform: 'uppercase', marginTop: 4 }}>{label}</div>
              </div>
            ))
          ) : (
            <div style={{ color: v.electric, fontFamily: display, fontSize: 18, fontWeight: 700 }}>
              GW1 live — check the picks above
            </div>
          )}
        </div>
      )}

      {gw1Data && gw1Data !== false && (
        <div style={{
          background: 'rgba(0,255,135,0.06)', border: `1px solid rgba(0,255,135,0.2)`,
          borderRadius: 12, padding: '18px 20px',
        }}>
          <div style={{ color: v.electric, fontFamily: mono, fontSize: 11, fontWeight: 700,
            letterSpacing: '0.06em', textTransform: 'uppercase', marginBottom: 8 }}>
            GW1 locked · {gw1Data.revealed ? 'squad revealed' : 'hash committed'}
          </div>
          {gw1Data.squad && (
            <div style={{ color: v.textDim, fontFamily: mono, fontSize: 12 }}>
              {gw1Data.squad.filter(p => p.is_xi).map(p => p.name).join(', ')}
            </div>
          )}
          {!gw1Data.squad && (
            <div style={{ color: v.textDim, fontFamily: mono, fontSize: 12 }}>
              Hash: {gw1Data.squad_hash?.slice(0, 24)}…
            </div>
          )}
          {gw1Data.result && (
            <div style={{ marginTop: 12, display: 'flex', gap: 20 }}>
              <Stat label="Net pts"   value={gw1Data.result.net_points} accent />
              <Stat label="Gross pts" value={gw1Data.result.gross_points} />
            </div>
          )}
        </div>
      )}

      <div style={{ marginTop: 20, color: v.textVeryDim, fontFamily: mono, fontSize: 12 }}>
        GW1 deadline: Fri 21 Aug ~18:00 BST &nbsp;·&nbsp;
        Lock runs ~10h before &nbsp;·&nbsp;
        Track record updates after each GW
      </div>
    </Section>
  );
}

// ── Shared layout pieces ───────────────────────────────────────────

function Section({ title, tag, sub, children }) {
  return (
    <div style={{
      background: 'linear-gradient(180deg, rgba(255,255,255,0.04) 0%, rgba(255,255,255,0.01) 100%)',
      border: `1px solid ${v.border}`, borderRadius: 20, padding: '32px 36px', marginBottom: 16,
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: sub ? 8 : 24 }}>
        <h2 style={{ color: v.text, fontFamily: display, fontSize: 22, fontWeight: 700,
          letterSpacing: '-0.02em', margin: 0 }}>{title}</h2>
        <span style={{ color: v.electric, fontFamily: mono, fontSize: 10, fontWeight: 700,
          letterSpacing: '0.08em', padding: '3px 8px',
          background: 'rgba(0,255,135,0.12)', borderRadius: 4 }}>{tag}</span>
        <span style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 5, color: v.electric, fontFamily: mono, fontSize: 10 }}>
          <span style={{ width: 6, height: 6, borderRadius: 999, background: v.electric, boxShadow: `0 0 6px ${v.electric}` }} />
          LIVE
        </span>
      </div>
      {sub && (
        <p style={{ color: v.textVeryDim, fontFamily: display, fontSize: 14, lineHeight: 1.5,
          marginTop: 0, marginBottom: 24 }}>{sub}</p>
      )}
      {children}
    </div>
  );
}

function AppNav() {
  return (
    <div style={{
      position: 'sticky', top: 0, zIndex: 50,
      background: 'rgba(13,1,24,0.82)', backdropFilter: 'blur(14px)',
      borderBottom: `1px solid ${v.border}`,
      padding: '14px 40px', display: 'flex', alignItems: 'center', gap: 24,
    }}>
      <a href="/" style={{ display: 'flex', alignItems: 'center', gap: 8, textDecoration: 'none' }}>
        <img src="/assets/logo.png" width={28} height={28} alt="TheModelSays"
          style={{ borderRadius: '50%' }} />
        <span style={{ color: v.text, fontFamily: display, fontSize: 16, fontWeight: 700, letterSpacing: '-0.02em' }}>
          TheModel<span style={{ color: v.electric }}>Says</span>
        </span>
      </a>
      <div style={{ display: 'flex', gap: 22, marginLeft: 8 }}>
        {[['Captains', '#captains'], ['Optimise', '#optimise'], ['FPL GW1', '#fpl']].map(([label, href]) => (
          <a key={label} href={href} style={{ color: v.textDim, fontFamily: display,
            fontSize: 14, fontWeight: 500, textDecoration: 'none' }}>{label}</a>
        ))}
      </div>
      <a href="/" style={{ marginLeft: 'auto', color: v.textDim, fontFamily: display,
        fontSize: 13, textDecoration: 'none' }}>← Back to home</a>
    </div>
  );
}

// ── Page root ──────────────────────────────────────────────────────

export default function AppPage() {
  return (
    <div style={{ background: v.bg, minHeight: '100vh', color: v.text }}>
      <AppNav />
      <div style={{ maxWidth: 860, margin: '0 auto', padding: '48px 24px 80px' }}>
        <div style={{ marginBottom: 40 }}>
          <div style={{ color: v.electric, fontFamily: mono, fontSize: 11, fontWeight: 700,
            letterSpacing: '0.08em', textTransform: 'uppercase', marginBottom: 10 }}>
            // TheModelSays · Live tools
          </div>
          <h1 style={{ color: v.text, fontFamily: display, fontSize: 48, fontWeight: 700,
            letterSpacing: '-0.04em', lineHeight: 1, margin: 0 }}>
            The model is<br /><span style={{ color: v.electric }}>running.</span>
          </h1>
          <p style={{ color: v.textDim, fontFamily: display, fontSize: 16, lineHeight: 1.55,
            marginTop: 14, maxWidth: 560 }}>
            Live captain picks, squad optimisation, and the FPL GW1 countdown.
            All powered by the same Dixon-Coles model that called Spain in March.
          </p>
        </div>

        <div id="captains">
          <CaptainPicks />
        </div>
        <div id="optimise">
          <SquadOptimiser />
        </div>
        <div id="fpl">
          <FplCountdown />
        </div>
      </div>
    </div>
  );
}
