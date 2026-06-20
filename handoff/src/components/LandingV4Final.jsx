// Auto-converted from variations/v4-final.jsx in the design project.
// Original prototype used CDN React + Babel with window globals; this is the
// proper ES-module form for Vite.

import React from 'react';
import { FPL_DATA, TEAM_COLORS } from '../data.js';

// TheModelSays — final landing page
// Broad scope: FPL (primary) + Premier League match predictions + World Cup 2026.
// User-facing copy only; no model internals, no implementation jargon.

const v4 = {
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
const mono = 'JetBrains Mono, monospace';

// One-time global @keyframes for the rotating model-says line.
if (typeof document !== 'undefined' && !document.getElementById('tms-anim')) {
  const s = document.createElement('style');
  s.id = 'tms-anim';
  s.textContent = `
    @keyframes tmsFade {
      0%   { opacity: 0; transform: translateY(8px); }
      15%  { opacity: 1; transform: translateY(0); }
      85%  { opacity: 1; transform: translateY(0); }
      100% { opacity: 0; transform: translateY(-6px); }
    }
    @keyframes tmsPulse {
      0%, 100% { box-shadow: 0 0 0 0 rgba(0,255,135,0.55); }
      50%      { box-shadow: 0 0 0 6px rgba(0,255,135,0); }
    }
  `;
  document.head.appendChild(s);
}

// ── Logo: cropped circular brand mark ──────────────────────────────
function TmsLogo({ size = 32 }) {
  return (
    <img
      src="/assets/logo.png"
      width={size} height={size}
      alt="TheModelSays"
      style={{ display: 'block', borderRadius: '50%' }}
    />
  );
}

function TmsBrand({ size = 18 }) {
  return (
    <span style={{ color: v4.text, fontSize: size, fontWeight: 700, letterSpacing: '-0.02em', fontFamily: display }}>
      TheModel<span style={{ color: v4.electric }}>Says</span>
    </span>
  );
}

// ── Tags / pills ────────────────────────────────────────────────────
function V4Tag({ children, accent }) {
  return (
    <span style={{
      background: accent ? 'rgba(0,255,135,0.12)' : 'rgba(0,0,0,0.5)',
      color: accent ? v4.electric : v4.text,
      backdropFilter: 'blur(8px)',
      border: `1px solid ${accent ? 'rgba(0,255,135,0.28)' : v4.borderHi}`,
      borderRadius: 6, padding: '6px 10px',
      fontSize: 11, fontWeight: 700, letterSpacing: '0.06em',
      fontFamily: mono,
    }}>{children}</span>
  );
}

// ── Nav ─────────────────────────────────────────────────────────────
function V4Nav() {
  return (
    <div style={{
      position: 'sticky', top: 0, zIndex: 50,
      background: 'rgba(13,1,24,0.78)', backdropFilter: 'blur(14px)',
      borderBottom: `1px solid ${v4.border}`,
      padding: '16px 56px', display: 'flex', alignItems: 'center', gap: 36,
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        <TmsLogo />
        <TmsBrand />
      </div>
      <div style={{ display: 'flex', gap: 28, marginLeft: 12, fontFamily: display }}>
        {[
          ['FPL',         'app/index.html'],
          ['Matches',     'app/index.html'],
          ['World Cup',   'app/index.html'],
          ['Track record','pages/track-record.html'],
          ['Updates',     'pages/changelog.html'],
        ].map(([label, href]) => (
          <a key={label} href={href} style={{ color: v4.textDim, fontSize: 14, fontWeight: 500, textDecoration: 'none', cursor: 'pointer' }}>{label}</a>
        ))}
      </div>
      <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 18 }}>
        <span style={{ color: v4.textDim, fontSize: 14, fontWeight: 500, fontFamily: display, cursor: 'pointer' }}>Sign in</span>
        <a href="/app" style={{
          background: v4.electric, color: v4.bg, border: 0, borderRadius: 999,
          padding: '9px 18px', fontSize: 13.5, fontWeight: 700, cursor: 'pointer',
          letterSpacing: '0.03em', textTransform: 'uppercase', fontFamily: display,
          boxShadow: `0 4px 18px rgba(0,255,135,0.25)`, textDecoration: 'none', display: 'inline-block',
        }}>Open app →</a>
      </div>
    </div>
  );
}

// ── Hero ────────────────────────────────────────────────────────────
const SAYINGS = [
  'Take Haaland (C).',
  'Captain Salah away.',
  'Sell Foden. Buy Palmer.',
  'Arsenal beat Chelsea.',
  'Back England to lift it.',
];

function V4Hero() {
  const { squad } = FPL_DATA;
  const [i, setI] = React.useState(0);
  React.useEffect(() => {
    const t = setInterval(() => setI(p => (p + 1) % SAYINGS.length), 2800);
    return () => clearInterval(t);
  }, []);

  return (
    <div style={{ position: 'relative', overflow: 'hidden' }}>
      <div style={{ position: 'absolute', inset: 0, background: 'radial-gradient(ellipse at 80% 30%, rgba(0,255,135,0.12), transparent 50%)' }} />
      <div style={{ position: 'absolute', inset: 0, background: 'radial-gradient(ellipse at 5% 70%, rgba(123,46,227,0.45), transparent 55%)' }} />
      <div style={{ position: 'absolute', inset: 0, background: 'radial-gradient(ellipse at 60% 95%, rgba(255,40,130,0.12), transparent 50%)' }} />

      <div style={{ display: 'grid', gridTemplateColumns: '1.05fr 1fr', gap: 48, position: 'relative', padding: '110px 56px 90px' }}>
        {/* left — copy */}
        <div>
          <div style={{
            display: 'inline-flex', alignItems: 'center', gap: 8,
            padding: '5px 12px 5px 6px', borderRadius: 999,
            background: 'rgba(0,255,135,0.1)', border: `1px solid rgba(0,255,135,0.25)`,
            fontSize: 12, color: v4.electric, marginBottom: 28,
            fontFamily: mono, fontWeight: 500,
          }}>
            <span style={{ background: v4.electric, color: v4.bg, padding: '2px 7px', borderRadius: 999, fontWeight: 700, fontSize: 10 }}>GW28</span>
            FPL DEADLINE FRIDAY · 18:30 GMT
          </div>

          <h1 style={{
            color: v4.text, fontSize: 96, fontWeight: 700, letterSpacing: '-0.045em',
            lineHeight: 0.96, margin: 0, fontFamily: display,
          }}>
            The model<br/>says.
          </h1>

          <div style={{ marginTop: 18, height: 64, position: 'relative' }}>
            <span key={i} style={{
              display: 'inline-block', fontFamily: display, fontWeight: 700, fontSize: 44, letterSpacing: '-0.03em',
              background: `linear-gradient(120deg, ${v4.electric} 0%, ${v4.green} 60%, ${v4.electric} 100%)`,
              WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', backgroundClip: 'text',
              animation: 'tmsFade 2.8s ease infinite',
            }}>
              “{SAYINGS[i]}”
            </span>
          </div>

          <p style={{ color: v4.textDim, fontSize: 18, lineHeight: 1.55, marginTop: 22, maxWidth: 540, fontWeight: 400 }}>
            AI picks for FPL, every Premier League match, and the 2026 World Cup.
            One model. Every call. <span style={{ color: v4.text, fontWeight: 500 }}>Free, open beta.</span>
          </p>

          <div style={{ display: 'flex', gap: 12, marginTop: 36, alignItems: 'center', flexWrap: 'wrap' }}>
            <a href="/app" style={{
              background: v4.electric, color: v4.bg, border: 0, borderRadius: 999,
              padding: '15px 24px', fontSize: 14.5, fontWeight: 700, cursor: 'pointer',
              letterSpacing: '0.02em', textTransform: 'uppercase', fontFamily: display,
              boxShadow: `0 6px 28px rgba(0,255,135,0.32)`, textDecoration: 'none',
            }}>See the picks →</a>
            <a href="/track-record" style={{
              background: 'transparent', color: v4.text,
              border: `1.5px solid ${v4.borderHi}`, borderRadius: 999,
              padding: '15px 22px', fontSize: 14, fontWeight: 600, cursor: 'pointer',
              fontFamily: display, textDecoration: 'none',
            }}>How accurate?</a>
          </div>

          {/* stats row */}
          <div style={{ display: 'flex', gap: 32, marginTop: 52, alignItems: 'center' }}>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span style={{ width: 8, height: 8, borderRadius: 999, background: v4.electric, animation: 'tmsPulse 2s infinite' }} />
                <span style={{ color: v4.electric, fontSize: 11, fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', fontFamily: mono }}>ACCURACY</span>
              </div>
              <div style={{ color: v4.text, fontSize: 30, fontWeight: 700, fontFamily: display, letterSpacing: '-0.02em', marginTop: 4 }}>+11%</div>
              <div style={{ color: v4.textDim, fontSize: 12 }}>vs FPL's own picks</div>
            </div>
            <div style={{ width: 1, height: 56, background: v4.border }} />
            <div>
              <div style={{ color: v4.textVeryDim, fontSize: 11, fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', fontFamily: mono }}>COVERAGE</div>
              <div style={{ color: v4.text, fontSize: 30, fontWeight: 700, fontFamily: display, letterSpacing: '-0.02em', marginTop: 4 }}>Every match</div>
              <div style={{ color: v4.textDim, fontSize: 12 }}>PL season + WC 2026</div>
            </div>
            <div style={{ width: 1, height: 56, background: v4.border }} />
            <div>
              <div style={{ color: v4.textVeryDim, fontSize: 11, fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', fontFamily: mono }}>PRICE</div>
              <div style={{ color: v4.text, fontSize: 30, fontWeight: 700, fontFamily: display, letterSpacing: '-0.02em', marginTop: 4 }}>Free</div>
              <div style={{ color: v4.textDim, fontSize: 12 }}>open beta</div>
            </div>
          </div>
        </div>

        {/* right — pitch */}
        <V4PitchHero squad={squad} />
      </div>
    </div>
  );
}

function V4PitchHero({ squad }) {
  return (
    <div style={{ position: 'relative' }}>
      <div style={{
        position: 'relative', borderRadius: 18, overflow: 'hidden', height: 720,
        background: `linear-gradient(180deg, transparent 0%, rgba(0,0,0,0.3) 100%), linear-gradient(170deg, #0a4f2e 0%, #0d6839 50%, #0a4f2e 100%)`,
        boxShadow: '0 30px 80px -20px rgba(0,0,0,0.6), inset 0 0 0 1px rgba(255,255,255,0.1)',
      }}>
        <V4PitchLines />

        {/* top HUD */}
        <div style={{ position: 'absolute', top: 16, left: 16, right: 16, display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', zIndex: 5 }}>
          <div style={{ display: 'flex', gap: 8 }}>
            <V4Tag>3-4-3</V4Tag>
            <V4Tag accent>£99.6 / £100.0m</V4Tag>
          </div>
          <div style={{ background: 'rgba(0,0,0,0.55)', backdropFilter: 'blur(8px)', border: `1px solid ${v4.borderHi}`, borderRadius: 10, padding: '10px 14px', textAlign: 'right' }}>
            <div style={{ color: v4.textDim, fontSize: 10, letterSpacing: '0.08em', textTransform: 'uppercase', fontFamily: mono, fontWeight: 600 }}>Predicted</div>
            <div style={{ color: v4.electric, fontSize: 32, fontWeight: 700, fontFamily: display, letterSpacing: '-0.02em', lineHeight: 1 }}>71.4</div>
            <div style={{ color: v4.electric, fontSize: 11, fontFamily: mono }}>+8.2 vs your XV</div>
          </div>
        </div>

        {/* players */}
        <div style={{ position: 'absolute', inset: '92px 24px 60px', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
          <V4PitchRow players={squad.fwd} />
          <V4PitchRow players={squad.mid} />
          <V4PitchRow players={squad.def} />
          <V4PitchRow players={squad.gk} />
        </div>

        <div style={{ position: 'absolute', bottom: 16, left: 16, color: 'rgba(255,255,255,0.45)', fontFamily: mono, fontSize: 11, fontWeight: 600, letterSpacing: '0.05em' }}>
          GW28 · THE MODEL'S XV
        </div>
        <div style={{ position: 'absolute', bottom: 16, right: 16 }}>
          <V4Tag accent>+8.2 vs your team</V4Tag>
        </div>
      </div>
    </div>
  );
}

function V4PitchLines() {
  return (
    <svg style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', opacity: 0.18 }} viewBox="0 0 400 720" preserveAspectRatio="none">
      <rect x="20" y="20" width="360" height="680" stroke="white" strokeWidth="1.5" fill="none" />
      <line x1="20" y1="360" x2="380" y2="360" stroke="white" strokeWidth="1.5" />
      <circle cx="200" cy="360" r="56" stroke="white" strokeWidth="1.5" fill="none" />
      <circle cx="200" cy="360" r="2" fill="white" />
      <rect x="100" y="20" width="200" height="100" stroke="white" strokeWidth="1.5" fill="none" />
      <rect x="100" y="600" width="200" height="100" stroke="white" strokeWidth="1.5" fill="none" />
      <rect x="140" y="20" width="120" height="40" stroke="white" strokeWidth="1.5" fill="none" />
      <rect x="140" y="660" width="120" height="40" stroke="white" strokeWidth="1.5" fill="none" />
    </svg>
  );
}

function V4PitchRow({ players }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-around' }}>
      {players.map((p, i) => <V4PlayerCard key={i} player={p} />)}
    </div>
  );
}

function V4PlayerCard({ player }) {
  const tc = TEAM_COLORS[player.team] || '#888';
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4, position: 'relative' }}>
      <div style={{
        width: 56, height: 64,
        background: `linear-gradient(180deg, ${tc} 0%, ${tc} 65%, white 65%, white 100%)`,
        borderRadius: '8px 8px 16px 16px', boxShadow: '0 6px 18px rgba(0,0,0,0.5)', position: 'relative',
      }}>
        <div style={{ position: 'absolute', top: 6, left: '50%', transform: 'translateX(-50%)', color: 'white', fontSize: 9, fontWeight: 700, fontFamily: mono, letterSpacing: '0.05em' }}>{player.team}</div>
      </div>
      <div style={{ background: 'rgba(0,0,0,0.7)', backdropFilter: 'blur(6px)', borderRadius: 4, padding: '2px 7px', fontSize: 10, color: v4.text, fontWeight: 700, minWidth: 60, textAlign: 'center', fontFamily: display }}>
        {player.name}
      </div>
      <div style={{ background: v4.electric, color: v4.bg, borderRadius: 99, padding: '1px 7px', fontSize: 10, fontWeight: 800, fontFamily: mono, letterSpacing: '0.02em' }}>
        {player.ev.toFixed(1)}
      </div>
    </div>
  );
}

// ── Marquee strip ──────────────────────────────────────────────────
function V4Marquee() {
  const items = ['PICKED BY THE MODEL', 'EVERY FPL GAMEWEEK', 'EVERY PL MATCH', 'WORLD CUP 2026 READY', 'BEATS FPL\u2019s OWN PICKS', 'FREE FOREVER'];
  const all = [...items, ...items, ...items];
  return (
    <div style={{ background: v4.electric, color: v4.bg, padding: '16px 0', overflow: 'hidden', position: 'relative', borderTop: `1px solid ${v4.borderHi}`, borderBottom: `1px solid ${v4.borderHi}` }}>
      <div style={{ display: 'flex', gap: 56, whiteSpace: 'nowrap', fontFamily: display, fontWeight: 800, fontSize: 18, letterSpacing: '-0.01em' }}>
        {all.map((it, i) => (
          <span key={i} style={{ display: 'inline-flex', alignItems: 'center', gap: 56 }}>
            {it}
            <span style={{ width: 8, height: 8, borderRadius: 999, background: v4.bg, display: 'inline-block' }} />
          </span>
        ))}
      </div>
    </div>
  );
}

// ── Features (4 cards) ─────────────────────────────────────────────
function V4Features() {
  return (
    <div style={{ padding: '120px 56px', position: 'relative' }}>
      <div style={{ maxWidth: 1320, margin: '0 auto' }}>
        <div style={{ marginBottom: 64, maxWidth: 820 }}>
          <div style={{ color: v4.electric, fontSize: 13, fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', fontFamily: mono, marginBottom: 14 }}>// what the model says</div>
          <h2 style={{ color: v4.text, fontSize: 58, fontWeight: 700, letterSpacing: '-0.035em', lineHeight: 1.02, margin: 0, fontFamily: display }}>
            Every gameweek. Every match. <span style={{ color: v4.electric }}>Every call.</span>
          </h2>
          <p style={{ color: v4.textDim, fontSize: 17, lineHeight: 1.5, marginTop: 20, maxWidth: 680 }}>
            FPL is the main event — but the same model that picks your XV also calls every Premier League fixture and the 2026 World Cup, end to end.
          </p>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          <V4FeatureCard
            num="01" tag="FPL · SQUAD"
            title="The optimal XV. Every gameweek."
            body="Drop your team ID. The model rebuilds your 15 under FPL's full rules — budget, formation, club caps — and tells you exactly who's playing, who's on the bench, and who wears the armband."
            footer={[['Predicted 71.4 pts', 'GW28 · 3-4-3 optimal']]}
            widget={<WidgetCaptain />}
            widgetLabel="THIS WEEK'S CAPTAIN"
          />
          <V4FeatureCard
            num="02" tag="FPL · TRANSFERS"
            title="Take the hit, or hold?"
            body="The model values your free transfers, prices in the -4 hits, and tells you what's actually worth doing this week — across eight gameweeks ahead, not just one."
            footer={[['1 FT · 0 hits', 'optimal next 8 GWs']]}
            widget={<WidgetPlanner />}
            widgetLabel="GW28 → GW33 OUTLOOK"
          />
          <V4FeatureCard
            num="03" tag="PREMIER LEAGUE"
            title="Every fixture, called."
            body="Score, winner, both teams to score, top scorer — the model previews every Premier League fixture each weekend, with the confidence figure right next to it."
            footer={[['32 matches/week', 'updated Friday']]}
            widget={<WidgetMatches />}
            widgetLabel="THIS WEEKEND'S MATCHES"
          />
          <V4FeatureCard
            num="04" tag="WORLD CUP 2026"
            title="From group stage to lifting the trophy."
            body="The model has run the whole tournament 50,000 times. Group standings, knockouts, the final, the golden boot — all live, all updating as games are played."
            footer={[['48 teams · 64 games', 'group → final']]}
            widget={<WidgetBracket />}
            widgetLabel="MODEL'S WORLD CUP CALL"
          />
        </div>
      </div>
    </div>
  );
}

function V4FeatureCard({ num, tag, title, body, footer, widget, widgetLabel }) {
  return (
    <div style={{
      background: 'linear-gradient(180deg, rgba(255,255,255,0.04) 0%, rgba(255,255,255,0.01) 100%)',
      border: `1px solid ${v4.border}`, borderRadius: 20, padding: 32,
      display: 'grid', gridTemplateRows: 'auto auto 1fr auto', gap: 20, minHeight: 480,
    }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <span style={{ color: v4.textVeryDim, fontFamily: mono, fontSize: 12, fontWeight: 700, letterSpacing: '0.05em' }}>{num}</span>
        <span style={{ color: v4.electric, fontFamily: mono, fontSize: 11, fontWeight: 700, letterSpacing: '0.08em', padding: '4px 10px', background: 'rgba(0,255,135,0.12)', borderRadius: 4 }}>{tag}</span>
      </div>
      <div>
        <div style={{ color: v4.text, fontSize: 26, fontWeight: 700, letterSpacing: '-0.02em', lineHeight: 1.2, marginBottom: 12, fontFamily: display }}>{title}</div>
        <div style={{ color: v4.textDim, fontSize: 14.5, lineHeight: 1.55 }}>{body}</div>
      </div>
      <div style={{
        background: v4.bg2, border: `1px solid ${v4.border}`, borderRadius: 14,
        padding: '18px 18px 16px', alignSelf: 'stretch', display: 'flex', flexDirection: 'column',
      }}>
        <div style={{ color: v4.textVeryDim, fontSize: 10, letterSpacing: '0.08em', textTransform: 'uppercase', fontFamily: mono, fontWeight: 700, marginBottom: 12, display: 'flex', justifyContent: 'space-between' }}>
          <span>{widgetLabel}</span>
          <span style={{ color: v4.electric, display: 'flex', alignItems: 'center', gap: 4 }}>
            <span style={{ width: 6, height: 6, borderRadius: 999, background: v4.electric, boxShadow: `0 0 6px ${v4.electric}` }} />LIVE
          </span>
        </div>
        <div style={{ flex: 1 }}>{widget}</div>
      </div>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 10, paddingTop: 16, borderTop: `1px solid ${v4.border}` }}>
        <span style={{ color: v4.electric, fontSize: 18, fontWeight: 700, fontFamily: mono }}>{footer[0][0]}</span>
        <span style={{ color: v4.textVeryDim, fontSize: 12 }}>{footer[0][1]}</span>
      </div>
    </div>
  );
}

// ── Widgets ────────────────────────────────────────────────────────
function WidgetCaptain() {
  const rows = [
    ['Haaland', 9.8, v4.electric, '(C)'],
    ['Salah',   8.4, v4.text,    ''],
    ['Palmer',  6.9, v4.textDim, ''],
    ['Saka',    5.2, v4.textVeryDim, ''],
  ];
  const max = rows[0][1];
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      {rows.map(([n, p, c, mark]) => (
        <div key={n} style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <div style={{ width: 90, fontSize: 13, color: v4.text, fontWeight: 600, fontFamily: display }}>
            {n}{mark && <span style={{ color: v4.electric, fontFamily: mono, marginLeft: 6, fontSize: 11, fontWeight: 800 }}>{mark}</span>}
          </div>
          <div style={{ flex: 1, height: 8, background: v4.border, borderRadius: 99, overflow: 'hidden' }}>
            <div style={{ width: `${(p / max) * 100}%`, height: '100%', background: c, borderRadius: 99, boxShadow: c === v4.electric ? `0 0 12px ${v4.electric}80` : 'none' }} />
          </div>
          <div style={{ width: 50, textAlign: 'right', fontFamily: mono, fontSize: 13, color: c === v4.electric ? v4.electric : v4.textDim, fontWeight: 700 }}>{p.toFixed(1)} pts</div>
        </div>
      ))}
      <div style={{ marginTop: 4, fontSize: 11, color: v4.textVeryDim, fontFamily: mono }}>
        Captain pick · expected points this GW
      </div>
    </div>
  );
}

function WidgetPlanner() {
  const gws = FPL_DATA.gws;
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      <div style={{ display: 'flex', gap: 6 }}>
        {gws.map((g, i) => (
          <div key={i} style={{ flex: 1, background: v4.bg, border: `1px solid ${g.action !== 'hold' ? 'rgba(0,255,135,0.4)' : v4.border}`, borderRadius: 8, padding: '8px 6px', position: 'relative' }}>
            <div style={{ fontSize: 9, color: v4.textVeryDim, fontFamily: mono, fontWeight: 600 }}>GW{g.gw}</div>
            <div style={{ fontSize: 11, color: v4.text, fontWeight: 700, marginTop: 2, fontFamily: display }}>{g.fix}</div>
            <div style={{ display: 'flex', gap: 1, marginTop: 6 }}>
              {Array.from({ length: 5 }).map((_, k) => (
                <div key={k} style={{ flex: 1, height: 3, background: k < g.diff ? (g.diff <= 2 ? v4.electric : g.diff <= 3 ? v4.amber : v4.red) : v4.border, borderRadius: 1 }} />
              ))}
            </div>
            <div style={{ fontSize: 11, color: v4.electric, fontFamily: mono, fontWeight: 700, marginTop: 6 }}>{g.ev}</div>
            {g.action !== 'hold' && (
              <div style={{ position: 'absolute', top: -8, right: -4, padding: '2px 5px', background: v4.electric, color: v4.bg, fontSize: 8, fontWeight: 800, borderRadius: 3, fontFamily: mono, textTransform: 'uppercase', letterSpacing: '0.05em', whiteSpace: 'nowrap' }}>
                {g.action.split(' ')[0].toUpperCase()}
              </div>
            )}
          </div>
        ))}
      </div>
      <div style={{ marginTop: 6, display: 'flex', alignItems: 'center', gap: 14, fontSize: 11, color: v4.textVeryDim, fontFamily: mono, flexWrap: 'wrap' }}>
        <span style={{ display: 'inline-flex', alignItems: 'center', gap: 5 }}>
          <span style={{ width: 8, height: 8, background: v4.electric, borderRadius: 2 }} />EASY
        </span>
        <span style={{ display: 'inline-flex', alignItems: 'center', gap: 5 }}>
          <span style={{ width: 8, height: 8, background: v4.amber, borderRadius: 2 }} />MED
        </span>
        <span style={{ display: 'inline-flex', alignItems: 'center', gap: 5 }}>
          <span style={{ width: 8, height: 8, background: v4.red, borderRadius: 2 }} />HARD
        </span>
        <span style={{ marginLeft: 'auto', color: v4.textDim }}>chip windows highlighted</span>
      </div>
    </div>
  );
}

function WidgetMatches() {
  const fixtures = [
    { home: 'ARS', away: 'CHE', time: 'Sat 17:30', pick: 'ARS 2-1', conf: 68 },
    { home: 'LIV', away: 'MCI', time: 'Sun 16:30', pick: 'Draw 1-1', conf: 42 },
    { home: 'NEW', away: 'TOT', time: 'Sun 14:00', pick: 'NEW 2-0', conf: 54 },
  ];
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
      {fixtures.map((f, i) => (
        <div key={i} style={{ display: 'grid', gridTemplateColumns: '1fr auto auto', gap: 12, alignItems: 'center', padding: '8px 0', borderBottom: i < fixtures.length - 1 ? `1px solid ${v4.border}` : 'none' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <div style={{ width: 18, height: 18, background: TEAM_COLORS[f.home], borderRadius: 4 }} />
            <span style={{ color: v4.text, fontFamily: mono, fontSize: 11, fontWeight: 700 }}>{f.home}</span>
            <span style={{ color: v4.textVeryDim, fontSize: 11 }}>vs</span>
            <span style={{ color: v4.text, fontFamily: mono, fontSize: 11, fontWeight: 700 }}>{f.away}</span>
            <div style={{ width: 18, height: 18, background: TEAM_COLORS[f.away], borderRadius: 4 }} />
            <span style={{ color: v4.textVeryDim, fontFamily: mono, fontSize: 10, marginLeft: 6 }}>{f.time}</span>
          </div>
          <div style={{ background: 'rgba(0,255,135,0.1)', color: v4.electric, padding: '3px 9px', borderRadius: 4, fontFamily: mono, fontSize: 11, fontWeight: 700 }}>{f.pick}</div>
          <div style={{ width: 36, textAlign: 'right', color: v4.textDim, fontFamily: mono, fontSize: 11, fontWeight: 600 }}>{f.conf}%</div>
        </div>
      ))}
    </div>
  );
}

function WidgetBracket() {
  const knockouts = [
    { round: 'QF', a: 'Spain',   b: 'Brazil',    score: '2-1', winner: 'Spain' },
    { round: 'QF', a: 'England', b: 'Germany',   score: '2-1', winner: 'England' },
    { round: 'SF', a: 'Spain',   b: 'England',   score: '1-2', winner: 'England' },
    { round: 'FINAL', a: 'England', b: 'Argentina', score: '2-1', winner: 'England' },
  ];
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      {knockouts.map((k, i) => {
        const last = i === knockouts.length - 1;
        return (
          <div key={i} style={{
            display: 'grid', gridTemplateColumns: '38px 1fr auto 1fr', gap: 10, alignItems: 'center',
            padding: '6px 8px', borderRadius: 6,
            background: last ? 'rgba(0,255,135,0.08)' : 'transparent',
            border: last ? `1px solid rgba(0,255,135,0.25)` : '1px solid transparent',
          }}>
            <span style={{ color: last ? v4.electric : v4.textVeryDim, fontFamily: mono, fontSize: 10, fontWeight: 800, letterSpacing: '0.05em' }}>{k.round}</span>
            <span style={{ color: k.winner === k.a ? v4.text : v4.textDim, fontFamily: display, fontWeight: k.winner === k.a ? 700 : 500, fontSize: 12, textAlign: 'right' }}>{k.a}</span>
            <span style={{ color: v4.electric, fontFamily: mono, fontSize: 11, fontWeight: 700, padding: '2px 8px', background: 'rgba(0,0,0,0.4)', borderRadius: 4 }}>{k.score}</span>
            <span style={{ color: k.winner === k.b ? v4.text : v4.textDim, fontFamily: display, fontWeight: k.winner === k.b ? 700 : 500, fontSize: 12 }}>{k.b}</span>
          </div>
        );
      })}
      <div style={{ marginTop: 6, fontSize: 11, color: v4.textVeryDim, fontFamily: mono, display: 'flex', justifyContent: 'space-between' }}>
        <span>🏆 ENGLAND</span>
        <span>32% win prob.</span>
      </div>
    </div>
  );
}

// ── Accuracy snapshot ─────────────────────────────────────────────
function V4Accuracy() {
  const bars = [
    { gw: 22, model: 2.05, fpl: 2.19 },
    { gw: 23, model: 1.83, fpl: 2.11 },
    { gw: 24, model: 1.84, fpl: 2.12 },
    { gw: 25, model: 1.91, fpl: 2.07 },
    { gw: 26, model: 1.71, fpl: 2.08 },
    { gw: 27, model: 1.86, fpl: 2.13 },
  ];
  const max = 2.4;
  return (
    <div style={{ padding: '110px 56px', background: v4.bg2, borderTop: `1px solid ${v4.border}`, borderBottom: `1px solid ${v4.border}` }}>
      <div style={{ maxWidth: 1180, margin: '0 auto', display: 'grid', gridTemplateColumns: '1fr 1.2fr', gap: 64, alignItems: 'center' }}>
        <div>
          <div style={{ color: v4.electric, fontSize: 13, fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', fontFamily: mono, marginBottom: 14 }}>// track record</div>
          <h2 style={{ color: v4.text, fontSize: 48, fontWeight: 700, letterSpacing: '-0.035em', lineHeight: 1.05, margin: 0, fontFamily: display }}>
            The model beats FPL's <span style={{ color: v4.electric }}>own forecasts.</span>
          </h2>
          <p style={{ color: v4.textDim, fontSize: 16, lineHeight: 1.55, marginTop: 18, maxWidth: 460 }}>
            We grade ourselves against the same official numbers FPL shows you. Six of the last six gameweeks, the model's been closer to the truth.
          </p>
          <a href="/track-record" style={{
            display: 'inline-flex', alignItems: 'center', gap: 8, marginTop: 24,
            color: v4.electric, fontFamily: display, fontWeight: 700, fontSize: 14,
            textDecoration: 'none', letterSpacing: '0.02em',
          }}>See the full track record →</a>
        </div>

        <div style={{ background: v4.surface, border: `1px solid ${v4.border}`, borderRadius: 16, padding: 28 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', marginBottom: 18 }}>
            <div>
              <div style={{ color: v4.textVeryDim, fontFamily: mono, fontSize: 10, letterSpacing: '0.08em', fontWeight: 700, textTransform: 'uppercase' }}>last 6 gameweeks</div>
              <div style={{ color: v4.text, fontFamily: display, fontSize: 22, fontWeight: 700, letterSpacing: '-0.02em', marginTop: 4 }}>Lower bar = closer to reality</div>
            </div>
            <div style={{ display: 'flex', gap: 14, fontFamily: mono, fontSize: 11 }}>
              <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6, color: v4.text }}>
                <span style={{ width: 10, height: 10, background: v4.electric, borderRadius: 2 }} />The model
              </span>
              <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6, color: v4.textDim }}>
                <span style={{ width: 10, height: 10, background: v4.textVeryDim, borderRadius: 2 }} />FPL
              </span>
            </div>
          </div>
          <div style={{ display: 'flex', alignItems: 'flex-end', gap: 18, height: 200 }}>
            {bars.map(r => (
              <div key={r.gw} style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8 }}>
                <div style={{ display: 'flex', alignItems: 'flex-end', gap: 5, height: 170, width: '100%', justifyContent: 'center' }}>
                  <div style={{ width: 18, height: `${(r.model / max) * 100}%`, background: v4.electric, borderRadius: '2px 2px 0 0', boxShadow: `0 0 12px ${v4.electric}80` }} />
                  <div style={{ width: 18, height: `${(r.fpl / max) * 100}%`, background: v4.textVeryDim, borderRadius: '2px 2px 0 0' }} />
                </div>
                <div style={{ fontFamily: mono, fontSize: 10, color: v4.textVeryDim, fontWeight: 600 }}>GW{r.gw}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Quote ──────────────────────────────────────────────────────────
function V4Quote() {
  return (
    <div style={{ padding: '120px 56px', position: 'relative', overflow: 'hidden' }}>
      <div style={{ position: 'absolute', inset: 0, background: 'radial-gradient(ellipse at 50% 50%, rgba(123,46,227,0.18), transparent 60%)' }} />
      <div style={{ maxWidth: 1180, margin: '0 auto', position: 'relative', display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: 64, alignItems: 'center' }}>
        <div>
          <div style={{ color: v4.electric, fontSize: 200, fontFamily: display, fontWeight: 700, lineHeight: 0.8, letterSpacing: '-0.05em', height: 100, overflow: 'hidden' }}>“</div>
          <div style={{ color: v4.text, fontSize: 30, lineHeight: 1.32, fontWeight: 600, fontFamily: display, letterSpacing: '-0.02em', marginTop: 24 }}>
            I built TheModelSays because I was tired of guessing. Now every Friday morning, I just open the app and see <span style={{ color: v4.electric }}>what the model says.</span>
          </div>
          <div style={{ marginTop: 28, display: 'flex', alignItems: 'center', gap: 14 }}>
            <div style={{ width: 48, height: 48, borderRadius: '50%', background: `linear-gradient(135deg, ${v4.electric}, ${v4.purple})`, display: 'flex', alignItems: 'center', justifyContent: 'center', color: v4.bg, fontFamily: display, fontWeight: 800, fontSize: 18 }}>NS</div>
            <div>
              <div style={{ color: v4.text, fontSize: 15, fontWeight: 700, fontFamily: display }}>Nabeeh Shameem</div>
              <div style={{ color: v4.textDim, fontSize: 13, fontFamily: mono }}>creator</div>
            </div>
          </div>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
          {[
            ['+11%',  'better picks',  'than FPL\u2019s own'],
            ['32',     'PL matches',   'called every week'],
            ['64',     'WC fixtures',  'all predicted'],
            ['£0',     'forever',      'open beta'],
          ].map(([n, l, s]) => (
            <div key={l} style={{ background: v4.surface, border: `1px solid ${v4.border}`, borderRadius: 14, padding: 24 }}>
              <div style={{ color: v4.text, fontSize: 40, fontWeight: 700, letterSpacing: '-0.03em', fontFamily: display, lineHeight: 1 }}>{n}</div>
              <div style={{ color: v4.electric, fontSize: 12, marginTop: 8, fontWeight: 700, letterSpacing: '0.06em', textTransform: 'uppercase', fontFamily: mono }}>{l}</div>
              <div style={{ color: v4.textDim, fontSize: 12, marginTop: 4 }}>{s}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ── How it works ──────────────────────────────────────────────────
function V4HowItWorks() {
  const steps = [
    { num: '01', kicker: 'CONNECT',  title: 'Drop your FPL team ID.',     body: 'No password, no signup. We pull your squad, transfers, and chips in five seconds.' },
    { num: '02', kicker: 'ASK',       title: 'See what the model says.',  body: 'Optimal XV, captain, transfers worth taking, every match this weekend — all in one feed.' },
    { num: '03', kicker: 'PLAY',     title: 'Make the call.',             body: 'Apply moves on the FPL site yourself. Your team, your decision — the model just gives you the read.' },
  ];
  return (
    <div style={{ padding: '120px 56px' }}>
      <div style={{ maxWidth: 1320, margin: '0 auto' }}>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: 80, alignItems: 'flex-start' }}>
          <div>
            <div style={{ color: v4.electric, fontSize: 13, fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', fontFamily: mono, marginBottom: 16 }}>// how it works</div>
            <h2 style={{ color: v4.text, fontSize: 56, fontWeight: 700, letterSpacing: '-0.04em', lineHeight: 1, margin: 0, fontFamily: display }}>
              Three steps to <span style={{ color: v4.electric }}>the call.</span>
            </h2>
            <p style={{ color: v4.textDim, fontSize: 15, lineHeight: 1.55, marginTop: 20 }}>Thirty seconds end to end the first time. After that, just open the app on Friday morning.</p>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            {steps.map(s => (
              <div key={s.num} style={{ background: v4.surface, border: `1px solid ${v4.border}`, borderRadius: 18, padding: 32, display: 'grid', gridTemplateColumns: '110px 1fr', gap: 28, alignItems: 'flex-start' }}>
                <div style={{ color: v4.electric, fontSize: 60, fontWeight: 700, fontFamily: display, letterSpacing: '-0.04em', lineHeight: 0.9 }}>{s.num}</div>
                <div>
                  <div style={{ color: v4.electric, fontSize: 11, letterSpacing: '0.1em', fontFamily: mono, fontWeight: 700, marginBottom: 8 }}>{s.kicker}</div>
                  <div style={{ color: v4.text, fontSize: 26, fontWeight: 700, letterSpacing: '-0.025em', fontFamily: display, marginBottom: 8 }}>{s.title}</div>
                  <div style={{ color: v4.textDim, fontSize: 14.5, lineHeight: 1.55 }}>{s.body}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

// ── CTA ────────────────────────────────────────────────────────────
function V4CTA() {
  return (
    <div style={{ padding: '140px 56px', position: 'relative', overflow: 'hidden', borderTop: `1px solid ${v4.border}` }}>
      <div style={{ position: 'absolute', inset: 0, background: `radial-gradient(ellipse at 50% 80%, rgba(0,255,135,0.18), transparent 50%), radial-gradient(ellipse at 30% 20%, rgba(123,46,227,0.3), transparent 50%)` }} />
      <div style={{ position: 'relative', maxWidth: 980, margin: '0 auto', textAlign: 'center' }}>
        <div style={{ color: v4.electric, fontSize: 13, fontWeight: 700, letterSpacing: '0.1em', textTransform: 'uppercase', fontFamily: mono, marginBottom: 20 }}>FPL deadline · Friday 18:30 GMT</div>
        <h2 style={{ color: v4.text, fontSize: 88, fontWeight: 700, letterSpacing: '-0.045em', lineHeight: 0.96, margin: 0, fontFamily: display }}>
          See what<br/>
          <span style={{ background: `linear-gradient(120deg, ${v4.electric}, ${v4.green})`, WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', backgroundClip: 'text' }}>the model says.</span>
        </h2>
        <p style={{ color: v4.textDim, fontSize: 18, marginTop: 24, maxWidth: 540, marginInline: 'auto' }}>
          Thirty seconds to connect. Free forever. Built for everyone who plays.
        </p>
        <div style={{ display: 'flex', gap: 14, justifyContent: 'center', marginTop: 40 }}>
          <a href="/app" style={{
            background: v4.electric, color: v4.bg, border: 0, borderRadius: 999,
            padding: '18px 32px', fontSize: 16, fontWeight: 700, cursor: 'pointer',
            letterSpacing: '0.02em', textTransform: 'uppercase', fontFamily: display,
            boxShadow: `0 8px 32px rgba(0,255,135,0.35)`, textDecoration: 'none',
          }}>Open the app</a>
          <a href="/track-record" style={{
            background: 'transparent', color: v4.text, border: `1.5px solid ${v4.borderHi}`, borderRadius: 999,
            padding: '18px 26px', fontSize: 14, fontWeight: 600, cursor: 'pointer', fontFamily: display, textDecoration: 'none',
          }}>See accuracy</a>
        </div>
      </div>
    </div>
  );
}

// ── Social links ──────────────────────────────────────────────────
// Replace the `href` values with your real profile URLs.
const SOCIALS = [
  { name: 'TikTok',    handle: '@themodelsays', href: 'https://www.tiktok.com/@themodelsays',    icon: (
      <svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor" aria-hidden="true">
        <path d="M19.6 6.3a5.4 5.4 0 01-3.2-1V15a5.8 5.8 0 11-5-5.7v3.1a2.7 2.7 0 102 2.6V2h3a5.4 5.4 0 003.2 4z"/>
      </svg>
  )},
  { name: 'YouTube',   handle: '@themodelsays', href: 'https://www.youtube.com/@themodelsays',   icon: (
      <svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor" aria-hidden="true">
        <path d="M23 7.2a3 3 0 00-2.1-2.1C19 4.5 12 4.5 12 4.5s-7 0-8.9.6A3 3 0 001 7.2 31 31 0 00.5 12 31 31 0 001 16.8 3 3 0 003.1 19c1.9.5 8.9.5 8.9.5s7 0 8.9-.6a3 3 0 002.1-2.1A31 31 0 0023.5 12 31 31 0 0023 7.2zM9.8 15.5v-7l6 3.5z"/>
      </svg>
  )},
  { name: 'Instagram', handle: '@themodelsays', href: 'https://www.instagram.com/themodelsays',  icon: (
      <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true">
        <rect x="3" y="3" width="18" height="18" rx="5"/>
        <circle cx="12" cy="12" r="4"/>
        <circle cx="17.5" cy="6.5" r="1" fill="currentColor" stroke="none"/>
      </svg>
  )},
];

function V4SocialLink({ s }) {
  const [hover, setHover] = React.useState(false);
  return (
    <a href={s.href} target="_blank" rel="noopener noreferrer"
       onMouseEnter={() => setHover(true)} onMouseLeave={() => setHover(false)}
       style={{
         display: 'flex', alignItems: 'center', gap: 12,
         color: hover ? v4.electric : v4.textDim,
         textDecoration: 'none', fontFamily: display, fontSize: 14, fontWeight: 500,
         transition: 'color 160ms ease',
       }}>
      <span style={{
        width: 34, height: 34, borderRadius: 8,
        display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
        background: hover ? 'rgba(0,255,135,0.10)' : v4.surface,
        border: `1px solid ${hover ? 'rgba(0,255,135,0.32)' : v4.border}`,
        transition: 'all 160ms ease',
      }}>{s.icon}</span>
      <span style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
        <span>{s.name}</span>
        <span style={{ color: v4.textVeryDim, fontFamily: mono, fontSize: 11 }}>{s.handle}</span>
      </span>
    </a>
  );
}

// ── Footer ────────────────────────────────────────────────────────
function V4Footer() {
  const cols = [
    ['App',   ['FPL Squad', 'Transfers', 'Captain', 'PL Matches', 'World Cup 2026']],
    ['Trust', ['Track record', 'Updates', 'Privacy', 'Contact']],
  ];
  return (
    <div style={{ borderTop: `1px solid ${v4.border}`, padding: '72px 56px 36px', background: v4.bg }}>
      <div style={{ maxWidth: 1320, margin: '0 auto' }}>
        <div style={{ display: 'grid', gridTemplateColumns: '1.8fr 1fr 1fr 1.3fr', gap: 48 }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 16 }}>
              <TmsLogo />
              <TmsBrand />
            </div>
            <p style={{ color: v4.textDim, fontSize: 14, lineHeight: 1.6, maxWidth: 320, margin: 0 }}>
              AI picks for FPL, every Premier League match, and the World Cup. The same model, every weekend.
            </p>
          </div>
          {cols.map(([title, items]) => (
            <div key={title}>
              <div style={{ color: v4.electric, fontSize: 11, letterSpacing: '0.1em', textTransform: 'uppercase', fontFamily: mono, fontWeight: 700, marginBottom: 16 }}>{title}</div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                {items.map(it => <a key={it} style={{ color: v4.textDim, fontSize: 14, textDecoration: 'none', cursor: 'pointer', fontFamily: display }}>{it}</a>)}
              </div>
            </div>
          ))}
          <div>
            <div style={{ color: v4.electric, fontSize: 11, letterSpacing: '0.1em', textTransform: 'uppercase', fontFamily: mono, fontWeight: 700, marginBottom: 16 }}>Follow</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
              {SOCIALS.map(s => <V4SocialLink key={s.name} s={s} />)}
            </div>
          </div>
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 56, paddingTop: 24, borderTop: `1px solid ${v4.border}`, color: v4.textVeryDim, fontSize: 12, fontFamily: mono }}>
          <span>© 2026 THEMODELSAYS · NOT AFFILIATED WITH THE PREMIER LEAGUE OR FIFA</span>
          <span>OPEN BETA · v0.4</span>
        </div>
      </div>
    </div>
  );
}

// ── Page ──────────────────────────────────────────────────────────
function LandingV4Final() {
  return (
    <div style={{ background: v4.bg, color: v4.text, fontFamily: 'Inter, sans-serif', width: '100%', minHeight: '100%' }}>
      <V4Nav />
      <V4Hero />
      <V4Marquee />
      <V4Features />
      <V4Accuracy />
      <V4HowItWorks />
      <V4Quote />
      <V4CTA />
      <V4Footer />
    </div>
  );
}

export default LandingV4Final;
