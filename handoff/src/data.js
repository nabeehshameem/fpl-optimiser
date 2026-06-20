// Shared FPL-realistic sample data used by the landing page.
// This is presentational sample data, NOT live API data. The widgets on the
// landing page are marketing illustrations. Live data lives behind the
// product app (/app), which calls the Railway backend via src/lib/api.js.

export const FPL_DATA = {
  topPicks: [
    { name: 'M. Salah',     team: 'LIV', pos: 'MID', price: 13.4, ev: 7.8, own: 52.3, form: 8.4, fix: 'easy' },
    { name: 'E. Haaland',   team: 'MCI', pos: 'FWD', price: 15.0, ev: 8.2, own: 64.1, form: 9.1, fix: 'med'  },
    { name: 'C. Palmer',    team: 'CHE', pos: 'MID', price: 10.8, ev: 7.1, own: 41.7, form: 7.8, fix: 'easy' },
    { name: 'B. Saka',      team: 'ARS', pos: 'MID', price: 10.2, ev: 6.9, own: 35.4, form: 6.7, fix: 'easy' },
    { name: 'A. Isak',      team: 'NEW', pos: 'FWD', price:  8.7, ev: 6.4, own: 28.9, form: 7.2, fix: 'med'  },
    { name: 'O. Watkins',   team: 'AVL', pos: 'FWD', price:  9.2, ev: 6.0, own: 22.3, form: 6.4, fix: 'med'  },
    { name: 'P. Foden',     team: 'MCI', pos: 'MID', price:  9.4, ev: 5.8, own: 18.7, form: 5.9, fix: 'med'  },
    { name: 'B. Fernandes', team: 'MUN', pos: 'MID', price:  8.4, ev: 5.6, own: 14.2, form: 5.7, fix: 'hard' },
  ],
  differentials: [
    { name: 'D. Mukiele', team: 'NFO', pos: 'DEF', price: 4.6, ev: 4.8, own: 2.1 },
    { name: 'M. Kudus',   team: 'WHU', pos: 'MID', price: 6.5, ev: 5.2, own: 3.4 },
    { name: 'J. Pedro',   team: 'BHA', pos: 'FWD', price: 5.7, ev: 4.9, own: 4.2 },
    { name: 'K. Schade',  team: 'BRE', pos: 'MID', price: 5.4, ev: 4.6, own: 1.8 },
  ],
  gws: [
    { gw: 28, fix: 'LIV(H)', diff: 2, ev: 64.2, action: 'hold' },
    { gw: 29, fix: 'NEW(A)', diff: 4, ev: 58.1, action: '−1 transfer' },
    { gw: 30, fix: 'BHA(H)', diff: 2, ev: 71.4, action: 'wildcard' },
    { gw: 31, fix: 'CHE(A)', diff: 3, ev: 66.8, action: 'hold' },
    { gw: 32, fix: 'IPS(H)', diff: 1, ev: 78.2, action: 'triple cap' },
    { gw: 33, fix: 'MCI(A)', diff: 5, ev: 52.4, action: 'free hit' },
  ],
  squad: {
    gk:  [{ name: 'Raya', team: 'ARS', price: 5.5, ev: 5.1 }],
    def: [
      { name: 'Gabriel',   team: 'ARS', price: 6.2, ev: 5.4 },
      { name: 'Trippier',  team: 'NEW', price: 6.5, ev: 5.0 },
      { name: 'Robertson', team: 'LIV', price: 6.7, ev: 4.8 },
    ],
    mid: [
      { name: 'Salah',  team: 'LIV', price: 13.4, ev: 7.8 },
      { name: 'Palmer', team: 'CHE', price: 10.8, ev: 7.1 },
      { name: 'Saka',   team: 'ARS', price: 10.2, ev: 6.9 },
      { name: 'Foden',  team: 'MCI', price:  9.4, ev: 5.8 },
    ],
    fwd: [
      { name: 'Haaland', team: 'MCI', price: 15.0, ev: 8.2 },
      { name: 'Isak',    team: 'NEW', price:  8.7, ev: 6.4 },
      { name: 'Watkins', team: 'AVL', price:  9.2, ev: 6.0 },
    ],
  },
};

export const TEAM_COLORS = {
  ARS: '#EF0107', AVL: '#95BFE5', BHA: '#0057B8', BOU: '#DA291C', BRE: '#E30613',
  CHE: '#034694', CRY: '#1B458F', EVE: '#003399', FUL: '#000000', IPS: '#3764A8',
  LEI: '#003090', LIV: '#C8102E', MCI: '#6CABDD', MUN: '#DA291C', NEW: '#241F20',
  NFO: '#DD0000', SOU: '#D71920', TOT: '#132257', WHU: '#7A263A', WOL: '#FDB913',
};
