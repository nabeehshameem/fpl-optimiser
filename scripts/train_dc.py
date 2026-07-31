"""Train and save the FPL Dixon-Coles model on PL 2025/26 data."""
import sys
sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from src.dc_model import FPLDCPredictor

ap = argparse.ArgumentParser()
ap.add_argument("--db", type=Path, default=None,
                help="fit on this DB instead of data/fpl.db "
                     "(e.g. --db data/fpl_2526.db after rollover)")
args = ap.parse_args()

src = args.db or "data/fpl.db (default)"
print(f"Fitting FPL Dixon-Coles model on {src} ...")
p = FPLDCPredictor(db_path=args.db)
diag = p.fit()

print(f"\nFit complete:")
print(f"  Teams:    {diag['n_teams']}")
print(f"  Fixtures: {diag['n_fixtures']}")
print(f"  Home adv: {diag['home_adv']:.4f}  ({(diag['home_adv']-1)*100:.1f}%)")
print(f"  Rho:      {diag['rho']:.4f}")
print(f"  Converged: {diag['converged']} — {diag['message']}")

print("\nTop 10 teams by attack strength:")
for t in p.team_strengths()[:10]:
    form = f"form_atk={t['form_atk']:.2f} form_def={t['form_def']:.2f}"
    print(f"  {t['short_name']:<6} atk={t['attack']:.3f}  def={t['defense']:.3f}  {form}")

# Quick sanity check — last season's top fixtures (pass short_name directly)
print("\nSanity check predictions (a few top-team fixtures):")
strengths = p.team_strengths()
top4 = [t["short_name"] for t in strengths[:4]]
for i in range(min(3, len(top4))):
    h, a = top4[i], top4[i + 1]
    r = p.predict(h, a)
    print(f"  {r['home_name']} vs {r['away_name']}: "
          f"xG {r['home_xg']:.2f}-{r['away_xg']:.2f}  "
          f"W/D/L {r['win_pct']:.0f}/{r['draw_pct']:.0f}/{r['loss_pct']:.0f}  "
          f"CS: home={r['home_cs_pct']:.0f}% away={r['away_cs_pct']:.0f}%")

p.save()
