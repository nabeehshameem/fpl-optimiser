"""Train and save the FPL Dixon-Coles model on PL 2025/26 data."""
import sys
sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dc_model import FPLDCPredictor

print("Fitting FPL Dixon-Coles model on PL 2025/26 data...")
p = FPLDCPredictor()
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
    print(f"  {t['name']:<25} atk={t['attack']:.3f}  def={t['defense']:.3f}  {form}")

# Quick sanity check — last season's top fixtures
print("\nSanity check predictions (a few top-team fixtures):")
strengths = p.team_strengths()
top4_ids = [t["team_id"] for t in strengths[:4]]
for i in range(min(3, len(top4_ids))):
    h, a = top4_ids[i], top4_ids[i + 1]
    r = p.predict(h, a)
    print(f"  {r['home_name']} vs {r['away_name']}: "
          f"xG {r['home_xg']:.2f}-{r['away_xg']:.2f}  "
          f"W/D/L {r['win_pct']:.0f}/{r['draw_pct']:.0f}/{r['loss_pct']:.0f}  "
          f"CS: home={r['home_cs_pct']:.0f}% away={r['away_cs_pct']:.0f}%")

p.save()
