import json, sqlite3, sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent.parent))
from wc.src.score_predictor import _canonical

conn = sqlite3.connect('wc/data/wc.db')
d = json.load(open('wc/models/dc_params.json'))

print("=== Iran recent_results ===")
iran_rows = conn.execute(
    "SELECT match_date, home_team, away_team, home_score, away_score, tournament FROM recent_results "
    "WHERE home_team LIKE '%Iran%' OR away_team LIKE '%Iran%' ORDER BY match_date DESC LIMIT 15"
).fetchall()
for r in iran_rows:
    print(" ", r)

print("\n=== France recent_results (from Mar 2026) ===")
france_rows = conn.execute(
    "SELECT match_date, home_team, away_team, home_score, away_score, tournament FROM recent_results "
    "WHERE (home_team LIKE '%France%' OR away_team LIKE '%France%') AND match_date >= '2026-03-01' "
    "ORDER BY match_date DESC LIMIT 15"
).fetchall()
for r in france_rows:
    print(" ", r)

print("\n=== New Zealand recent_results ===")
nz_rows = conn.execute(
    "SELECT match_date, home_team, away_team, home_score, away_score, tournament FROM recent_results "
    "WHERE home_team LIKE '%New Zealand%' OR away_team LIKE '%New Zealand%' ORDER BY match_date DESC LIMIT 10"
).fetchall()
for r in nz_rows:
    print(" ", r)

print("\n=== Portugal recent_results (recent) ===")
por_rows = conn.execute(
    "SELECT match_date, home_team, away_team, home_score, away_score, tournament FROM recent_results "
    "WHERE home_team LIKE '%Portugal%' OR away_team LIKE '%Portugal%' ORDER BY match_date DESC LIMIT 10"
).fetchall()
for r in por_rows:
    print(" ", r)

print("\n=== Norway recent_results ===")
nor_rows = conn.execute(
    "SELECT match_date, home_team, away_team, home_score, away_score, tournament FROM recent_results "
    "WHERE home_team LIKE '%Norway%' OR away_team LIKE '%Norway%' ORDER BY match_date DESC LIMIT 10"
).fetchall()
for r in nor_rows:
    print(" ", r)

conn.close()

# Also show effective stats for key matches
print("\n=== Effective stats (DC params * form adjustments) ===")
fa = d.get('form_adjustments', {})
teams_data = [
    ('ir iran', 'Iran'),
    ('new zealand', 'New Zealand'),
    ('france', 'France'),
    ('norway', 'Norway'),
    ('portugal', 'Portugal'),
    ('argentina', 'Argentina'),
    ('algeria', 'Algeria'),
]
print(f"{'Team':<18} {'DC_atk':>7} {'DC_def':>7} {'f_atk':>6} {'f_def':>6} {'EFF_atk':>8} {'EFF_def':>8}")
print('-' * 65)
for key, label in teams_data:
    tp = d['team_params'].get(key, {})
    atk = tp.get('attack', 0)
    dff = tp.get('defense', 0)
    faj = fa.get(key, [1.0, 1.0])
    eff_atk = atk * faj[0]
    eff_def = dff * faj[1]
    print(f"{label:<18} {atk:>7.3f} {dff:>7.3f} {faj[0]:>6.3f} {faj[1]:>6.3f} {eff_atk:>8.3f} {eff_def:>8.3f}")
