import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent.parent))
from wc.src.score_predictor import DCPredictor
import sqlite3

p = DCPredictor()
p.load()
conn = sqlite3.connect(str(__import__('pathlib').Path(__file__).resolve().parent.parent / 'data' / 'wc.db'))
rows = conn.execute(
    'SELECT f.kickoff_time, t1.name, t1.team_id, t2.name, t2.team_id, f.group_id, f.matchday '
    'FROM fixtures f '
    'JOIN teams t1 ON f.home_team_id=t1.team_id '
    'JOIN teams t2 ON f.away_team_id=t2.team_id '
    'ORDER BY f.kickoff_time'
).fetchall()
conn.close()

print(f"{'Date':<12} {'Match':<45} {'Pred':^5} {'H%':>5} {'D%':>5} {'A%':>5}")
print('-' * 80)
for kt, hname, hid, aname, aid, grp, matchday in rows:
    if grp not in ('G', 'H', 'I', 'J', 'K', 'L'):
        continue
    try:
        boost = 0.10 if matchday == 1 else 0.0
        r = p.predict(hid, aid, home_advantage=False, draw_boost=boost)
        ps = f"{round(r['home_xg'])}-{round(r['away_xg'])}"
        md_tag = f"MD{matchday}"
        print(f"{str(kt)[:10]:<12} [{grp}][{md_tag}] {hname} vs {aname:<22} {ps:^5} {r['win_pct']:>5.1f} {r['draw_pct']:>5.1f} {r['loss_pct']:>5.1f}")
    except Exception as e:
        print(f"{str(kt)[:10]:<12} [{grp}] {hname} vs {aname}  ERROR: {e}")
