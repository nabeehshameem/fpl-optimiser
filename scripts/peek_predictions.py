"""
peek_predictions.py
Show the latest predictions joined with player names for human inspection.
"""

import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "fpl.db"


def peek():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Find the latest predictions per (player, gameweek, model)
    cursor.execute("""
        SELECT
            p.web_name,
            t.short_name AS team,
            pos.position_name,
            pred.gameweek_id,
            pred.model_name,
            ROUND(pred.predicted_points, 2) AS pred_pts,
            ROUND(s.ep_next, 2) AS fpl_ep_next
        FROM predictions pred
        JOIN players p ON pred.player_id = p.player_id
        JOIN teams t ON p.team_id = t.team_id
        JOIN (
            SELECT 1 AS pos_id, 'GK'  AS position_name UNION ALL
            SELECT 2,           'DEF'                   UNION ALL
            SELECT 3,           'MID'                   UNION ALL
            SELECT 4,           'FWD'
        ) pos ON p.position = pos.pos_id
        LEFT JOIN player_snapshots s
            ON s.player_id = pred.player_id
            AND s.gameweek_id = pred.gameweek_id
            AND s.snapshot_id = (
                SELECT MAX(snapshot_id)
                FROM player_snapshots
                WHERE player_id = pred.player_id AND gameweek_id = pred.gameweek_id
            )
        WHERE pred.prediction_id IN (
            SELECT MAX(prediction_id)
            FROM predictions
            GROUP BY player_id, gameweek_id, model_name
        )
        ORDER BY pred.predicted_points DESC
        LIMIT 25
    """)
    rows = cursor.fetchall()

    print(f"Top 25 predictions (latest run, per player/model):\n")
    print(f"{'Player':22s} {'Team':4s} {'Pos':4s} {'GW':>3s} {'Model':12s} {'Pred':>6s} {'FPL EP':>7s}")
    print("-" * 70)
    for row in rows:
        web_name, team, pos, gw, model, pred_pts, ep_next = row
        ep_str = f"{ep_next:6.2f}" if ep_next is not None else "  n/a"
        print(f"{web_name:22s} {team:4s} {pos:4s} {gw:>3d} {model:12s} {pred_pts:6.2f} {ep_str}")

    conn.close()


if __name__ == "__main__":
    peek()