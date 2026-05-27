"""
seed_fantasy_players.py
Seeds WC2026 fantasy player estimates into the fantasy_players table.

This is placeholder data based on known squad members and estimated prices.
Replace with real FIFA data once available:
  python wc/scripts/ingest_fantasy_players.py --json players.json

To get real FIFA data: open play.fifa.com/fantasy, open DevTools → Network,
filter by XHR/Fetch, look for the bootstrap/players request, copy the JSON.
"""

import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from wc.scripts.init_db import DB_PATH

# (team_name, display_name, position, price_tenths)
# price_tenths: price in £0.1m units  (e.g. 120 = £12.0m)
SEED_PLAYERS = [
    # ── France ──────────────────────────────────────────────────────────────
    ("France",      "K. Mbappé",          "FWD", 120),
    ("France",      "O. Dembélé",         "FWD",  85),
    ("France",      "A. Griezmann",       "MID",  80),
    ("France",      "A. Tchouaméni",      "MID",  65),
    ("France",      "E. Camavinga",       "MID",  65),
    ("France",      "T. Hernández",       "DEF",  65),
    ("France",      "D. Upamecano",       "DEF",  60),
    ("France",      "B. Pavard",          "DEF",  55),
    ("France",      "M. Maignan",         "GK",   55),
    # ── Brazil ──────────────────────────────────────────────────────────────
    ("Brazil",      "Vinicius Jr",        "FWD", 120),
    ("Brazil",      "Rodrygo",            "FWD",  80),
    ("Brazil",      "Raphinha",           "MID",  85),
    ("Brazil",      "B. Guimarães",       "MID",  75),
    ("Brazil",      "L. Paquetá",         "MID",  70),
    ("Brazil",      "É. Militão",         "DEF",  65),
    ("Brazil",      "Danilo",             "DEF",  55),
    ("Brazil",      "Alisson",            "GK",   55),
    # ── England ─────────────────────────────────────────────────────────────
    ("England",     "J. Bellingham",      "MID", 110),
    ("England",     "H. Kane",            "FWD", 110),
    ("England",     "B. Saka",            "MID",  95),
    ("England",     "C. Palmer",          "MID",  85),
    ("England",     "P. Foden",           "MID",  85),
    ("England",     "T. Alexander-Arnold","DEF",  75),
    ("England",     "K. Trippier",        "DEF",  65),
    ("England",     "J. Guehi",           "DEF",  55),
    ("England",     "J. Pickford",        "GK",   50),
    # ── Argentina ───────────────────────────────────────────────────────────
    ("Argentina",   "L. Messi",           "FWD", 120),
    ("Argentina",   "L. Martínez",        "FWD",  90),
    ("Argentina",   "A. Mac Allister",    "MID",  85),
    ("Argentina",   "R. De Paul",         "MID",  75),
    ("Argentina",   "N. Molina",          "DEF",  65),
    ("Argentina",   "C. Romero",          "DEF",  70),
    ("Argentina",   "E. Martínez",        "GK",   60),
    # ── Spain ───────────────────────────────────────────────────────────────
    ("Spain",       "L. Yamal",           "MID", 105),
    ("Spain",       "D. Olmo",            "MID",  85),
    ("Spain",       "Pedri",              "MID",  90),
    ("Spain",       "A. Morata",          "FWD",  80),
    ("Spain",       "A. Grimaldo",        "DEF",  65),
    ("Spain",       "R. Carvajal",        "DEF",  60),
    ("Spain",       "Unai Simón",         "GK",   55),
    # ── Portugal ────────────────────────────────────────────────────────────
    ("Portugal",    "C. Ronaldo",         "FWD", 110),
    ("Portugal",    "R. Leão",            "FWD",  90),
    ("Portugal",    "B. Silva",           "MID",  95),
    ("Portugal",    "R. Neves",           "MID",  75),
    ("Portugal",    "J. Cancelo",         "DEF",  65),
    ("Portugal",    "R. Dias",            "DEF",  70),
    ("Portugal",    "Diogo Costa",        "GK",   55),
    # ── Germany ─────────────────────────────────────────────────────────────
    ("Germany",     "J. Musiala",         "MID", 100),
    ("Germany",     "F. Wirtz",           "MID",  95),
    ("Germany",     "K. Havertz",         "FWD",  90),
    ("Germany",     "J. Kimmich",         "MID",  80),
    ("Germany",     "A. Rüdiger",         "DEF",  65),
    ("Germany",     "D. Raum",            "DEF",  60),
    ("Germany",     "M. Neuer",           "GK",   55),
    # ── Netherlands ─────────────────────────────────────────────────────────
    ("Netherlands", "C. Gakpo",           "FWD",  90),
    ("Netherlands", "F. de Jong",         "MID",  85),
    ("Netherlands", "V. van Dijk",        "DEF",  80),
    ("Netherlands", "D. Dumfries",        "DEF",  65),
    ("Netherlands", "N. Madueke",         "MID",  75),
    ("Netherlands", "B. Flekken",         "GK",   50),
    # ── Belgium ─────────────────────────────────────────────────────────────
    ("Belgium",     "K. De Bruyne",       "MID", 105),
    ("Belgium",     "R. Lukaku",          "FWD", 100),
    ("Belgium",     "J. Doku",            "MID",  80),
    ("Belgium",     "T. Castagne",        "DEF",  60),
    ("Belgium",     "T. Courtois",        "GK",   60),
    ("Belgium",     "A. Theate",          "DEF",  60),
    # ── Croatia ─────────────────────────────────────────────────────────────
    ("Croatia",     "L. Modrić",          "MID",  85),
    ("Croatia",     "I. Kramarić",        "FWD",  80),
    ("Croatia",     "J. Gvardiol",        "DEF",  80),
    ("Croatia",     "I. Perišić",         "MID",  70),
    ("Croatia",     "D. Livaković",       "GK",   55),
    ("Croatia",     "J. Šutalo",          "DEF",  55),
    # ── Morocco ─────────────────────────────────────────────────────────────
    ("Morocco",     "A. Hakimi",          "DEF",  75),
    ("Morocco",     "H. Ziyech",          "MID",  75),
    ("Morocco",     "Y. En-Nesyri",       "FWD",  75),
    ("Morocco",     "A. Ounahi",          "MID",  65),
    ("Morocco",     "Y. Bono",            "GK",   55),
    ("Morocco",     "N. Aguerd",          "DEF",  60),
    # ── Uruguay ─────────────────────────────────────────────────────────────
    ("Uruguay",     "F. Valverde",        "MID",  90),
    ("Uruguay",     "D. Núñez",           "FWD",  95),
    ("Uruguay",     "R. Bentancur",       "MID",  70),
    ("Uruguay",     "R. Araújo",          "DEF",  70),
    ("Uruguay",     "J. Rochet",          "GK",   50),
    ("Uruguay",     "M. Olivera",         "DEF",  60),
    # ── Japan ───────────────────────────────────────────────────────────────
    ("Japan",       "T. Kubo",            "MID",  75),
    ("Japan",       "H. Maeda",           "FWD",  65),
    ("Japan",       "T. Minamino",        "MID",  65),
    ("Japan",       "H. Ito",             "DEF",  55),
    ("Japan",       "M. Tomiyasu",        "DEF",  60),
    ("Japan",       "S. Gonda",           "GK",   50),
    # ── United States ───────────────────────────────────────────────────────
    ("United States","C. Pulisic",        "MID",  85),
    ("United States","T. Adams",          "MID",  70),
    ("United States","G. Reyna",          "MID",  65),
    ("United States","Y. Musah",          "MID",  65),
    ("United States","A. Robinson",       "DEF",  60),
    ("United States","M. Turner",         "GK",   50),
    # ── Mexico ──────────────────────────────────────────────────────────────
    ("Mexico",      "H. Lozano",          "FWD",  75),
    ("Mexico",      "R. Jiménez",         "FWD",  70),
    ("Mexico",      "H. Moreno",          "DEF",  65),
    ("Mexico",      "E. Álvarez",         "MID",  60),
    ("Mexico",      "G. Ochoa",           "GK",   55),
    ("Mexico",      "J. Gallardo",        "DEF",  55),
    # ── Poland ──────────────────────────────────────────────────────────────
    ("Poland",      "R. Lewandowski",     "FWD",  95),
    ("Poland",      "P. Zieliński",       "MID",  70),
    ("Poland",      "S. Szymański",       "MID",  65),
    ("Poland",      "J. Bednarek",        "DEF",  55),
    ("Poland",      "W. Szczęsny",        "GK",   55),
    # ── Serbia ──────────────────────────────────────────────────────────────
    ("Serbia",      "D. Vlahović",        "FWD",  80),
    ("Serbia",      "A. Mitrović",        "FWD",  85),
    ("Serbia",      "D. Tadić",           "MID",  75),
    ("Serbia",      "N. Milenković",      "DEF",  55),
    ("Serbia",      "P. Rajković",        "GK",   50),
    # ── Denmark ─────────────────────────────────────────────────────────────
    ("Denmark",     "R. Højlund",         "FWD",  80),
    ("Denmark",     "C. Eriksen",         "MID",  75),
    ("Denmark",     "J. Mæhle",           "DEF",  60),
    ("Denmark",     "P. Højbjerg",        "MID",  65),
    ("Denmark",     "K. Schmeichel",      "GK",   50),
    # ── Colombia ────────────────────────────────────────────────────────────
    ("Colombia",    "L. Díaz",            "FWD",  85),
    ("Colombia",    "J. Lerma",           "MID",  65),
    ("Colombia",    "R. Quintero",        "MID",  70),
    ("Colombia",    "D. Sánchez",         "DEF",  60),
    ("Colombia",    "D. Ospina",          "GK",   55),
    # ── Ecuador ─────────────────────────────────────────────────────────────
    ("Ecuador",     "M. Caicedo",         "MID",  85),
    ("Ecuador",     "E. Valencia",        "FWD",  75),
    ("Ecuador",     "A. Preciado",        "DEF",  60),
    ("Ecuador",     "B. Méndez",          "MID",  60),
    ("Ecuador",     "A. Domínguez",       "GK",   50),
    # ── Senegal ─────────────────────────────────────────────────────────────
    ("Senegal",     "S. Mané",            "FWD",  90),
    ("Senegal",     "I. Gueye",           "MID",  70),
    ("Senegal",     "K. Koulibaly",       "DEF",  70),
    ("Senegal",     "A. Diallo",          "DEF",  60),
    ("Senegal",     "E. Mendy",           "GK",   55),
    # ── Ghana ───────────────────────────────────────────────────────────────
    ("Ghana",       "M. Kudus",           "MID",  80),
    ("Ghana",       "T. Partey",          "MID",  70),
    ("Ghana",       "J. Ayew",            "FWD",  65),
    ("Ghana",       "D. Amartey",         "DEF",  55),
    ("Ghana",       "L. Bati",            "GK",   50),
    # ── Switzerland ─────────────────────────────────────────────────────────
    ("Switzerland", "G. Xhaka",           "MID",  75),
    ("Switzerland", "X. Shaqiri",         "MID",  75),
    ("Switzerland", "B. Embolo",          "FWD",  70),
    ("Switzerland", "M. Akanji",          "DEF",  65),
    ("Switzerland", "Y. Sommer",          "GK",   55),
    # ── Australia ───────────────────────────────────────────────────────────
    ("Australia",   "M. Leckie",          "MID",  60),
    ("Australia",   "A. Hrustic",         "MID",  65),
    ("Australia",   "J. Irvine",          "MID",  60),
    ("Australia",   "H. Souttar",         "DEF",  55),
    ("Australia",   "M. Ryan",            "GK",   50),
    # ── South Korea ─────────────────────────────────────────────────────────
    ("South Korea", "Son Heung-min",      "FWD",  95),
    ("South Korea", "Lee Jae-sung",       "MID",  70),
    ("South Korea", "Kim Min-jae",        "DEF",  75),
    ("South Korea", "Hwang Hee-chan",     "FWD",  70),
    ("South Korea", "Kim Seung-gyu",      "GK",   50),
    # ── Iran ────────────────────────────────────────────────────────────────
    ("Iran",        "S. Azmoun",          "FWD",  75),
    ("Iran",        "M. Taremi",          "FWD",  70),
    ("Iran",        "A. Jahanbakhsh",     "MID",  65),
    ("Iran",        "R. Rezaeian",        "DEF",  55),
    ("Iran",        "A. Beiranvand",      "GK",   50),
    # ── Canada ──────────────────────────────────────────────────────────────
    ("Canada",      "A. Davies",          "DEF",  80),
    ("Canada",      "J. David",           "FWD",  85),
    ("Canada",      "C. Buchanan",        "DEF",  65),
    ("Canada",      "S. Larin",           "FWD",  65),
    ("Canada",      "M. Borjan",          "GK",   50),
    # ── Saudi Arabia ────────────────────────────────────────────────────────
    ("Saudi Arabia","S. Al-Dawsari",      "MID",  70),
    ("Saudi Arabia","F. Al-Buraikan",     "FWD",  65),
    ("Saudi Arabia","A. Al-Shahrani",     "DEF",  60),
    ("Saudi Arabia","M. Al-Owais",        "GK",   50),
    # ── Nigeria ─────────────────────────────────────────────────────────────
    ("Nigeria",     "V. Osimhen",         "FWD",  80),
    ("Nigeria",     "T. Lookman",         "MID",  75),
    ("Nigeria",     "F. Onyeka",          "MID",  60),
    ("Nigeria",     "W. Troost-Ekong",    "DEF",  60),
    ("Nigeria",     "S. Obi",             "GK",   50),
    # ── Tunisia ─────────────────────────────────────────────────────────────
    ("Tunisia",     "Y. Msakni",          "MID",  65),
    ("Tunisia",     "S. Jaziri",          "FWD",  60),
    ("Tunisia",     "D. Bronn",           "DEF",  55),
    ("Tunisia",     "A. Dahmen",          "GK",   50),
    # ── Qatar ───────────────────────────────────────────────────────────────
    ("Qatar",       "A. Afif",            "MID",  65),
    ("Qatar",       "A. Ali",             "FWD",  60),
    ("Qatar",       "B. Khoukhi",         "DEF",  55),
    ("Qatar",       "M. Barsham",         "GK",   50),
    # ── Cameroon ────────────────────────────────────────────────────────────
    ("Cameroon",    "A. Anguissa",        "MID",  75),
    ("Cameroon",    "V. Aboubakar",       "FWD",  70),
    ("Cameroon",    "C. Tolo",            "DEF",  55),
    ("Cameroon",    "A. Onana",           "GK",   60),
    # ── Wales ───────────────────────────────────────────────────────────────
    ("Wales",       "D. James",           "FWD",  70),
    ("Wales",       "K. Moore",           "FWD",  65),
    ("Wales",       "J. Allen",           "MID",  60),
    ("Wales",       "N. Williams",        "DEF",  60),
    ("Wales",       "W. Hennessey",       "GK",   50),
    # ── Sweden ──────────────────────────────────────────────────────────────
    ("Sweden",      "A. Isak",            "FWD",  85),
    ("Sweden",      "D. Kulusevski",      "MID",  80),
    ("Sweden",      "V. Lindelöf",        "DEF",  65),
    ("Sweden",      "R. Olsen",           "GK",   50),
    # ── Peru ────────────────────────────────────────────────────────────────
    ("Peru",        "A. Lapadula",        "FWD",  65),
    ("Peru",        "C. Cueva",           "MID",  60),
    ("Peru",        "L. Advíncula",       "DEF",  55),
    ("Peru",        "P. Gallese",         "GK",   50),
    # ── Iceland ─────────────────────────────────────────────────────────────
    ("Iceland",     "G. Sigurdsson",      "MID",  65),
    ("Iceland",     "A. Böðvarsson",      "FWD",  60),
    ("Iceland",     "J. Sigurdsson",      "DEF",  55),
    ("Iceland",     "H. Valdimarsson",    "GK",   50),
    # ── Panama ──────────────────────────────────────────────────────────────
    ("Panama",      "R. Torres",          "DEF",  60),
    ("Panama",      "A. Murillo",         "DEF",  55),
    ("Panama",      "G. Gómez",           "MID",  55),
    ("Panama",      "L. Mejía",           "GK",   50),
    # ── Costa Rica ──────────────────────────────────────────────────────────
    ("Costa Rica",  "K. Navas",           "GK",   60),
    ("Costa Rica",  "J. Campbell",        "FWD",  65),
    ("Costa Rica",  "O. Duarte",          "DEF",  55),
    ("Costa Rica",  "C. Borges",          "MID",  55),
    # ── Egypt ───────────────────────────────────────────────────────────────
    ("Egypt",       "M. Salah",           "FWD", 110),
    ("Egypt",       "T. Mohamed",         "MID",  65),
    ("Egypt",       "O. Kamal",           "DEF",  55),
    ("Egypt",       "M. El-Shenawy",      "GK",   50),
    # ── Russia ──────────────────────────────────────────────────────────────
    ("Russia",      "A. Golovin",         "MID",  65),
    ("Russia",      "F. Smolov",          "FWD",  60),
    ("Russia",      "Y. Zhirkov",         "DEF",  55),
    ("Russia",      "M. Safonov",         "GK",   50),
]


def seed(overwrite: bool = False) -> int:
    conn  = sqlite3.connect(DB_PATH)
    count = conn.execute("SELECT COUNT(*) FROM fantasy_players").fetchone()[0]

    if count > 0 and not overwrite:
        print(f"  {count} fantasy players already seeded. Use --overwrite to replace.")
        conn.close()
        return count

    if overwrite:
        conn.execute("DELETE FROM fantasy_players")

    team_map: dict[str, int] = {}
    for row in conn.execute("SELECT team_id, name FROM teams"):
        team_map[row[1].lower()] = row[0]

    inserted, unmatched = 0, set()
    for i, (team, name, pos, price) in enumerate(SEED_PLAYERS):
        team_id = team_map.get(team.lower())
        if team_id is None:
            for db_team, tid in team_map.items():
                if team.lower() in db_team or db_team in team.lower():
                    team_id = tid
                    break
            if team_id is None:
                unmatched.add(team)

        conn.execute(
            """
            INSERT OR REPLACE INTO fantasy_players
              (fantasy_id, player_id, name, name_norm, team_id, position, price, ownership)
            VALUES (?, NULL, ?, ?, ?, ?, ?, 0.0)
            """,
            (i + 1, name, name.lower().strip(), team_id, pos, price),
        )
        inserted += 1

    conn.commit()
    conn.close()

    if unmatched:
        print(f"  Unmatched teams (stored without team_id): {unmatched}")
    print(f"  {inserted} fantasy players seeded.")
    return inserted


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--overwrite", action="store_true", help="Replace existing seed data")
    args = p.parse_args()
    seed(overwrite=args.overwrite)
