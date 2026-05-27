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
# price_tenths: price in $0.1m units  (e.g. 105 = $10.5m)
# Confirmed real prices: Mbappe/Kane/Haaland = $10.5m, Messi = $10.0m
SEED_PLAYERS = [
    # ── France ──────────────────────────────────────────────────────────────
    ("France",      "K. Mbappé",          "FWD", 105),
    ("France",      "O. Dembélé",         "FWD",  80),
    ("France",      "A. Griezmann",       "MID",  75),
    ("France",      "A. Tchouaméni",      "MID",  60),
    ("France",      "E. Camavinga",       "MID",  60),
    ("France",      "T. Hernández",       "DEF",  60),
    ("France",      "D. Upamecano",       "DEF",  55),
    ("France",      "B. Pavard",          "DEF",  50),
    ("France",      "M. Maignan",         "GK",   55),
    # ── Brazil ──────────────────────────────────────────────────────────────
    ("Brazil",      "Vinicius Jr",        "FWD", 105),
    ("Brazil",      "Rodrygo",            "FWD",  80),
    ("Brazil",      "Raphinha",           "MID",  85),
    ("Brazil",      "B. Guimarães",       "MID",  75),
    ("Brazil",      "L. Paquetá",         "MID",  65),
    ("Brazil",      "É. Militão",         "DEF",  60),
    ("Brazil",      "Danilo",             "DEF",  50),
    ("Brazil",      "Alisson",            "GK",   55),
    # ── England ─────────────────────────────────────────────────────────────
    ("England",     "J. Bellingham",      "MID", 100),
    ("England",     "H. Kane",            "FWD", 105),
    ("England",     "B. Saka",            "MID",  90),
    ("England",     "C. Palmer",          "MID",  80),
    ("England",     "P. Foden",           "MID",  80),
    ("England",     "T. Alexander-Arnold","DEF",  70),
    ("England",     "K. Trippier",        "DEF",  60),
    ("England",     "J. Guehi",           "DEF",  50),
    ("England",     "J. Pickford",        "GK",   50),
    # ── Argentina ───────────────────────────────────────────────────────────
    ("Argentina",   "L. Messi",           "FWD", 100),
    ("Argentina",   "L. Martínez",        "FWD",  85),
    ("Argentina",   "A. Mac Allister",    "MID",  80),
    ("Argentina",   "R. De Paul",         "MID",  70),
    ("Argentina",   "N. Molina",          "DEF",  60),
    ("Argentina",   "C. Romero",          "DEF",  65),
    ("Argentina",   "E. Martínez",        "GK",   60),
    # ── Spain ───────────────────────────────────────────────────────────────
    ("Spain",       "L. Yamal",           "MID", 100),
    ("Spain",       "D. Olmo",            "MID",  80),
    ("Spain",       "Pedri",              "MID",  85),
    ("Spain",       "A. Morata",          "FWD",  75),
    ("Spain",       "A. Grimaldo",        "DEF",  60),
    ("Spain",       "R. Carvajal",        "DEF",  55),
    ("Spain",       "Unai Simón",         "GK",   55),
    # ── Portugal ────────────────────────────────────────────────────────────
    ("Portugal",    "C. Ronaldo",         "FWD",  95),
    ("Portugal",    "R. Leão",            "FWD",  85),
    ("Portugal",    "B. Silva",           "MID",  90),
    ("Portugal",    "R. Neves",           "MID",  70),
    ("Portugal",    "J. Cancelo",         "DEF",  60),
    ("Portugal",    "R. Dias",            "DEF",  65),
    ("Portugal",    "Diogo Costa",        "GK",   55),
    # ── Germany ─────────────────────────────────────────────────────────────
    ("Germany",     "J. Musiala",         "MID",  95),
    ("Germany",     "F. Wirtz",           "MID",  90),
    ("Germany",     "K. Havertz",         "FWD",  85),
    ("Germany",     "J. Kimmich",         "MID",  75),
    ("Germany",     "A. Rüdiger",         "DEF",  60),
    ("Germany",     "D. Raum",            "DEF",  55),
    ("Germany",     "M. Neuer",           "GK",   50),
    # ── Netherlands ─────────────────────────────────────────────────────────
    ("Netherlands", "C. Gakpo",           "FWD",  85),
    ("Netherlands", "F. de Jong",         "MID",  80),
    ("Netherlands", "V. van Dijk",        "DEF",  75),
    ("Netherlands", "D. Dumfries",        "DEF",  60),
    ("Netherlands", "N. Madueke",         "MID",  70),
    ("Netherlands", "B. Flekken",         "GK",   50),
    # ── Belgium ─────────────────────────────────────────────────────────────
    ("Belgium",     "K. De Bruyne",       "MID",  95),
    ("Belgium",     "R. Lukaku",          "FWD",  90),
    ("Belgium",     "J. Doku",            "MID",  75),
    ("Belgium",     "T. Castagne",        "DEF",  55),
    ("Belgium",     "T. Courtois",        "GK",   60),
    ("Belgium",     "A. Theate",          "DEF",  55),
    # ── Croatia ─────────────────────────────────────────────────────────────
    ("Croatia",     "L. Modrić",          "MID",  80),
    ("Croatia",     "I. Kramarić",        "FWD",  75),
    ("Croatia",     "J. Gvardiol",        "DEF",  75),
    ("Croatia",     "I. Perišić",         "MID",  65),
    ("Croatia",     "D. Livaković",       "GK",   55),
    ("Croatia",     "J. Šutalo",          "DEF",  50),
    # ── Morocco ─────────────────────────────────────────────────────────────
    ("Morocco",     "A. Hakimi",          "DEF",  70),
    ("Morocco",     "H. Ziyech",          "MID",  70),
    ("Morocco",     "Y. En-Nesyri",       "FWD",  70),
    ("Morocco",     "A. Ounahi",          "MID",  60),
    ("Morocco",     "Y. Bono",            "GK",   55),
    ("Morocco",     "N. Aguerd",          "DEF",  55),
    # ── Norway ──────────────────────────────────────────────────────────────
    ("Norway",      "E. Haaland",         "FWD", 105),
    ("Norway",      "M. Ødegaard",        "MID",  90),
    ("Norway",      "A. Sørloth",         "FWD",  70),
    ("Norway",      "S. Berge",           "MID",  65),
    ("Norway",      "F. Aursnes",         "MID",  60),
    ("Norway",      "L. Østigård",        "DEF",  55),
    ("Norway",      "Ø. Nyland",          "GK",   50),
    # ── Uruguay ─────────────────────────────────────────────────────────────
    ("Uruguay",     "F. Valverde",        "MID",  85),
    ("Uruguay",     "D. Núñez",           "FWD",  90),
    ("Uruguay",     "R. Bentancur",       "MID",  65),
    ("Uruguay",     "R. Araújo",          "DEF",  65),
    ("Uruguay",     "J. Rochet",          "GK",   50),
    ("Uruguay",     "M. Olivera",         "DEF",  55),
    # ── Japan ───────────────────────────────────────────────────────────────
    ("Japan",       "T. Kubo",            "MID",  70),
    ("Japan",       "H. Maeda",           "FWD",  60),
    ("Japan",       "T. Minamino",        "MID",  60),
    ("Japan",       "H. Ito",             "DEF",  50),
    ("Japan",       "M. Tomiyasu",        "DEF",  55),
    ("Japan",       "S. Gonda",           "GK",   50),
    # ── United States ───────────────────────────────────────────────────────
    ("United States","C. Pulisic",        "MID",  80),
    ("United States","T. Adams",          "MID",  65),
    ("United States","G. Reyna",          "MID",  60),
    ("United States","Y. Musah",          "MID",  60),
    ("United States","A. Robinson",       "DEF",  55),
    ("United States","M. Turner",         "GK",   50),
    # ── Mexico ──────────────────────────────────────────────────────────────
    ("Mexico",      "H. Lozano",          "FWD",  70),
    ("Mexico",      "R. Jiménez",         "FWD",  65),
    ("Mexico",      "H. Moreno",          "DEF",  60),
    ("Mexico",      "E. Álvarez",         "MID",  55),
    ("Mexico",      "G. Ochoa",           "GK",   50),
    ("Mexico",      "J. Gallardo",        "DEF",  50),
    # ── Poland ──────────────────────────────────────────────────────────────
    ("Poland",      "R. Lewandowski",     "FWD",  90),
    ("Poland",      "P. Zieliński",       "MID",  65),
    ("Poland",      "S. Szymański",       "MID",  60),
    ("Poland",      "J. Bednarek",        "DEF",  50),
    ("Poland",      "W. Szczęsny",        "GK",   55),
    # ── Serbia ──────────────────────────────────────────────────────────────
    ("Serbia",      "D. Vlahović",        "FWD",  75),
    ("Serbia",      "A. Mitrović",        "FWD",  80),
    ("Serbia",      "D. Tadić",           "MID",  70),
    ("Serbia",      "N. Milenković",      "DEF",  50),
    ("Serbia",      "P. Rajković",        "GK",   50),
    # ── Denmark ─────────────────────────────────────────────────────────────
    ("Denmark",     "R. Højlund",         "FWD",  75),
    ("Denmark",     "C. Eriksen",         "MID",  70),
    ("Denmark",     "J. Mæhle",           "DEF",  55),
    ("Denmark",     "P. Højbjerg",        "MID",  60),
    ("Denmark",     "K. Schmeichel",      "GK",   50),
    # ── Colombia ────────────────────────────────────────────────────────────
    ("Colombia",    "L. Díaz",            "FWD",  80),
    ("Colombia",    "J. Lerma",           "MID",  60),
    ("Colombia",    "R. Quintero",        "MID",  65),
    ("Colombia",    "D. Sánchez",         "DEF",  55),
    ("Colombia",    "D. Ospina",          "GK",   50),
    # ── Ecuador ─────────────────────────────────────────────────────────────
    ("Ecuador",     "M. Caicedo",         "MID",  80),
    ("Ecuador",     "E. Valencia",        "FWD",  70),
    ("Ecuador",     "A. Preciado",        "DEF",  55),
    ("Ecuador",     "B. Méndez",          "MID",  55),
    ("Ecuador",     "A. Domínguez",       "GK",   50),
    # ── Senegal ─────────────────────────────────────────────────────────────
    ("Senegal",     "S. Mané",            "FWD",  85),
    ("Senegal",     "I. Gueye",           "MID",  65),
    ("Senegal",     "K. Koulibaly",       "DEF",  65),
    ("Senegal",     "A. Diallo",          "DEF",  55),
    ("Senegal",     "E. Mendy",           "GK",   55),
    # ── Ghana ───────────────────────────────────────────────────────────────
    ("Ghana",       "M. Kudus",           "MID",  75),
    ("Ghana",       "T. Partey",          "MID",  65),
    ("Ghana",       "J. Ayew",            "FWD",  60),
    ("Ghana",       "D. Amartey",         "DEF",  50),
    ("Ghana",       "L. Bati",            "GK",   50),
    # ── Switzerland ─────────────────────────────────────────────────────────
    ("Switzerland", "G. Xhaka",           "MID",  70),
    ("Switzerland", "X. Shaqiri",         "MID",  70),
    ("Switzerland", "B. Embolo",          "FWD",  65),
    ("Switzerland", "M. Akanji",          "DEF",  60),
    ("Switzerland", "Y. Sommer",          "GK",   55),
    # ── Australia ───────────────────────────────────────────────────────────
    ("Australia",   "M. Leckie",          "MID",  55),
    ("Australia",   "A. Hrustic",         "MID",  60),
    ("Australia",   "J. Irvine",          "MID",  55),
    ("Australia",   "H. Souttar",         "DEF",  50),
    ("Australia",   "M. Ryan",            "GK",   50),
    # ── South Korea ─────────────────────────────────────────────────────────
    ("South Korea", "Son Heung-min",      "FWD",  90),
    ("South Korea", "Lee Jae-sung",       "MID",  65),
    ("South Korea", "Kim Min-jae",        "DEF",  70),
    ("South Korea", "Hwang Hee-chan",     "FWD",  65),
    ("South Korea", "Kim Seung-gyu",      "GK",   50),
    # ── Iran ────────────────────────────────────────────────────────────────
    ("Iran",        "S. Azmoun",          "FWD",  70),
    ("Iran",        "M. Taremi",          "FWD",  65),
    ("Iran",        "A. Jahanbakhsh",     "MID",  60),
    ("Iran",        "R. Rezaeian",        "DEF",  50),
    ("Iran",        "A. Beiranvand",      "GK",   50),
    # ── Canada ──────────────────────────────────────────────────────────────
    ("Canada",      "A. Davies",          "DEF",  75),
    ("Canada",      "J. David",           "FWD",  80),
    ("Canada",      "C. Buchanan",        "DEF",  60),
    ("Canada",      "S. Larin",           "FWD",  60),
    ("Canada",      "M. Borjan",          "GK",   50),
    # ── Saudi Arabia ────────────────────────────────────────────────────────
    ("Saudi Arabia","S. Al-Dawsari",      "MID",  65),
    ("Saudi Arabia","F. Al-Buraikan",     "FWD",  60),
    ("Saudi Arabia","A. Al-Shahrani",     "DEF",  55),
    ("Saudi Arabia","M. Al-Owais",        "GK",   50),
    # ── Nigeria ─────────────────────────────────────────────────────────────
    ("Nigeria",     "V. Osimhen",         "FWD",  80),
    ("Nigeria",     "T. Lookman",         "MID",  70),
    ("Nigeria",     "F. Onyeka",          "MID",  55),
    ("Nigeria",     "W. Troost-Ekong",    "DEF",  55),
    ("Nigeria",     "S. Obi",             "GK",   50),
    # ── Tunisia ─────────────────────────────────────────────────────────────
    ("Tunisia",     "Y. Msakni",          "MID",  60),
    ("Tunisia",     "S. Jaziri",          "FWD",  55),
    ("Tunisia",     "D. Bronn",           "DEF",  50),
    ("Tunisia",     "A. Dahmen",          "GK",   50),
    # ── Qatar ───────────────────────────────────────────────────────────────
    ("Qatar",       "A. Afif",            "MID",  60),
    ("Qatar",       "A. Ali",             "FWD",  55),
    ("Qatar",       "B. Khoukhi",         "DEF",  50),
    ("Qatar",       "M. Barsham",         "GK",   50),
    # ── Cameroon ────────────────────────────────────────────────────────────
    ("Cameroon",    "A. Anguissa",        "MID",  70),
    ("Cameroon",    "V. Aboubakar",       "FWD",  65),
    ("Cameroon",    "C. Tolo",            "DEF",  50),
    ("Cameroon",    "A. Onana",           "GK",   60),
    # ── Wales ───────────────────────────────────────────────────────────────
    ("Wales",       "D. James",           "FWD",  65),
    ("Wales",       "K. Moore",           "FWD",  60),
    ("Wales",       "J. Allen",           "MID",  55),
    ("Wales",       "N. Williams",        "DEF",  55),
    ("Wales",       "W. Hennessey",       "GK",   50),
    # ── Sweden ──────────────────────────────────────────────────────────────
    ("Sweden",      "A. Isak",            "FWD",  80),
    ("Sweden",      "D. Kulusevski",      "MID",  75),
    ("Sweden",      "V. Lindelöf",        "DEF",  60),
    ("Sweden",      "R. Olsen",           "GK",   50),
    # ── Peru ────────────────────────────────────────────────────────────────
    ("Peru",        "A. Lapadula",        "FWD",  60),
    ("Peru",        "C. Cueva",           "MID",  55),
    ("Peru",        "L. Advíncula",       "DEF",  50),
    ("Peru",        "P. Gallese",         "GK",   50),
    # ── Iceland ─────────────────────────────────────────────────────────────
    ("Iceland",     "G. Sigurdsson",      "MID",  60),
    ("Iceland",     "A. Böðvarsson",      "FWD",  55),
    ("Iceland",     "J. Sigurdsson",      "DEF",  50),
    ("Iceland",     "H. Valdimarsson",    "GK",   50),
    # ── Panama ──────────────────────────────────────────────────────────────
    ("Panama",      "R. Torres",          "DEF",  55),
    ("Panama",      "A. Murillo",         "DEF",  50),
    ("Panama",      "G. Gómez",           "MID",  50),
    ("Panama",      "L. Mejía",           "GK",   50),
    # ── Costa Rica ──────────────────────────────────────────────────────────
    ("Costa Rica",  "K. Navas",           "GK",   60),
    ("Costa Rica",  "J. Campbell",        "FWD",  60),
    ("Costa Rica",  "O. Duarte",          "DEF",  50),
    ("Costa Rica",  "C. Borges",          "MID",  50),
    # ── Egypt ───────────────────────────────────────────────────────────────
    ("Egypt",       "M. Salah",           "FWD", 100),
    ("Egypt",       "T. Mohamed",         "MID",  60),
    ("Egypt",       "O. Kamal",           "DEF",  50),
    ("Egypt",       "M. El-Shenawy",      "GK",   50),
    # ── Russia ──────────────────────────────────────────────────────────────
    ("Russia",      "A. Golovin",         "MID",  60),
    ("Russia",      "F. Smolov",          "FWD",  55),
    ("Russia",      "Y. Zhirkov",         "DEF",  50),
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
