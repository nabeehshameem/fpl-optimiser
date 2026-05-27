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
# Real prices from FIFA WC Fantasy app (FWD/MID confirmed; DEF/GK estimated)
SEED_PLAYERS = [
    # ── France ──────────────────────────────────────────────────────────────
    ("France",      "K. Mbappé",          "FWD", 105),
    ("France",      "O. Dembélé",         "FWD", 100),
    ("France",      "M. Olise",           "FWD",  95),
    ("France",      "B. Barcola",         "FWD",  80),
    ("France",      "R. Cherki",          "MID",  80),
    ("France",      "M. Thuram",          "FWD",  75),
    ("France",      "D. Doué",            "MID",  75),
    ("France",      "Y. Fofana",          "MID",  68),
    ("France",      "J. Mateta",          "FWD",  65),
    ("France",      "A. Tchouaméni",      "MID",  65),
    ("France",      "T. Hernández",       "DEF",  60),
    ("France",      "D. Upamecano",       "DEF",  55),
    ("France",      "M. Maignan",         "GK",   55),
    # ── Brazil ──────────────────────────────────────────────────────────────
    ("Brazil",      "Vinicius Jr",        "FWD", 100),
    ("Brazil",      "Raphinha",           "MID",  82),
    ("Brazil",      "M. Cunha",           "FWD",  73),
    ("Brazil",      "Neymar",             "FWD",  72),
    ("Brazil",      "B. Guimarães",       "MID",  68),
    ("Brazil",      "G. Martinelli",      "FWD",  65),
    ("Brazil",      "L. Paquetá",         "MID",  65),
    ("Brazil",      "É. Militão",         "DEF",  58),
    ("Brazil",      "Alisson",            "GK",   55),
    # ── England ─────────────────────────────────────────────────────────────
    ("England",     "H. Kane",            "FWD", 105),
    ("England",     "B. Saka",            "MID",  95),
    ("England",     "J. Bellingham",      "MID",  83),
    ("England",     "P. Foden",           "MID",  80),
    ("England",     "E. Eze",             "MID",  80),
    ("England",     "O. Watkins",         "FWD",  79),
    ("England",     "M. Rashford",        "FWD",  75),
    ("England",     "I. Toney",           "FWD",  75),
    ("England",     "M. Rogers",          "MID",  72),
    ("England",     "A. Gordon",          "MID",  70),
    ("England",     "D. Rice",            "MID",  70),
    ("England",     "C. Palmer",          "MID",  80),
    ("England",     "T. Alexander-Arnold","DEF",  68),
    ("England",     "J. Pickford",        "GK",   52),
    # ── Argentina ───────────────────────────────────────────────────────────
    ("Argentina",   "L. Messi",           "FWD", 100),
    ("Argentina",   "L. Martínez",        "FWD",  88),
    ("Argentina",   "J. Álvarez",         "FWD",  86),
    ("Argentina",   "E. Fernández",       "MID",  75),
    ("Argentina",   "A. Mac Allister",    "MID",  66),
    ("Argentina",   "E. Buendía",         "MID",  65),
    ("Argentina",   "R. De Paul",         "MID",  65),
    ("Argentina",   "C. Romero",          "DEF",  62),
    ("Argentina",   "E. Martínez",        "GK",   58),
    # ── Spain ───────────────────────────────────────────────────────────────
    ("Spain",       "L. Yamal",           "MID", 100),
    ("Spain",       "Pedri",              "MID",  81),
    ("Spain",       "M. Oyarzabal",       "FWD",  81),
    ("Spain",       "D. Olmo",            "MID",  77),
    ("Spain",       "F. Torres",          "FWD",  78),
    ("Spain",       "N. Williams",        "MID",  78),
    ("Spain",       "Rodri",              "MID",  75),
    ("Spain",       "Gavi",               "MID",  65),
    ("Spain",       "R. Carvajal",        "DEF",  55),
    ("Spain",       "Unai Simón",         "GK",   55),
    # ── Portugal ────────────────────────────────────────────────────────────
    ("Portugal",    "C. Ronaldo",         "FWD", 100),
    ("Portugal",    "B. Fernandes",       "MID",  85),
    ("Portugal",    "B. Silva",           "MID",  78),
    ("Portugal",    "R. Leão",            "FWD",  78),
    ("Portugal",    "G. Ramos",           "FWD",  75),
    ("Portugal",    "J. Félix",           "FWD",  65),
    ("Portugal",    "J. Neves",           "MID",  65),
    ("Portugal",    "R. Dias",            "DEF",  62),
    ("Portugal",    "Diogo Costa",        "GK",   55),
    # ── Germany ─────────────────────────────────────────────────────────────
    ("Germany",     "J. Musiala",         "MID",  80),
    ("Germany",     "F. Wirtz",           "MID",  75),
    ("Germany",     "K. Havertz",         "FWD",  78),
    ("Germany",     "L. Sané",            "MID",  74),
    ("Germany",     "N. Woltemade",       "FWD",  72),
    ("Germany",     "J. Kimmich",         "MID",  75),
    ("Germany",     "D. Undav",           "FWD",  66),
    ("Germany",     "M. Beier",           "FWD",  65),
    ("Germany",     "A. Rüdiger",         "DEF",  58),
    ("Germany",     "M. Neuer",           "GK",   52),
    # ── Netherlands ─────────────────────────────────────────────────────────
    ("Netherlands", "C. Gakpo",           "FWD",  77),
    ("Netherlands", "X. Simons",          "MID",  65),
    ("Netherlands", "T. Reijnders",       "MID",  65),
    ("Netherlands", "F. de Jong",         "MID",  80),
    ("Netherlands", "N. Madueke",         "MID",  70),
    ("Netherlands", "V. van Dijk",        "DEF",  72),
    ("Netherlands", "B. Flekken",         "GK",   50),
    # ── Belgium ─────────────────────────────────────────────────────────────
    ("Belgium",     "K. De Bruyne",       "MID",  75),
    ("Belgium",     "R. Lukaku",          "FWD",  74),
    ("Belgium",     "J. Doku",            "MID",  75),
    ("Belgium",     "L. Trossard",        "MID",  66),
    ("Belgium",     "T. Courtois",        "GK",   60),
    ("Belgium",     "T. Castagne",        "DEF",  55),
    # ── Croatia ─────────────────────────────────────────────────────────────
    ("Croatia",     "A. Budimir",         "FWD",  68),
    ("Croatia",     "L. Modrić",          "MID",  75),
    ("Croatia",     "M. Baturina",        "MID",  65),
    ("Croatia",     "I. Perišić",         "MID",  65),
    ("Croatia",     "J. Gvardiol",        "DEF",  72),
    ("Croatia",     "D. Livaković",       "GK",   55),
    # ── Morocco ─────────────────────────────────────────────────────────────
    ("Morocco",     "I. Saibari",         "MID",  68),
    ("Morocco",     "Y. En-Nesyri",       "FWD",  70),
    ("Morocco",     "H. Ziyech",          "MID",  68),
    ("Morocco",     "A. Hakimi",          "DEF",  70),
    ("Morocco",     "Y. Bono",            "GK",   55),
    # ── Norway ──────────────────────────────────────────────────────────────
    ("Norway",      "E. Haaland",         "FWD", 105),
    ("Norway",      "M. Ødegaard",        "MID",  77),
    ("Norway",      "A. Sørloth",         "FWD",  68),
    ("Norway",      "F. Aursnes",         "MID",  65),
    ("Norway",      "S. Berge",           "MID",  62),
    ("Norway",      "L. Østigård",        "DEF",  52),
    ("Norway",      "Ø. Nyland",          "GK",   50),
    # ── Uruguay ─────────────────────────────────────────────────────────────
    ("Uruguay",     "D. Núñez",           "FWD",  75),
    ("Uruguay",     "F. Valverde",        "MID",  75),
    ("Uruguay",     "G. De Arrascaeta",   "MID",  65),
    ("Uruguay",     "R. Bentancur",       "MID",  68),
    ("Uruguay",     "R. Araújo",          "DEF",  62),
    ("Uruguay",     "J. Rochet",          "GK",   50),
    # ── Japan ───────────────────────────────────────────────────────────────
    ("Japan",       "T. Kubo",            "MID",  70),
    ("Japan",       "A. Ueda",            "FWD",  70),
    ("Japan",       "T. Minamino",        "MID",  62),
    ("Japan",       "H. Maeda",           "FWD",  60),
    ("Japan",       "M. Tomiyasu",        "DEF",  55),
    ("Japan",       "S. Gonda",           "GK",   50),
    # ── United States ───────────────────────────────────────────────────────
    ("United States","C. Pulisic",        "MID",  70),
    ("United States","Y. Musah",          "MID",  62),
    ("United States","T. Adams",          "MID",  62),
    ("United States","G. Reyna",          "MID",  60),
    ("United States","M. Turner",         "GK",   50),
    # ── Mexico ──────────────────────────────────────────────────────────────
    ("Mexico",      "S. Giménez",         "FWD",  68),
    ("Mexico",      "R. Jiménez",         "FWD",  70),
    ("Mexico",      "H. Lozano",          "FWD",  68),
    ("Mexico",      "E. Álvarez",         "MID",  58),
    ("Mexico",      "G. Ochoa",           "GK",   50),
    # ── Poland ──────────────────────────────────────────────────────────────
    ("Poland",      "R. Lewandowski",     "FWD",  90),
    ("Poland",      "P. Zieliński",       "MID",  65),
    ("Poland",      "S. Szymański",       "MID",  60),
    ("Poland",      "W. Szczęsny",        "GK",   55),
    # ── Serbia ──────────────────────────────────────────────────────────────
    ("Serbia",      "D. Vlahović",        "FWD",  75),
    ("Serbia",      "A. Mitrović",        "FWD",  75),
    ("Serbia",      "D. Tadić",           "MID",  68),
    ("Serbia",      "P. Rajković",        "GK",   50),
    # ── Denmark ─────────────────────────────────────────────────────────────
    ("Denmark",     "R. Højlund",         "FWD",  75),
    ("Denmark",     "C. Eriksen",         "MID",  70),
    ("Denmark",     "P. Højbjerg",        "MID",  62),
    ("Denmark",     "K. Schmeichel",      "GK",   50),
    # ── Colombia ────────────────────────────────────────────────────────────
    ("Colombia",    "L. Díaz",            "FWD",  81),
    ("Colombia",    "J. Rodríguez",       "MID",  65),
    ("Colombia",    "R. Quintero",        "MID",  65),
    ("Colombia",    "D. Ospina",          "GK",   50),
    # ── Ecuador ─────────────────────────────────────────────────────────────
    ("Ecuador",     "M. Caicedo",         "MID",  68),
    ("Ecuador",     "E. Valencia",        "FWD",  68),
    ("Ecuador",     "B. Méndez",          "MID",  58),
    ("Ecuador",     "A. Domínguez",       "GK",   50),
    # ── Senegal ─────────────────────────────────────────────────────────────
    ("Senegal",     "S. Mané",            "FWD",  76),
    ("Senegal",     "I. Ndiaye",          "FWD",  65),
    ("Senegal",     "I. Gueye",           "MID",  62),
    ("Senegal",     "E. Mendy",           "GK",   52),
    # ── Ghana ───────────────────────────────────────────────────────────────
    ("Ghana",       "M. Kudus",           "MID",  75),
    ("Ghana",       "A. Semenyo",         "FWD",  72),
    ("Ghana",       "T. Partey",          "MID",  65),
    ("Ghana",       "J. Ayew",            "FWD",  60),
    ("Ghana",       "L. Bati",            "GK",   50),
    # ── Switzerland ─────────────────────────────────────────────────────────
    ("Switzerland", "B. Embolo",          "FWD",  75),
    ("Switzerland", "D. Ndoye",           "MID",  68),
    ("Switzerland", "G. Xhaka",           "MID",  68),
    ("Switzerland", "X. Shaqiri",         "MID",  65),
    ("Switzerland", "Y. Sommer",          "GK",   55),
    # ── Australia ───────────────────────────────────────────────────────────
    ("Australia",   "A. Hrustic",         "MID",  62),
    ("Australia",   "M. Leckie",          "MID",  58),
    ("Australia",   "J. Irvine",          "MID",  55),
    ("Australia",   "M. Ryan",            "GK",   50),
    # ── South Korea ─────────────────────────────────────────────────────────
    ("South Korea", "Son Heung-min",      "FWD",  74),
    ("South Korea", "Hwang Hee-chan",     "FWD",  65),
    ("South Korea", "Lee Jae-sung",       "MID",  62),
    ("South Korea", "Kim Min-jae",        "DEF",  68),
    ("South Korea", "Kim Seung-gyu",      "GK",   50),
    # ── Iran ────────────────────────────────────────────────────────────────
    ("Iran",        "M. Taremi",          "FWD",  68),
    ("Iran",        "S. Azmoun",          "FWD",  65),
    ("Iran",        "A. Jahanbakhsh",     "MID",  60),
    ("Iran",        "A. Beiranvand",      "GK",   50),
    # ── Canada ──────────────────────────────────────────────────────────────
    ("Canada",      "J. David",           "FWD",  70),
    ("Canada",      "A. Davies",          "DEF",  72),
    ("Canada",      "S. Larin",           "FWD",  60),
    ("Canada",      "M. Borjan",          "GK",   50),
    # ── Saudi Arabia ────────────────────────────────────────────────────────
    ("Saudi Arabia","S. Al-Dawsari",      "MID",  72),
    ("Saudi Arabia","F. Al-Buraikan",     "FWD",  62),
    ("Saudi Arabia","M. Al-Owais",        "GK",   50),
    # ── Nigeria ─────────────────────────────────────────────────────────────
    ("Nigeria",     "V. Osimhen",         "FWD",  78),
    ("Nigeria",     "T. Lookman",         "MID",  72),
    ("Nigeria",     "F. Onyeka",          "MID",  58),
    ("Nigeria",     "W. Troost-Ekong",    "DEF",  52),
    ("Nigeria",     "S. Obi",             "GK",   50),
    # ── Tunisia ─────────────────────────────────────────────────────────────
    ("Tunisia",     "Y. Msakni",          "MID",  60),
    ("Tunisia",     "S. Jaziri",          "FWD",  55),
    ("Tunisia",     "A. Dahmen",          "GK",   50),
    # ── Qatar ───────────────────────────────────────────────────────────────
    ("Qatar",       "A. Afif",            "MID",  62),
    ("Qatar",       "A. Ali",             "FWD",  55),
    ("Qatar",       "M. Barsham",         "GK",   50),
    # ── Cameroon ────────────────────────────────────────────────────────────
    ("Cameroon",    "A. Anguissa",        "MID",  70),
    ("Cameroon",    "V. Aboubakar",       "FWD",  65),
    ("Cameroon",    "A. Onana",           "GK",   60),
    # ── Sweden ──────────────────────────────────────────────────────────────
    ("Sweden",      "A. Isak",            "FWD",  80),
    ("Sweden",      "V. Gyökeres",        "FWD",  78),
    ("Sweden",      "D. Kulusevski",      "MID",  75),
    ("Sweden",      "V. Lindelöf",        "DEF",  58),
    ("Sweden",      "R. Olsen",           "GK",   50),
    # ── Peru ────────────────────────────────────────────────────────────────
    ("Peru",        "A. Lapadula",        "FWD",  58),
    ("Peru",        "C. Cueva",           "MID",  55),
    ("Peru",        "P. Gallese",         "GK",   50),
    # ── Iceland ─────────────────────────────────────────────────────────────
    ("Iceland",     "G. Sigurdsson",      "MID",  60),
    ("Iceland",     "A. Böðvarsson",      "FWD",  55),
    ("Iceland",     "H. Valdimarsson",    "GK",   50),
    # ── Panama ──────────────────────────────────────────────────────────────
    ("Panama",      "A. Carrasquilla",    "MID",  65),
    ("Panama",      "R. Torres",          "DEF",  55),
    ("Panama",      "L. Mejía",           "GK",   50),
    # ── Costa Rica ──────────────────────────────────────────────────────────
    ("Costa Rica",  "K. Navas",           "GK",   60),
    ("Costa Rica",  "J. Campbell",        "FWD",  58),
    ("Costa Rica",  "C. Borges",          "MID",  52),
    # ── Egypt ───────────────────────────────────────────────────────────────
    ("Egypt",       "M. Salah",           "FWD", 100),
    ("Egypt",       "O. Marmoush",        "FWD",  78),
    ("Egypt",       "M. El-Shenawy",      "GK",   50),
    # ── Turkey ──────────────────────────────────────────────────────────────
    ("Turkey",      "H. Çalhanoğlu",      "MID",  71),
    ("Turkey",      "K. Yıldız",          "MID",  70),
    ("Turkey",      "A. Güler",           "MID",  70),
    ("Turkey",      "M. Günok",           "GK",   50),
    # ── Czech Republic ──────────────────────────────────────────────────────
    ("Czech Republic","P. Schick",        "FWD",  73),
    ("Czech Republic","A. Hložek",        "FWD",  65),
    ("Czech Republic","T. Souček",        "MID",  58),
    ("Czech Republic","J. Staněk",        "GK",   50),
    # ── Austria ─────────────────────────────────────────────────────────────
    ("Austria",     "M. Sabitzer",        "MID",  68),
    ("Austria",     "C. Baumgartner",     "MID",  67),
    ("Austria",     "M. Arnautović",      "FWD",  62),
    ("Austria",     "P. Pentz",           "GK",   50),
    # ── Scotland ────────────────────────────────────────────────────────────
    ("Scotland",    "S. McTominay",       "MID",  65),
    ("Scotland",    "L. Docherty",        "MID",  55),
    ("Scotland",    "A. Robertson",       "DEF",  60),
    ("Scotland",    "A. Gunn",            "GK",   50),
    # ── Algeria ─────────────────────────────────────────────────────────────
    ("Algeria",     "R. Mahrez",          "FWD",  65),
    ("Algeria",     "I. Bennacer",        "MID",  60),
    ("Algeria",     "Y. Atal",            "DEF",  55),
    ("Algeria",     "R. M'Bolhi",         "GK",   50),
    # ── Paraguay ────────────────────────────────────────────────────────────
    ("Paraguay",    "J. Enciso",          "MID",  66),
    ("Paraguay",    "A. Sanabria",        "FWD",  65),
    ("Paraguay",    "M. Almiron",         "MID",  62),
    ("Paraguay",    "A. Silva",           "GK",   50),
    # ── DR Congo ────────────────────────────────────────────────────────────
    ("DR Congo",    "C. Bakambu",         "FWD",  65),
    ("DR Congo",    "A. Mbemba",          "DEF",  52),
    ("DR Congo",    "Y. Mpasi",           "GK",   50),
    # ── Uzbekistan ──────────────────────────────────────────────────────────
    ("Uzbekistan",  "E. Shomurodov",      "FWD",  65),
    ("Uzbekistan",  "O. Zubidov",         "MID",  55),
    ("Uzbekistan",  "U. Jaloliddinov",    "GK",   50),
    # ── New Zealand ─────────────────────────────────────────────────────────
    ("New Zealand", "C. Wood",            "FWD",  65),
    ("New Zealand", "C. Sargeant",        "FWD",  55),
    ("New Zealand", "O. Sail",            "GK",   50),
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
