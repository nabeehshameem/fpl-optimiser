"""
seed_fantasy_players.py
Seeds WC2026 fantasy player estimates into the fantasy_players table.

Real FWD/MID prices sourced from FIFA WC Fantasy app (as of May 2026).
DEF/GK prices are estimated until officially released.
Replace with full confirmed data after June 2 squad deadline:
  python wc/scripts/ingest_fantasy_players.py --json players.json
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
# WC2026 groups: A=MEX/KOR/RSA/CZE  B=CAN/SUI/QAT/BIH  C=BRA/MAR/SCO/HAI
#                D=USA/AUS/PAR/TUR  E=GER/CUW/CIV/ECU  F=NED/JPN/TUN/SWE
#                G=BEL/IRN/EGY/NZL  H=ESP/URU/KSA/CPV  I=FRA/SEN/NOR/IRQ
#                J=ARG/AUT/ALG/JOR  K=POR/COL/UZB/COD  L=ENG/CRO/PAN/GHA
SEED_PLAYERS = [
    # ── GROUP I ─────────────────────────────────────────────────────────────
    # France
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
    ("France",      "A. Rabiot",          "MID",  64),
    ("France",      "W. Zaïre-Emery",     "MID",  61),
    ("France",      "T. Hernández",       "DEF",  60),
    ("France",      "D. Upamecano",       "DEF",  55),
    ("France",      "M. Maignan",         "GK",   55),
    # Senegal
    ("Senegal",     "S. Mané",            "FWD",  76),
    ("Senegal",     "I. Ndiaye",          "FWD",  65),
    ("Senegal",     "I. Sarr",            "FWD",  62),
    ("Senegal",     "L. Camara",          "MID",  62),
    ("Senegal",     "I. Gueye",           "MID",  62),
    ("Senegal",     "K. Diatta",          "FWD",  61),
    ("Senegal",     "E. Mendy",           "GK",   52),
    # Norway
    ("Norway",      "E. Haaland",         "FWD", 105),
    ("Norway",      "M. Ødegaard",        "MID",  77),
    ("Norway",      "A. Sørloth",         "FWD",  68),
    ("Norway",      "F. Aursnes",         "MID",  65),
    ("Norway",      "A. Schjelderup",     "MID",  62),
    ("Norway",      "K. Thorstvedt",      "MID",  62),
    ("Norway",      "S. Berge",           "MID",  62),
    ("Norway",      "A. Nusa",            "FWD",  61),
    ("Norway",      "L. Østigård",        "DEF",  52),
    ("Norway",      "Ø. Nyland",          "GK",   50),
    # Iraq
    ("Iraq",        "M. Ali",             "FWD",  62),
    ("Iraq",        "B. Resan",           "MID",  60),
    ("Iraq",        "A. Al-Dakhil",       "DEF",  60),
    ("Iraq",        "J. Hassan",          "GK",   50),

    # ── GROUP C ─────────────────────────────────────────────────────────────
    # Brazil
    ("Brazil",      "Vinicius Jr",        "FWD", 100),
    ("Brazil",      "Raphinha",           "MID",  82),
    ("Brazil",      "M. Cunha",           "FWD",  73),
    ("Brazil",      "Neymar",             "FWD",  72),
    ("Brazil",      "B. Guimarães",       "MID",  68),
    ("Brazil",      "G. Martinelli",      "FWD",  65),
    ("Brazil",      "L. Paquetá",         "MID",  65),
    ("Brazil",      "Casemiro",           "MID",  63),
    ("Brazil",      "L. Henrique",        "FWD",  61),
    ("Brazil",      "É. Militão",         "DEF",  58),
    ("Brazil",      "Alisson",            "GK",   55),
    # Morocco
    ("Morocco",     "Y. En-Nesyri",       "FWD",  70),
    ("Morocco",     "I. Saibari",         "MID",  68),
    ("Morocco",     "H. Ziyech",          "MID",  68),
    ("Morocco",     "M. Akliouche",       "MID",  64),
    ("Morocco",     "A. Ounahi",          "MID",  62),
    ("Morocco",     "B. El Khannouss",    "MID",  62),
    ("Morocco",     "A. Hakimi",          "DEF",  70),
    ("Morocco",     "Y. Bono",            "GK",   55),
    # Scotland
    ("Scotland",    "S. McTominay",       "MID",  65),
    ("Scotland",    "L. Docherty",        "MID",  55),
    ("Scotland",    "A. Robertson",       "DEF",  60),
    ("Scotland",    "A. Gunn",            "GK",   50),
    # Haiti
    ("Haiti",       "F. Pierrot",         "FWD",  62),
    ("Haiti",       "D. Nazon",           "FWD",  58),
    ("Haiti",       "J. Placide",         "GK",   50),

    # ── GROUP L ─────────────────────────────────────────────────────────────
    # England
    ("England",     "H. Kane",            "FWD", 105),
    ("England",     "B. Saka",            "MID",  95),
    ("England",     "J. Bellingham",      "MID",  83),
    ("England",     "P. Foden",           "MID",  80),
    ("England",     "E. Eze",             "MID",  80),
    ("England",     "C. Palmer",          "MID",  80),
    ("England",     "O. Watkins",         "FWD",  79),
    ("England",     "M. Rashford",        "FWD",  75),
    ("England",     "I. Toney",           "FWD",  75),
    ("England",     "M. Rogers",          "MID",  72),
    ("England",     "A. Gordon",          "MID",  70),
    ("England",     "D. Rice",            "MID",  70),
    ("England",     "K. Mainoo",          "MID",  61),
    ("England",     "T. Alexander-Arnold","DEF",  68),
    ("England",     "J. Pickford",        "GK",   52),
    # Croatia
    ("Croatia",     "A. Budimir",         "FWD",  68),
    ("Croatia",     "M. Pasalic",         "MID",  64),
    ("Croatia",     "L. Modrić",          "MID",  62),
    ("Croatia",     "A. Kramarić",        "FWD",  62),
    ("Croatia",     "M. Baturina",        "MID",  65),
    ("Croatia",     "I. Perišić",         "MID",  65),
    ("Croatia",     "N. Vlašić",          "MID",  61),
    ("Croatia",     "J. Gvardiol",        "DEF",  72),
    ("Croatia",     "D. Livaković",       "GK",   55),
    # Panama
    ("Panama",      "A. Carrasquilla",    "MID",  65),
    ("Panama",      "R. Torres",          "DEF",  55),
    ("Panama",      "L. Mejía",           "GK",   50),
    # Ghana
    ("Ghana",       "M. Kudus",           "MID",  75),
    ("Ghana",       "A. Semenyo",         "FWD",  72),
    ("Ghana",       "A. Fatawu",          "FWD",  64),
    ("Ghana",       "T. Partey",          "MID",  65),
    ("Ghana",       "J. Ayew",            "FWD",  60),
    ("Ghana",       "L. Bati",            "GK",   50),

    # ── GROUP J ─────────────────────────────────────────────────────────────
    # Argentina
    ("Argentina",   "L. Messi",           "FWD", 100),
    ("Argentina",   "L. Martínez",        "FWD",  88),
    ("Argentina",   "J. Álvarez",         "FWD",  86),
    ("Argentina",   "E. Fernández",       "MID",  75),
    ("Argentina",   "A. Mac Allister",    "MID",  66),
    ("Argentina",   "E. Buendía",         "MID",  65),
    ("Argentina",   "R. De Paul",         "MID",  65),
    ("Argentina",   "G. Lo Celso",        "MID",  62),
    ("Argentina",   "M. Soulé",           "MID",  62),
    ("Argentina",   "E. Barco",           "MID",  62),
    ("Argentina",   "C. Romero",          "DEF",  62),
    ("Argentina",   "E. Martínez",        "GK",   58),
    # Austria
    ("Austria",     "M. Sabitzer",        "MID",  68),
    ("Austria",     "C. Baumgartner",     "MID",  67),
    ("Austria",     "M. Arnautović",      "FWD",  62),
    ("Austria",     "P. Pentz",           "GK",   50),
    # Algeria
    ("Algeria",     "R. Mahrez",          "FWD",  65),
    ("Algeria",     "N. Amiri",           "MID",  63),
    ("Algeria",     "N. Amoura",          "FWD",  62),
    ("Algeria",     "A. Gouiri",          "FWD",  62),
    ("Algeria",     "I. Bennacer",        "MID",  60),
    ("Algeria",     "Y. Atal",            "DEF",  55),
    ("Algeria",     "R. M'Bolhi",         "GK",   50),
    # Jordan
    ("Jordan",      "M. Al-Taamari",      "FWD",  62),
    ("Jordan",      "Y. Al-Naimat",       "FWD",  60),
    ("Jordan",      "A. Bani Yaseen",     "MID",  58),
    ("Jordan",      "A. Shafi",           "GK",   50),

    # ── GROUP H ─────────────────────────────────────────────────────────────
    # Spain
    ("Spain",       "L. Yamal",           "MID", 100),
    ("Spain",       "Pedri",              "MID",  81),
    ("Spain",       "M. Oyarzabal",       "FWD",  81),
    ("Spain",       "F. Torres",          "FWD",  78),
    ("Spain",       "N. Williams",        "MID",  78),
    ("Spain",       "D. Olmo",            "MID",  77),
    ("Spain",       "Rodri",              "MID",  75),
    ("Spain",       "Gavi",               "MID",  65),
    ("Spain",       "M. Merino",          "MID",  62),
    ("Spain",       "M. Zubimendi",       "MID",  61),
    ("Spain",       "R. Carvajal",        "DEF",  55),
    ("Spain",       "Unai Simón",         "GK",   55),
    # Uruguay
    ("Uruguay",     "D. Núñez",           "FWD",  75),
    ("Uruguay",     "F. Valverde",        "MID",  75),
    ("Uruguay",     "R. Bentancur",       "MID",  68),
    ("Uruguay",     "G. De Arrascaeta",   "MID",  65),
    ("Uruguay",     "R. Araújo",          "DEF",  64),
    ("Uruguay",     "J. Rochet",          "GK",   50),
    # Saudi Arabia
    ("Saudi Arabia","S. Al-Dawsari",      "MID",  72),
    ("Saudi Arabia","F. Al-Buraikan",     "FWD",  62),
    ("Saudi Arabia","M. Al-Owais",        "GK",   50),
    # Cape Verde
    ("Cape Verde",  "J. Tavares",         "FWD",  65),
    ("Cape Verde",  "R. Mendes",          "FWD",  63),
    ("Cape Verde",  "G. Rodrigues",       "FWD",  62),
    ("Cape Verde",  "Vozinha",            "GK",   50),

    # ── GROUP K ─────────────────────────────────────────────────────────────
    # Portugal
    ("Portugal",    "C. Ronaldo",         "FWD", 100),
    ("Portugal",    "B. Fernandes",       "MID",  85),
    ("Portugal",    "B. Silva",           "MID",  78),
    ("Portugal",    "R. Leão",            "FWD",  78),
    ("Portugal",    "G. Ramos",           "FWD",  75),
    ("Portugal",    "Vitinha",            "MID",  64),
    ("Portugal",    "J. Félix",           "FWD",  65),
    ("Portugal",    "J. Neves",           "MID",  65),
    ("Portugal",    "M. Nunes",           "MID",  61),
    ("Portugal",    "R. Dias",            "DEF",  62),
    ("Portugal",    "Diogo Costa",        "GK",   55),
    # Colombia
    ("Colombia",    "L. Díaz",            "FWD",  81),
    ("Colombia",    "Cucho Hernández",    "FWD",  63),
    ("Colombia",    "J. Rodríguez",       "MID",  62),
    ("Colombia",    "R. Quintero",        "MID",  65),
    ("Colombia",    "J. Carrascal",       "MID",  61),
    ("Colombia",    "J. Córdoba",         "FWD",  61),
    ("Colombia",    "C. Vargas",          "GK",   50),
    # Uzbekistan
    ("Uzbekistan",  "E. Shomurodov",      "FWD",  65),
    ("Uzbekistan",  "A. Fayzullayev",     "MID",  62),
    ("Uzbekistan",  "J. Masharipov",      "MID",  61),
    ("Uzbekistan",  "O. Zubidov",         "MID",  55),
    ("Uzbekistan",  "U. Jaloliddinov",    "GK",   50),
    # DR Congo
    ("DR Congo",    "C. Bakambu",         "FWD",  65),
    ("DR Congo",    "Y. Wissa",           "FWD",  62),
    ("DR Congo",    "T. Bongonda",        "FWD",  62),
    ("DR Congo",    "A. Mbemba",          "DEF",  52),
    ("DR Congo",    "Y. Mpasi",           "GK",   50),

    # ── GROUP D ─────────────────────────────────────────────────────────────
    # United States
    ("United States","C. Pulisic",        "MID",  70),
    ("United States","Y. Musah",          "MID",  62),
    ("United States","T. Adams",          "MID",  62),
    ("United States","M. Tillman",        "MID",  61),
    ("United States","W. McKennie",       "MID",  61),
    ("United States","G. Reyna",          "MID",  60),
    ("United States","M. Turner",         "GK",   50),
    # Australia
    ("Australia",   "A. Hrustic",         "MID",  62),
    ("Australia",   "M. Leckie",          "MID",  58),
    ("Australia",   "J. Irvine",          "MID",  55),
    ("Australia",   "M. Ryan",            "GK",   50),
    # Paraguay
    ("Paraguay",    "J. Enciso",          "MID",  66),
    ("Paraguay",    "A. Sanabria",        "FWD",  65),
    ("Paraguay",    "M. Almiron",         "MID",  62),
    ("Paraguay",    "A. Silva",           "GK",   50),
    # Turkey
    ("Turkey",      "H. Çalhanoğlu",      "MID",  71),
    ("Turkey",      "K. Yıldız",          "MID",  70),
    ("Turkey",      "A. Güler",           "MID",  70),
    ("Turkey",      "K. Aktürkoğlu",      "MID",  62),
    ("Turkey",      "B. Yılmaz",          "FWD",  61),
    ("Turkey",      "M. Günok",           "GK",   50),

    # ── GROUP E ─────────────────────────────────────────────────────────────
    # Germany
    ("Germany",     "J. Musiala",         "MID",  80),
    ("Germany",     "K. Havertz",         "FWD",  78),
    ("Germany",     "F. Wirtz",           "MID",  75),
    ("Germany",     "J. Kimmich",         "DEF",  75),
    ("Germany",     "L. Sané",            "MID",  74),
    ("Germany",     "N. Woltemade",       "FWD",  72),
    ("Germany",     "D. Undav",           "FWD",  66),
    ("Germany",     "M. Beier",           "FWD",  65),
    ("Germany",     "J. Leweling",        "MID",  64),
    ("Germany",     "L. Goretzka",        "MID",  61),
    ("Germany",     "A. Rüdiger",         "DEF",  58),
    ("Germany",     "M. Neuer",           "GK",   52),
    # Curaçao
    ("Curaçao",     "J. Antonia",         "FWD",  62),
    ("Curaçao",     "L. Bacuna",          "MID",  60),
    ("Curaçao",     "E. Room",            "GK",   50),
    # Ivory Coast
    ("Ivory Coast", "S. Adingra",         "FWD",  68),
    ("Ivory Coast", "S. Fofana",          "MID",  65),
    ("Ivory Coast", "S. Haller",          "FWD",  63),
    ("Ivory Coast", "I. Sangaré",         "MID",  65),
    ("Ivory Coast", "O. Koné",            "MID",  61),
    ("Ivory Coast", "Y. Fofana",          "GK",   52),
    # Ecuador
    ("Ecuador",     "M. Caicedo",         "MID",  68),
    ("Ecuador",     "E. Valencia",        "FWD",  68),
    ("Ecuador",     "B. Méndez",          "MID",  58),
    ("Ecuador",     "A. Domínguez",       "GK",   50),

    # ── GROUP F ─────────────────────────────────────────────────────────────
    # Netherlands
    ("Netherlands", "F. de Jong",         "MID",  80),
    ("Netherlands", "C. Gakpo",           "FWD",  77),
    ("Netherlands", "J. Timber",          "MID",  68),
    ("Netherlands", "X. Simons",          "MID",  65),
    ("Netherlands", "T. Reijnders",       "MID",  65),
    ("Netherlands", "T. Koopmeiners",     "MID",  62),
    ("Netherlands", "R. Gravenberch",     "MID",  61),
    ("Netherlands", "J. Schouten",        "MID",  61),
    ("Netherlands", "N. Madueke",         "MID",  61),
    ("Netherlands", "V. van Dijk",        "DEF",  72),
    ("Netherlands", "B. Flekken",         "GK",   50),
    # Japan
    ("Japan",       "T. Kubo",            "MID",  70),
    ("Japan",       "A. Ueda",            "FWD",  70),
    ("Japan",       "T. Minamino",        "MID",  62),
    ("Japan",       "H. Maeda",           "FWD",  60),
    ("Japan",       "M. Tomiyasu",        "DEF",  55),
    ("Japan",       "S. Gonda",           "GK",   50),
    # Tunisia
    ("Tunisia",     "Y. Msakni",          "MID",  60),
    ("Tunisia",     "S. Jaziri",          "FWD",  55),
    ("Tunisia",     "A. Dahmen",          "GK",   50),
    # Sweden
    ("Sweden",      "A. Isak",            "FWD",  80),
    ("Sweden",      "V. Gyökeres",        "FWD",  78),
    ("Sweden",      "D. Kulusevski",      "MID",  75),
    ("Sweden",      "M. Svanberg",        "MID",  61),
    ("Sweden",      "V. Lindelöf",        "DEF",  58),
    ("Sweden",      "R. Olsen",           "GK",   50),

    # ── GROUP G ─────────────────────────────────────────────────────────────
    # Belgium
    ("Belgium",     "K. De Bruyne",       "MID",  75),
    ("Belgium",     "J. Doku",            "MID",  75),
    ("Belgium",     "R. Lukaku",          "FWD",  74),
    ("Belgium",     "L. Trossard",        "MID",  66),
    ("Belgium",     "Y. Tielemans",       "MID",  61),
    ("Belgium",     "T. Courtois",        "GK",   60),
    ("Belgium",     "T. Castagne",        "DEF",  55),
    # Iran
    ("Iran",        "S. Azmoun",          "FWD",  65),
    ("Iran",        "M. Taremi",          "FWD",  61),
    ("Iran",        "A. Jahanbakhsh",     "MID",  60),
    ("Iran",        "A. Beiranvand",      "GK",   50),
    # Egypt
    ("Egypt",       "M. Salah",           "FWD", 100),
    ("Egypt",       "O. Marmoush",        "FWD",  78),
    ("Egypt",       "M. El-Shenawy",      "GK",   50),
    # New Zealand
    ("New Zealand", "C. Wood",            "FWD",  65),
    ("New Zealand", "C. Sargeant",        "FWD",  55),
    ("New Zealand", "O. Sail",            "GK",   50),

    # ── GROUP A ─────────────────────────────────────────────────────────────
    # Mexico
    ("Mexico",      "R. Jiménez",         "FWD",  70),
    ("Mexico",      "S. Giménez",         "FWD",  68),
    ("Mexico",      "H. Lozano",          "FWD",  68),
    ("Mexico",      "Á. Fidalgo",         "MID",  64),
    ("Mexico",      "B. Huerta",          "MID",  63),
    ("Mexico",      "E. Álvarez",         "MID",  58),
    ("Mexico",      "G. Ochoa",           "GK",   50),
    # South Korea
    ("South Korea", "Son Heung-min",      "FWD",  74),
    ("South Korea", "Lee Jae-sung",       "MID",  62),
    ("South Korea", "Hwang Hee-chan",     "FWD",  61),
    ("South Korea", "Lee Kang-in",        "MID",  61),
    ("South Korea", "Kim Min-jae",        "DEF",  68),
    ("South Korea", "Kim Seung-gyu",      "GK",   50),
    # South Africa
    ("South Africa","P. Tau",             "FWD",  65),
    ("South Africa","L. Foster",          "FWD",  63),
    ("South Africa","T. Zwane",           "MID",  62),
    ("South Africa","R. Williams",        "GK",   50),
    # Czech Republic
    ("Czech Republic","P. Schick",        "FWD",  73),
    ("Czech Republic","A. Hložek",        "FWD",  65),
    ("Czech Republic","T. Souček",        "MID",  58),
    ("Czech Republic","J. Staněk",        "GK",   50),

    # ── GROUP B ─────────────────────────────────────────────────────────────
    # Canada
    ("Canada",      "J. David",           "FWD",  70),
    ("Canada",      "A. Davies",          "DEF",  72),
    ("Canada",      "S. Larin",           "FWD",  62),
    ("Canada",      "M. Borjan",          "GK",   50),
    # Switzerland
    ("Switzerland", "B. Embolo",          "FWD",  75),
    ("Switzerland", "D. Ndoye",           "MID",  68),
    ("Switzerland", "X. Shaqiri",         "MID",  65),
    ("Switzerland", "G. Xhaka",           "MID",  62),
    ("Switzerland", "F. Rieder",          "MID",  62),
    ("Switzerland", "D. Sow",             "MID",  61),
    ("Switzerland", "D. Zakaria",         "MID",  61),
    ("Switzerland", "C. Fassnacht",       "MID",  61),
    ("Switzerland", "Y. Sommer",          "GK",   55),
    # Qatar
    ("Qatar",       "A. Afif",            "MID",  62),
    ("Qatar",       "A. Ali",             "FWD",  55),
    ("Qatar",       "M. Barsham",         "GK",   50),
    # Bosnia and Herzegovina
    ("Bosnia and Herzegovina","E. Demirović","FWD",62),
    ("Bosnia and Herzegovina","E. Džeko",   "FWD",61),
    ("Bosnia and Herzegovina","E. Kolašinac","DEF",55),
    ("Bosnia and Herzegovina","V. Kovačević","MID",58),
    ("Bosnia and Herzegovina","N. Šehić",   "GK", 50),
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
