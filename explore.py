import requests
import pandas as pd

# 1. Fetch the FPL bootstrap data
URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
response = requests.get(URL)
response.raise_for_status()  # crashes if the request failed
data = response.json()

# 2. See what the response contains at the top level
print("Top-level keys in the FPL API response:")
print(list(data.keys()))
print()

# 3. The 'elements' key contains player data. Load it into a DataFrame.
players = pd.DataFrame(data["elements"])
print(f"Total players: {len(players)}")
print(f"Columns available ({len(players.columns)}): {list(players.columns)}")
print()

# 4. Top 10 players by total points this season
top10 = players.nlargest(10, "total_points")[
    ["first_name", "second_name", "total_points", "now_cost", "minutes"]
]
print("Top 10 players by total points:")
print(top10.to_string(index=False))
print()

# 5. Bonus: top 10 by points-per-million (a basic value metric)
players["points_per_million"] = players["total_points"] / (players["now_cost"] / 10)
top_value = players[players["minutes"] > 500].nlargest(10, "points_per_million")[
    ["first_name", "second_name", "total_points", "now_cost", "points_per_million"]
]
print("Top 10 by points-per-million (min 500 minutes):")
print(top_value.to_string(index=False))