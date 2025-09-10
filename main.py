import pandas as pd 
import numpy as np
import nfl_data_py as nfl


#training_years = [2015,2016,2017]


#training_data = nfl.import_pbp_data(training_years, ['air_yards'])

#df = pd.DataFrame(training_data)
#print(training_data)
#print(df.head())

# weekly = nfl.import_weekly_data([2015])

# team_game_data = (
#     weekly.groupby(['season', 'week', 'player_name'])
#     .sum(numeric_only=True)
#     .reset_index()
# )


#print(nfl.see_weekly_cols())
# print(team_game_data.head())


#print(nfl.see_pbp_cols())
season= nfl.import_pbp_data([2024])

season_pbp = (
    season.groupby(['game_id', 'home_team', 'away_team'])
)
print(season_pbp.head())
