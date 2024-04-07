N_EPOCHS = 20
BATCH_SIZE = 512
LSTM_N_UNITS = 300
LR = 1e-4
DENSE_N_UNITS = 1
SEED = 0

DATA_PATH = "data/dataset.csv"
CHECKPOINT_FOLDER = "pickles/recommender"

CATEGORICAL_FEATURES = ['player_position', 'week_no']
COLUMN_NAMES = ['player_position', 'last10_ratio_cleanSheets_opp', 'last10_ratio_cleanSheets_own',
                'last10_ratio_wins_opp', 'last10_ratio_wins_own', 'last3_assists', 'last3_goals', 'last3_ratio_points',
                'last3_ycards', 'opp_team_rank', 'player_team_rank', 'player_value', 'ratio_assists',
                'ratio_attempted_passes', 'ratio_big_chancesCreated', 'ratio_big_chancesMiss', 'ratio_creativity',
                'ratio_dribbles', 'ratio_fouls', 'ratio_goals_conceded_opp_team', 'ratio_goals_conceded_player_team',
                'ratio_goals_opp_team', 'ratio_goals_player_team', 'ratio_goals_scored', 'ratio_key_passes',
                'ratio_leading_goal', 'ratio_minutes_played', 'ratio_offsides', 'ratio_open_playcross',
                'ratio_own_goals', 'ratio_penalties_conceded', 'ratio_penalties_missed', 'ratio_penalties_saved',
                'ratio_saves', 'ratio_selection', 'ratio_tackles', 'ratio_threat', 'week_no', 'week_points']

FEATURE_NAMES = ['last10_ratio_cleanSheets_opp', 'last10_ratio_cleanSheets_own',
                 'last10_ratio_wins_opp', 'last10_ratio_wins_own', 'last3_assists',
                 'last3_goals', 'last3_ratio_points', 'last3_ycards', 'opp_team_rank',
                 'player_team_rank', 'player_value', 'ratio_assists',
                 'ratio_attempted_passes', 'ratio_big_chancesCreated',
                 'ratio_big_chancesMiss', 'ratio_creativity', 'ratio_dribbles',
                 'ratio_fouls', 'ratio_goals_conceded_opp_team',
                 'ratio_goals_conceded_player_team', 'ratio_goals_opp_team',
                 'ratio_goals_player_team', 'ratio_goals_scored', 'ratio_key_passes',
                 'ratio_leading_goal', 'ratio_minutes_played', 'ratio_offsides',
                 'ratio_open_playcross', 'ratio_own_goals', 'ratio_penalties_conceded',
                 'ratio_penalties_missed', 'ratio_penalties_saved', 'ratio_saves',
                 'ratio_selection', 'ratio_tackles', 'ratio_threat',
                 'player_position_FWD', 'player_position_GKP', 'player_position_MID',
                 'week_no_3', 'week_no_4', 'week_no_5', 'week_no_6', 'week_no_7',
                 'week_no_8', 'week_no_9', 'week_no_10', 'week_no_11', 'week_no_12',
                 'week_no_13', 'week_no_14', 'week_no_15', 'week_no_16', 'week_no_17',
                 'week_no_18', 'week_no_19', 'week_no_20', 'week_no_21', 'week_no_22',
                 'week_no_23', 'week_no_24', 'week_no_25', 'week_no_26', 'week_no_27',
                 'week_no_28', 'week_no_29', 'week_no_30', 'week_no_31', 'week_no_32',
                 'week_no_33', 'week_no_34', 'week_no_35', 'week_no_36', 'week_no_37',
                 'week_no_38']

COLUMNS_TO_DISPLAY_IN_PREDICTION_WEB = ["player_name", "player_team_name", "player_position"]
COLUMNS_TO_DISPLAY_IN_PLAYERS_WEB = ["player_name",
                                     "player_team_name",
                                     "player_position",
                                     "last10_ratio_wins_own",
                                     "last3_assists",
                                     "last3_goals",
                                     "last3_ratio_points",
                                     "last3_ycards"]

TARGET_VARIABLE = "week_points"
WEEK_GAME_NUMBERS_2019 = [2, 3, 4, 5, 6, 7, 8]
