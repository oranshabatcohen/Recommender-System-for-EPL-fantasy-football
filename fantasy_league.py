from flask import Flask, render_template, request
from consts import WEEK_GAME_NUMBERS_2019, DATA_PATH, COLUMNS_TO_DISPLAY_IN_PLAYERS_WEB
from inference import infer, random_seed
import pandas as pd

app = Flask(__name__)


@app.route("/", methods=["GET"])
def home():
    return recommender()


@app.route("/about/", methods=["GET"])
def about():
    return render_template("about.html")


@app.route("/players/", methods=["GET"])
def players():
    df = pd.read_csv(DATA_PATH)
    df = df.query("season == 2019 and week_no == 8").reset_index(drop=True)
    df = df[COLUMNS_TO_DISPLAY_IN_PLAYERS_WEB]
    df["player_name"] = df.player_name.apply(lambda x: x.replace("_", " "))
    df["last3_ratio_points"] = df.last3_ratio_points.apply(lambda x: round(float(x), 2))
    df["last10_ratio_wins_own"] = df.last10_ratio_wins_own.apply(lambda x: round(float(x), 2))
    df = df.rename(columns={"player_name": "Player Name",
                            "player_team_name": "Team",
                            "player_position": "Position",
                            "last10_ratio_wins_own": "Wins ratio in last 10 games",
                            "last3_assists": "Assist in last 3 games",
                            "last3_goals": "Goals in last 3 games",
                            "last3_ratio_points": "Points ratio in last 3 games",
                            "last3_ycards":"Yellow cards in last 3 games",
                            })
    return render_template("players.html",
                           players=df.to_html())


@app.route("/recommender/", methods=["GET", "POST"])
def recommender():
    has_results = False
    if request.method == "POST":
        week_game_id = request.form.get("week_game_id")
        df = infer(week_number=week_game_id)
        df["player_name"] = df.player_name.apply(lambda x: x.replace("_", " "))
        df["week_points_pred"] = df.week_points_pred.apply(lambda x: round(float(x), 2))
        df = df.rename(columns={"player_name": "Player Name",
                                "player_team_name": "Team",
                                "player_position": "Position",
                                "week_points_pred": "Predicted Points"})
        if week_game_id is not None:
            has_results = True
        message = "" if has_results else "Error while processing"
        return render_template("recommender.html",
                               players=df.head(45).to_html(index=False),
                               message=message,
                               has_results=has_results,
                               week_game_id=week_game_id,
                               week_games=WEEK_GAME_NUMBERS_2019)

    return render_template("recommender.html",
                           has_results=has_results,
                           week_games=WEEK_GAME_NUMBERS_2019)


if __name__ == "__main__":
    random_seed()
    app.run(host="0.0.0.0", port=5000, debug=True)
