"""
IPL Win Predictor - Flask backend

Serves the frontend and exposes a /predict endpoint that wraps the
scikit-learn Pipeline trained in the notebook (pipe.pkl).

Run:
    pip install -r requirements.txt
    python app.py
Then open http://127.0.0.1:5000
"""
import json
import os
import pickle

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load the trained pipeline (produced by the notebook's `pickle.dump(pipe, ...)`
# cell). Place pipe.pkl in this same folder before running the app.
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "pipe.pkl")
pipe = None
model_load_error = None

if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, "rb") as f:
            pipe = pickle.load(f)
    except Exception as exc:  # noqa: BLE001
        model_load_error = str(exc)
else:
    model_load_error = (
        "pipe.pkl not found. Export it from the notebook "
        "(pickle.dump(pipe, open('pipe.pkl','wb'))) and place it in this folder."
    )

# Teams and cities the model was ACTUALLY trained on, pulled straight from
# the fitted OneHotEncoder's categories_ (see train_model.py) so the
# dropdowns can never send a value the model doesn't recognize. Falls back
# to a reasonable default list if teams_cities.json isn't present yet.
CATEGORIES_PATH = os.path.join(os.path.dirname(__file__), "teams_cities.json")

_DEFAULT_TEAMS = [
    "Sunrisers Hyderabad", "Mumbai Indians", "Royal Challengers Bangalore",
    "Kolkata Knight Riders", "Kings XI Punjab", "Chennai Super Kings",
    "Rajasthan Royals", "Delhi Capitals",
]
_DEFAULT_CITIES = sorted([
    "Hyderabad", "Bangalore", "Mumbai", "Indore", "Kolkata", "Delhi",
    "Chandigarh", "Jaipur", "Chennai", "Cape Town", "Port Elizabeth",
    "Durban", "Centurion", "East London", "Johannesburg", "Kimberley",
    "Bloemfontein", "Ahmedabad", "Cuttack", "Nagpur", "Dharamsala",
    "Visakhapatnam", "Pune", "Raipur", "Ranchi", "Abu Dhabi",
    "Sharjah", "Mohali", "Bengaluru",
])

if os.path.exists(CATEGORIES_PATH):
    with open(CATEGORIES_PATH) as f:
        _cats = json.load(f)
    TEAMS = _cats.get("teams", _DEFAULT_TEAMS)
    CITIES = _cats.get("cities", _DEFAULT_CITIES)
else:
    TEAMS = _DEFAULT_TEAMS
    CITIES = _DEFAULT_CITIES


@app.route("/")
def index():
    return render_template(
        "index.html",
        teams=TEAMS,
        cities=CITIES,
        model_ready=pipe is not None,
        model_error=model_load_error,
    )


@app.route("/predict", methods=["POST"])
def predict():
    if pipe is None:
        return jsonify({"error": model_load_error or "Model not loaded."}), 503

    data = request.get_json(force=True)

    try:
        batting_team = data["batting_team"]
        bowling_team = data["bowling_team"]
        city = data["city"]
        target = float(data["target"])
        current_score = float(data["current_score"])
        overs_completed = float(data["overs_completed"])  # e.g. 12.3 = 12 overs, 3 balls
        wickets_down = int(data["wickets_down"])
    except (KeyError, ValueError, TypeError):
        return jsonify({"error": "Missing or invalid input fields."}), 400

    if batting_team == bowling_team:
        return jsonify({"error": "Batting and bowling team must be different."}), 400

    whole_overs = int(overs_completed)
    balls_in_over = round((overs_completed - whole_overs) * 10)
    if balls_in_over < 0 or balls_in_over > 5:
        return jsonify({"error": "Overs must be given as overs.balls, e.g. 12.3 (balls 0-5)."}), 400

    balls_bowled = whole_overs * 6 + balls_in_over
    balls_left = 120 - balls_bowled
    runs_left = target - current_score
    wickets_left = 10 - wickets_down

    if balls_left <= 0:
        return jsonify({"error": "Overs completed must be less than 20."}), 400
    if wickets_left <= 0:
        return jsonify({"error": "All 10 wickets are already down."}), 400
    if runs_left < 0:
        # Chase is already won
        return jsonify({"win": 100.0, "lose": 0.0, "runs_left": 0, "balls_left": balls_left,
                         "wickets_left": wickets_left, "crr": None, "rrr": None})

    crr = (current_score * 6) / balls_bowled if balls_bowled > 0 else 0.0
    rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0.0

    input_df = pd.DataFrame(
        {
            "batting_team": [batting_team],
            "bowling_team": [bowling_team],
            "city": [city],
            "runs_left": [runs_left],
            "balls_left": [balls_left],
            "wickets": [wickets_left],
            "total_runs_x": [target],
            "crr": [crr],
            "rrr": [rrr],
        }
    )

    try:
        result = pipe.predict_proba(input_df)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Prediction failed: {exc}"}), 500

    lose_pct = round(float(result[0][0]) * 100, 1)
    win_pct = round(float(result[0][1]) * 100, 1)

    return jsonify(
        {
            "win": win_pct,
            "lose": lose_pct,
            "runs_left": int(runs_left),
            "balls_left": int(balls_left),
            "wickets_left": int(wickets_left),
            "crr": round(crr, 2),
            "rrr": round(rrr, 2),
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

