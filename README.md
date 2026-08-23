# IPL Win Predictor — Website

A Flask website wrapping the win-probability model from your notebook.

**`pipe.pkl` is already included** — it was trained on your actual
`matches.csv` / `deliveries.csv` (test accuracy ~80%, matching your
notebook). `teams_cities.json` holds the exact team/city categories the
model was trained on, so the dropdowns can't send it a value it doesn't
recognize. If you retrain later, `train_model.py` reproduces both files —
see "Retraining" below.

## 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Run it

```bash
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

## How it works

- `app.py` — Flask backend. Loads `pipe.pkl` once at startup, exposes `/predict`
  (POST, JSON in/out) which reproduces the exact feature engineering from your
  notebook (`runs_left`, `balls_left`, `wickets`, `crr`, `rrr`) before calling
  `pipe.predict_proba(...)`.
- `templates/index.html` + `static/style.css` + `static/script.js` — the
  frontend: a form for the match situation, and a live scoreboard-style bar
  showing win/lose probability.

## Project structure

```
ipl-win-predictor/
├── app.py                 <- Flask backend + /predict API
├── train_model.py         <- reproduces pipe.pkl + teams_cities.json from raw data
├── pipe.pkl                <- trained model (included)
├── teams_cities.json       <- exact team/city categories the model knows (included)
├── requirements.txt
├── templates/
│   └── index.html
└── static/
    ├── style.css
    └── script.js
```

## Retraining

If you get new data, put `matches.csv` and `deliveries.csv` (or `deliveries.csv.zip`)
in this folder and run:

```bash
python train_model.py
```

This overwrites `pipe.pkl` and `teams_cities.json` — the app picks up both
automatically on next restart.

## Notes

- **Heads up on team names**: the notebook's cleanup (`Delhi Daredevils` →
  `Delhi Capitals`, `Deccan Chargers` → `Sunrisers Hyderabad`) only renames
  `matches.csv`'s `team1`/`team2` columns, used to *filter* which matches are
  kept. The `batting_team`/`bowling_team` values that actually get fed to the
  model come straight from `deliveries.csv` and are never renamed — so the
  model was trained on, and the app's dropdowns include, both the old and
  new names as separate categories (e.g. `Deccan Chargers` and `Sunrisers
  Hyderabad` both appear). This is a quirk in the notebook's original logic,
  not something introduced here — worth fixing in the notebook if you retrain.
- "Overs completed" is entered as `overs.balls` (e.g. `12.3` = 12 overs, 3 balls),
  matching how the notebook derives `balls_left`.
- For production use (not just local testing), run behind a proper WSGI server
  (e.g. `gunicorn app:app`) instead of Flask's dev server.
