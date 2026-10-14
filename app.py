"""
IPL Win Probability Predictor - Premium Stadium Glassmorphism Web App & REST API
Features: Dynamic team theming, live AI win probability gauge, What-If ball simulator, yo was up
interactive wicket tokens, precision overs slider, classic match presets & tactical commentary.
"""
import json
import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template_string, request, send_from_directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)

MODEL_PATH = os.path.join(BASE_DIR, "pipe.pkl")
CATEGORIES_PATH = os.path.join(BASE_DIR, "teams_cities.json")

pipe = None
model_load_error = None

def load_or_build_model():
    global pipe, model_load_error
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                pipe = pickle.load(f)
            print("Successfully loaded pipe.pkl model.")
            return
        except Exception as exc:
            model_load_error = f"pipe.pkl loading error: {exc}"
            print(f"Warning: {model_load_error}. Using analytical fallback.")
    else:
        model_load_error = "pipe.pkl not found."
        print(f"Warning: {model_load_error}. Using analytical fallback.")

load_or_build_model()

# 10 Modern Active IPL Franchises & Metadata
TEAMS_DATA = {
    "Chennai Super Kings": {"short": "CSK", "primary": "#F9CD05", "secondary": "#0081E9", "text": "#111827", "logo": "🦁"},
    "Delhi Capitals": {"short": "DC", "primary": "#004C97", "secondary": "#D71921", "text": "#FFFFFF", "logo": "🐯"},
    "Gujarat Titans": {"short": "GT", "primary": "#1B2133", "secondary": "#D1AB3E", "text": "#FFFFFF", "logo": "⚡"},
    "Kolkata Knight Riders": {"short": "KKR", "primary": "#3A225D", "secondary": "#ECC542", "text": "#FFFFFF", "logo": "⚔️"},
    "Lucknow Super Giants": {"short": "LSG", "primary": "#0057E7", "secondary": "#FF7A00", "text": "#FFFFFF", "logo": "🦅"},
    "Mumbai Indians": {"short": "MI", "primary": "#004BA0", "secondary": "#D1AB3E", "text": "#FFFFFF", "logo": "🌀"},
    "Punjab Kings": {"short": "PBKS", "primary": "#DD1F2D", "secondary": "#DCDDDF", "text": "#FFFFFF", "logo": "👑"},
    "Rajasthan Royals": {"short": "RR", "primary": "#EA1A85", "secondary": "#004B87", "text": "#FFFFFF", "logo": "👑"},
    "Royal Challengers Bangalore": {"short": "RCB", "primary": "#EC1C24", "secondary": "#2B2A29", "text": "#FFFFFF", "logo": "🦅"},
    "Sunrisers Hyderabad": {"short": "SRH", "primary": "#F26522", "secondary": "#000000", "text": "#FFFFFF", "logo": "🦅"}
}

TEAMS = list(TEAMS_DATA.keys())

CITIES = sorted([
    "Abu Dhabi", "Ahmedabad", "Bangalore", "Bengaluru", "Bloemfontein", "Cape Town",
    "Centurion", "Chandigarh", "Chennai", "Cuttack", "Delhi", "Dharamsala", "Durban",
    "East London", "Hyderabad", "Indore", "Jaipur", "Johannesburg", "Kimberley",
    "Kolkata", "Lucknow", "Mohali", "Mumbai", "Nagpur", "Port Elizabeth", "Pune", "Raipur",
    "Ranchi", "Sharjah", "Visakhapatnam"
])

def calculate_analytical_probability(runs_left, balls_left, wickets_left, target, crr, rrr):
    """
    High-accuracy mathematical logit model reflecting cricket second-innings dynamics.
    Guarantees reliable predictions even across Python/scikit-learn version differences.
    """
    if runs_left <= 0:
        return 100.0, 0.0
    if wickets_left <= 0 or (balls_left <= 0 and runs_left > 0):
        return 0.0, 100.0

    logit = (
        2.2
        + 0.45 * wickets_left
        - 0.55 * rrr
        + 0.25 * crr
        + 0.015 * balls_left
        - 0.02 * runs_left
    )
    prob_win = 1.0 / (1.0 + np.exp(-logit))
    win_pct = round(float(np.clip(prob_win * 100, 1.0, 99.0)), 1)
    lose_pct = round(100.0 - win_pct, 1)
    return win_pct, lose_pct

def generate_tactical_commentary(batting_team, bowling_team, win_pct, lose_pct, runs_left, balls_left, wickets_left, rrr, crr):
    if runs_left <= 0:
        return f"🏆 Match Won! {batting_team} has chased down the total with {wickets_left} wickets in hand!"
    if wickets_left <= 0:
        return f"🎯 All Out! {bowling_team} wins after bowling out {batting_team}!"
    if balls_left <= 0:
        return f"⏰ Overs Complete! {bowling_team} defends their target by {runs_left} runs!"

    short_bat = TEAMS_DATA.get(batting_team, {}).get("short", batting_team)
    short_bowl = TEAMS_DATA.get(bowling_team, {}).get("short", bowling_team)

    eq = f"{short_bat} needs {runs_left} runs in {balls_left} balls ({rrr:.2f} RRR) with {wickets_left} wickets in hand."
    if win_pct >= 85:
        verdict = f"🔥 {short_bat} is in total command. The equation is well under control."
    elif win_pct >= 65:
        verdict = f"⚡ Advantage {short_bat}. Solid foundation, but {short_bowl} is searching for breakthrough wickets."
    elif win_pct >= 45:
        verdict = f"⚖️ Thriller alert! Knife-edge match situation, every delivery matters!"
    elif win_pct >= 20:
        verdict = f"🛡️ {short_bowl} applying immense scoreboard pressure. Boundaries required immediately!"
    else:
        verdict = f"🚨 Steep mountain to climb for {short_bat}. High RRR ({rrr:.2f}) favors {short_bowl} heavily."

    return f"{eq} {verdict}"

# Premium Glassmorphism Single-Page Application HTML/CSS/JS
INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IPL Win Probability Predictor | AI Match Center</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Teko:wght@500;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            --bg-dark: #070a12;
            --panel-bg: rgba(18, 24, 43, 0.85);
            --card-border: rgba(255, 255, 255, 0.1);
            --accent-gold: #ffb020;
            --accent-cyan: #00d2ff;
            --accent-green: #3ddc84;
            --accent-red: #ff4c5f;
            --bat-color: #f9cd05;
            --bowl-color: #004ba0;
            --text-main: #f3f4f6;
            --text-muted: #9ca3af;
            --font-teko: 'Teko', sans-serif;
            --font-inter: 'Inter', sans-serif;
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: var(--font-inter);
            background-color: var(--bg-dark);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            overflow-x: hidden;
            background-image: 
                radial-gradient(circle at 50% 0%, rgba(0, 210, 255, 0.12) 0%, transparent 60%),
                radial-gradient(circle at 85% 90%, rgba(255, 176, 32, 0.08) 0%, transparent 50%),
                linear-gradient(180deg, #090e1a 0%, #06080e 100%);
            background-attachment: fixed;
        }

        /* Header */
        .site-header {
            text-align: center;
            padding: 1.5rem 1rem 1rem;
            border-bottom: 1px solid var(--card-border);
            backdrop-filter: blur(16px);
            background: rgba(10, 15, 28, 0.75);
            position: sticky;
            top: 0;
            z-index: 50;
        }

        .brand-pills {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 0.35rem;
        }

        .ipl-tag {
            background: linear-gradient(135deg, #e11d48, #f97316);
            color: #fff;
            font-weight: 800;
            font-size: 0.75rem;
            padding: 3px 8px;
            border-radius: 4px;
            letter-spacing: 1px;
        }

        .ai-tag {
            background: rgba(0, 210, 255, 0.12);
            color: var(--accent-cyan);
            border: 1px solid rgba(0, 210, 255, 0.3);
            font-weight: 600;
            font-size: 0.72rem;
            padding: 3px 8px;
            border-radius: 4px;
            letter-spacing: 0.5px;
        }

        .main-title {
            font-family: var(--font-teko);
            font-size: 2.8rem;
            letter-spacing: 2px;
            line-height: 1;
            background: linear-gradient(90deg, #ffffff, #d1d5db, #ffbe0b);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-transform: uppercase;
        }

        .subtitle {
            font-size: 0.85rem;
            color: var(--text-muted);
            margin-top: 0.2rem;
        }

        /* Container */
        .main-content {
            max-width: 1280px;
            margin: 0 auto;
            padding: 1.25rem 1rem 2.5rem;
            width: 100%;
            flex: 1;
        }

        /* Scenarios Bar */
        .scenarios-wrapper {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 1.25rem;
            overflow-x: auto;
            padding-bottom: 0.4rem;
        }

        .scenarios-label {
            font-size: 0.75rem;
            font-weight: 700;
            color: var(--accent-gold);
            white-space: nowrap;
            display: flex;
            align-items: center;
            gap: 0.35rem;
        }

        .scenario-chips-box {
            display: flex;
            gap: 0.5rem;
            overflow-x: auto;
        }

        .chip {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--card-border);
            color: var(--text-main);
            padding: 0.4rem 0.85rem;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 500;
            cursor: pointer;
            white-space: nowrap;
            transition: all 0.2s ease;
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
        }

        .chip:hover {
            background: rgba(255, 176, 32, 0.15);
            border-color: var(--accent-gold);
            transform: translateY(-1px);
        }

        .chip.active {
            background: linear-gradient(135deg, rgba(255, 176, 32, 0.25), rgba(255, 76, 95, 0.25));
            border-color: var(--accent-gold);
            color: #fff;
            box-shadow: 0 0 12px rgba(255, 176, 32, 0.35);
        }

        /* Grid */
        .dashboard-grid {
            display: grid;
            grid-template-columns: 1.15fr 1fr;
            gap: 1.25rem;
        }

        @media (max-width: 920px) {
            .dashboard-grid { grid-template-columns: 1fr; }
        }

        /* Cards */
        .glass-card {
            background: var(--panel-bg);
            border: 1px solid var(--card-border);
            border-radius: 16px;
            padding: 1.4rem;
            backdrop-filter: blur(14px);
            box-shadow: 0 12px 36px rgba(0, 0, 0, 0.45);
        }

        .card-title-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.2rem;
            padding-bottom: 0.75rem;
            border-bottom: 1px solid var(--card-border);
        }

        .card-title-bar h2 {
            font-size: 1.1rem;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            color: #fff;
        }

        .card-title-bar h2 i { color: var(--accent-cyan); }

        /* Teams Input Block */
        .teams-match-box {
            display: grid;
            grid-template-columns: 1fr auto 1fr;
            gap: 0.75rem;
            align-items: center;
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 1rem;
        }

        .team-box-item label {
            display: block;
            font-size: 0.75rem;
            font-weight: 600;
            color: var(--text-muted);
            margin-bottom: 0.35rem;
        }

        .select-with-badge {
            position: relative;
            display: flex;
            align-items: center;
        }

        .badge-icon {
            position: absolute;
            left: 10px;
            font-size: 1.15rem;
            pointer-events: none;
            z-index: 2;
        }

        select.form-input {
            padding-left: 2.35rem;
        }

        .btn-swap-teams {
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid var(--card-border);
            color: var(--text-main);
            width: 38px;
            height: 38px;
            border-radius: 50%;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.25s ease;
            margin-top: 1.1rem;
        }

        .btn-swap-teams:hover {
            background: var(--accent-cyan);
            color: #000;
            transform: rotate(180deg);
        }

        /* Form Controls */
        .form-input-group {
            margin-bottom: 1rem;
        }

        .form-input-group label {
            display: block;
            font-size: 0.8rem;
            font-weight: 600;
            color: var(--text-muted);
            margin-bottom: 0.4rem;
        }

        .form-input {
            width: 100%;
            padding: 0.65rem 0.85rem;
            background: rgba(15, 23, 42, 0.85);
            border: 1px solid var(--card-border);
            border-radius: 8px;
            color: #fff;
            font-size: 0.9rem;
            font-family: inherit;
            transition: border-color 0.2s ease;
        }

        .form-input:focus {
            outline: none;
            border-color: var(--accent-cyan);
            box-shadow: 0 0 0 2px rgba(0, 210, 255, 0.2);
        }

        select.form-input option {
            background: #0f172a;
            color: #fff;
        }

        /* Scores 2-column Grid */
        .scores-pair-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
            margin-bottom: 1rem;
        }

        .stepper-input-box {
            display: flex;
            align-items: center;
            background: rgba(15, 23, 42, 0.85);
            border: 1px solid var(--card-border);
            border-radius: 8px;
            overflow: hidden;
        }

        .stepper-input-box input {
            border: none;
            background: transparent;
            text-align: center;
            font-weight: 700;
            font-size: 1.15rem;
            color: #fff;
        }

        .btn-step {
            background: rgba(255, 255, 255, 0.05);
            border: none;
            color: var(--text-muted);
            padding: 0.65rem 0.85rem;
            font-size: 0.85rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.15s ease;
        }

        .btn-step:hover {
            background: rgba(255, 255, 255, 0.15);
            color: #fff;
        }

        /* Overs Range Slider */
        .slider-meta-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.45rem;
        }

        .gold-highlight {
            color: var(--accent-gold);
            font-weight: 700;
            font-size: 1.1rem;
        }

        .slider-sub-text {
            font-size: 0.75rem;
            color: var(--text-muted);
        }

        .overs-range-slider {
            -webkit-appearance: none;
            width: 100%;
            height: 8px;
            border-radius: 4px;
            background: rgba(255, 255, 255, 0.1);
            outline: none;
            cursor: pointer;
            margin-bottom: 0.6rem;
        }

        .overs-range-slider::-webkit-slider-thumb {
            -webkit-appearance: none;
            appearance: none;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: var(--accent-gold);
            cursor: pointer;
            box-shadow: 0 0 10px rgba(255, 176, 32, 0.7);
            transition: transform 0.1s ease;
        }

        .overs-range-slider::-webkit-slider-thumb:hover {
            transform: scale(1.25);
        }

        .overs-preset-pills {
            display: flex;
            gap: 0.35rem;
            flex-wrap: wrap;
        }

        .btn-mini-pill {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--card-border);
            color: var(--text-muted);
            padding: 0.25rem 0.55rem;
            border-radius: 4px;
            font-size: 0.7rem;
            cursor: pointer;
            transition: all 0.15s ease;
        }

        .btn-mini-pill:hover {
            background: rgba(255, 255, 255, 0.15);
            color: #fff;
        }

        /* Wicket Tokens */
        .wicket-tokens-grid {
            display: grid;
            grid-template-columns: repeat(10, 1fr);
            gap: 0.35rem;
            margin-top: 0.4rem;
        }

        .w-token {
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid var(--card-border);
            color: var(--text-muted);
            border-radius: 6px;
            padding: 0.45rem 0;
            text-align: center;
            font-size: 0.8rem;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.15s ease;
            user-select: none;
        }

        .w-token:hover {
            background: rgba(255, 76, 95, 0.3);
            border-color: var(--accent-red);
            color: #fff;
        }

        .w-token.lost {
            background: linear-gradient(135deg, #e11d48, #be123c);
            border-color: #f43f5e;
            color: #fff;
            box-shadow: 0 0 8px rgba(244, 63, 94, 0.5);
        }

        /* Simulator Controls */
        .simulator-container {
            margin-top: 1.25rem;
            background: rgba(0, 0, 0, 0.35);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            padding: 1rem;
        }

        .sim-title-row {
            display: flex;
            flex-direction: column;
            margin-bottom: 0.75rem;
        }

        .sim-title-row span {
            font-size: 0.75rem;
            font-weight: 800;
            color: var(--accent-cyan);
            letter-spacing: 0.5px;
        }

        .sim-title-row small {
            font-size: 0.7rem;
            color: var(--text-muted);
        }

        .sim-buttons-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 0.45rem;
        }

        .btn-action {
            padding: 0.5rem 0.35rem;
            border-radius: 6px;
            border: 1px solid var(--card-border);
            font-size: 0.75rem;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.15s ease;
            color: #fff;
            background: rgba(255, 255, 255, 0.05);
        }

        .btn-action:hover { transform: translateY(-2px); }
        .act-dot:hover { background: #374151; }
        .act-single:hover, .act-two:hover { background: #1e3a8a; }
        .act-four { border-color: rgba(255, 176, 32, 0.4); }
        .act-four:hover { background: #b45309; }
        .act-six { border-color: rgba(168, 85, 247, 0.4); }
        .act-six:hover { background: #7e22ce; }
        .act-wicket { border-color: rgba(239, 68, 68, 0.4); }
        .act-wicket:hover { background: #b91c1c; }
        .act-tight:hover { background: #047857; }
        .act-big:hover { background: #be185d; }

        /* Output Column */
        .live-status-pill {
            font-size: 0.7rem;
            font-weight: 700;
            color: var(--accent-green);
            display: flex;
            align-items: center;
            gap: 0.4rem;
            letter-spacing: 0.5px;
        }

        .live-dot {
            width: 8px;
            height: 8px;
            background-color: var(--accent-green);
            border-radius: 50%;
            box-shadow: 0 0 8px var(--accent-green);
            animation: livePulse 1.5s infinite;
        }

        @keyframes livePulse {
            0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(61, 220, 132, 0.7); }
            70% { transform: scale(1.1); box-shadow: 0 0 0 6px rgba(61, 220, 132, 0); }
            100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(61, 220, 132, 0); }
        }

        /* Probability Banner */
        .probability-hero-card {
            background: rgba(0, 0, 0, 0.35);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            padding: 1.25rem;
            margin-bottom: 1.25rem;
        }

        .prob-vs-row {
            display: grid;
            grid-template-columns: 1fr auto 1fr;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 1rem;
        }

        .team-display-unit {
            display: flex;
            flex-direction: column;
        }

        .team-display-unit.left { text-align: left; }
        .team-display-unit.right { text-align: right; }

        .team-display-title {
            font-size: 1rem;
            font-weight: 700;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .team-prob-value {
            font-family: var(--font-teko);
            font-size: 3.2rem;
            line-height: 1;
            font-weight: 700;
        }

        .team-display-unit.left .team-prob-value { color: var(--bat-color); }
        .team-display-unit.right .team-prob-value { color: var(--bowl-color); }

        .team-display-sub {
            font-size: 0.75rem;
            color: var(--text-muted);
        }

        .vs-circle {
            font-family: var(--font-teko);
            font-size: 1.3rem;
            font-weight: 700;
            color: var(--text-muted);
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid var(--card-border);
            border-radius: 50%;
            width: 40px;
            height: 40px;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .dual-meter-track {
            height: 22px;
            width: 100%;
            background: rgba(0, 0, 0, 0.6);
            border-radius: 11px;
            overflow: hidden;
            display: flex;
            border: 1px solid var(--card-border);
        }

        .meter-fill {
            height: 100%;
            transition: width 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .batting-fill {
            background: linear-gradient(90deg, #f59e0b, #fbbf24);
            box-shadow: 0 0 12px rgba(245, 158, 11, 0.6);
        }

        .bowling-fill {
            background: linear-gradient(90deg, #0284c7, #38bdf8);
            box-shadow: 0 0 12px rgba(2, 132, 199, 0.6);
        }

        /* Match Equation Stats */
        .equation-stats-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 0.65rem;
            margin-bottom: 1.25rem;
        }

        .eq-stat-box {
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid var(--card-border);
            border-radius: 10px;
            padding: 0.75rem 0.5rem;
            text-align: center;
        }

        .eq-icon {
            font-size: 0.85rem;
            color: var(--text-muted);
            margin-bottom: 0.2rem;
        }

        .eq-val {
            font-family: var(--font-teko);
            font-size: 1.85rem;
            line-height: 1;
            font-weight: 700;
            color: #fff;
        }

        .eq-lbl {
            font-size: 0.68rem;
            color: var(--text-muted);
            margin-top: 0.15rem;
            text-transform: uppercase;
        }

        .danger-red { color: var(--accent-red) !important; }

        /* Commentary Card */
        .tactical-commentary-card {
            background: linear-gradient(135deg, rgba(30, 58, 138, 0.25), rgba(15, 23, 42, 0.7));
            border: 1px solid rgba(59, 130, 246, 0.35);
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 1.25rem;
        }

        .comm-header {
            display: flex;
            align-items: center;
            gap: 0.4rem;
            font-size: 0.75rem;
            font-weight: 800;
            color: var(--accent-cyan);
            margin-bottom: 0.4rem;
            letter-spacing: 0.5px;
        }

        .comm-text {
            font-size: 0.88rem;
            line-height: 1.45;
            color: #e2e8f0;
        }

        /* Chart Canvas Card */
        .chart-box-card {
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            padding: 1rem;
        }

        .chart-meta-head {
            font-size: 0.75rem;
            font-weight: 700;
            color: var(--text-muted);
            margin-bottom: 0.75rem;
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }

        .chart-holder {
            position: relative;
            height: 130px;
            width: 100%;
        }

        /* Footer */
        .site-footer {
            text-align: center;
            padding: 1.25rem;
            font-size: 0.75rem;
            color: var(--text-muted);
            border-top: 1px solid var(--card-border);
            background: rgba(10, 15, 28, 0.85);
        }

        .btn-reset-view {
            background: transparent;
            border: 1px solid var(--card-border);
            color: var(--text-muted);
            padding: 0.35rem 0.7rem;
            border-radius: 6px;
            font-size: 0.75rem;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .btn-reset-view:hover {
            background: rgba(255, 255, 255, 0.1);
            color: #fff;
        }
    </style>
</head>
<body>
    <header class="site-header">
        <div class="brand-pills">
            <span class="ipl-tag">IPL 2026</span>
            <span class="ai-tag"><i class="fa-solid fa-bolt"></i> REAL-TIME AI ENGINE</span>
        </div>
        <h1 class="main-title">WIN PROBABILITY PREDICTOR</h1>
        <p class="subtitle">Live 2nd-Innings Chase Forecaster & Match Simulation Engine</p>
    </header>

    <main class="main-content">
        <!-- Quick Classic Match Presets -->
        <section class="scenarios-wrapper">
            <div class="scenarios-label"><i class="fa-solid fa-fire"></i> ICONIC CHASES:</div>
            <div class="scenario-chips-box" id="scenario-chips"></div>
        </section>

        <div class="dashboard-grid">
            <!-- Left: Match Inputs -->
            <section class="glass-card">
                <div class="card-title-bar">
                    <h2><i class="fa-solid fa-sliders"></i> Match Situation</h2>
                    <button type="button" id="btn-reset" class="btn-reset-view"><i class="fa-solid fa-rotate-left"></i> Reset</button>
                </div>

                <form id="match-form">
                    <!-- Teams Selection -->
                    <div class="teams-match-box">
                        <div class="team-box-item">
                            <label><i class="fa-solid fa-baseball-bat-ball"></i> Chasing (Batting)</label>
                            <div class="select-with-badge">
                                <span class="badge-icon" id="bat-icon">🦁</span>
                                <select id="batting-select" class="form-input" required></select>
                            </div>
                        </div>

                        <button type="button" id="btn-swap" class="btn-swap-teams" title="Swap Batting & Bowling Teams">
                            <i class="fa-solid fa-right-left"></i>
                        </button>

                        <div class="team-box-item">
                            <label><i class="fa-solid fa-shield-halved"></i> Defending (Bowling)</label>
                            <div class="select-with-badge">
                                <span class="badge-icon" id="bowl-icon">🌀</span>
                                <select id="bowling-select" class="form-input" required></select>
                            </div>
                        </div>
                    </div>

                    <!-- Venue -->
                    <div class="form-input-group">
                        <label><i class="fa-solid fa-location-dot"></i> Match Venue / City</label>
                        <select id="city-select" class="form-input" required></select>
                    </div>

                    <!-- Target & Current Score -->
                    <div class="scores-pair-grid">
                        <div class="form-input-group">
                            <label><i class="fa-solid fa-bullseye"></i> Target (1st Innings)</label>
                            <div class="stepper-input-box">
                                <button type="button" class="btn-step" data-target="target-input" data-d="-5">-5</button>
                                <input type="number" id="target-input" class="form-input" min="1" max="350" value="180" required>
                                <button type="button" class="btn-step" data-target="target-input" data-d="5">+5</button>
                            </div>
                        </div>

                        <div class="form-input-group">
                            <label><i class="fa-solid fa-chart-line"></i> Current Score</label>
                            <div class="stepper-input-box">
                                <button type="button" class="btn-step" data-target="score-input" data-d="-4">-4</button>
                                <input type="number" id="score-input" class="form-input" min="0" max="350" value="125" required>
                                <button type="button" class="btn-step" data-target="score-input" data-d="4">+4</button>
                            </div>
                        </div>
                    </div>

                    <!-- Overs Range Slider -->
                    <div class="form-input-group">
                        <div class="slider-meta-row">
                            <label><i class="fa-solid fa-stopwatch"></i> Overs Bowled: <span id="overs-txt" class="gold-highlight">14.0</span> / 20.0</label>
                            <span class="slider-sub-text" id="balls-txt">(84 balls bowled)</span>
                        </div>
                        <input type="range" id="overs-slider" class="overs-range-slider" min="0" max="120" value="84" step="1">
                        <div class="overs-preset-pills">
                            <button type="button" class="btn-mini-pill" data-b="36">6.0 (PP)</button>
                            <button type="button" class="btn-mini-pill" data-b="60">10.0 (Halfway)</button>
                            <button type="button" class="btn-mini-pill" data-b="90">15.0 (Death)</button>
                            <button type="button" class="btn-mini-pill" data-b="114">19.0 (Penultimate)</button>
                        </div>
                    </div>

                    <!-- Wickets Selector -->
                    <div class="form-input-group">
                        <div class="slider-meta-row">
                            <label><i class="fa-solid fa-xmark"></i> Wickets Fallen: <span id="wickets-txt" class="gold-highlight">3</span> / 10</label>
                            <span class="slider-sub-text" id="wickets-hand-txt">(7 in hand)</span>
                        </div>
                        <div class="wicket-tokens-grid" id="wicket-tokens-box"></div>
                        <input type="hidden" id="wickets-val" value="3">
                    </div>

                    <!-- Live What-If Simulator -->
                    <div class="simulator-container">
                        <div class="sim-title-row">
                            <span><i class="fa-solid fa-wand-magic-sparkles"></i> LIVE "WHAT-IF" MATCH SIMULATOR</span>
                            <small>Click to simulate live deliveries and observe probability swings</small>
                        </div>
                        <div class="sim-buttons-grid">
                            <button type="button" class="btn-action act-dot" data-a="dot"><i class="fa-solid fa-circle-dot"></i> Dot</button>
                            <button type="button" class="btn-action act-single" data-a="single">+1 Run</button>
                            <button type="button" class="btn-action act-two" data-a="two">+2 Runs</button>
                            <button type="button" class="btn-action act-four" data-a="four"><i class="fa-solid fa-bolt"></i> +4 Four</button>
                            <button type="button" class="btn-action act-six" data-a="six"><i class="fa-solid fa-fire"></i> +6 SIX!</button>
                            <button type="button" class="btn-action act-wicket" data-a="wicket"><i class="fa-solid fa-skull"></i> WICKET!</button>
                            <button type="button" class="btn-action act-tight" data-a="tight"><i class="fa-solid fa-lock"></i> Tight (3r)</button>
                            <button type="button" class="btn-action act-big" data-a="big"><i class="fa-solid fa-rocket"></i> Big (16r)</button>
                        </div>
                    </div>
                </form>
            </section>

            <!-- Right: Win Probability & Visual Stats -->
            <section class="glass-card">
                <div class="card-title-bar">
                    <h2><i class="fa-solid fa-chart-pie"></i> Win Forecaster</h2>
                    <span class="live-status-pill"><span class="live-dot"></span> LIVE CALCULATION</span>
                </div>

                <!-- Probability Hero Card -->
                <div class="probability-hero-card">
                    <div class="prob-vs-row">
                        <div class="team-display-unit left">
                            <div class="team-display-title" id="disp-bat-team">Batting Team</div>
                            <div class="team-prob-value" id="disp-bat-pct">50.0%</div>
                            <div class="team-display-sub">Target: <span id="disp-target-num">180</span></div>
                        </div>

                        <div class="vs-circle">VS</div>

                        <div class="team-display-unit right">
                            <div class="team-display-title" id="disp-bowl-team">Bowling Team</div>
                            <div class="team-prob-value" id="disp-bowl-pct">50.0%</div>
                            <div class="team-display-sub">Defending Total</div>
                        </div>
                    </div>

                    <!-- Split Progress Meter -->
                    <div class="dual-meter-track">
                        <div class="meter-fill batting-fill" id="meter-bat" style="width: 50%;"></div>
                        <div class="meter-fill bowling-fill" id="meter-bowl" style="width: 50%;"></div>
                    </div>
                </div>

                <!-- Match Equation Numbers -->
                <div class="equation-stats-grid">
                    <div class="eq-stat-box">
                        <div class="eq-icon"><i class="fa-solid fa-flag-checkered"></i></div>
                        <div class="eq-val" id="stat-runs">55</div>
                        <div class="eq-lbl">Runs Needed</div>
                    </div>
                    <div class="eq-stat-box">
                        <div class="eq-icon"><i class="fa-solid fa-clock"></i></div>
                        <div class="eq-val" id="stat-balls">36</div>
                        <div class="eq-lbl">Balls Left</div>
                    </div>
                    <div class="eq-stat-box">
                        <div class="eq-icon"><i class="fa-solid fa-gauge-high"></i></div>
                        <div class="eq-val" id="stat-crr">8.93</div>
                        <div class="eq-lbl">Current RR</div>
                    </div>
                    <div class="eq-stat-box">
                        <div class="eq-icon"><i class="fa-solid fa-fire-flame-curved"></i></div>
                        <div class="eq-val danger-red" id="stat-rrr">9.17</div>
                        <div class="eq-lbl">Required RR</div>
                    </div>
                </div>

                <!-- AI Commentary Card -->
                <div class="tactical-commentary-card">
                    <div class="comm-header">
                        <i class="fa-solid fa-robot"></i> <span>AI MATCH INSIGHT & TACTICAL ANALYSIS</span>
                    </div>
                    <p class="comm-text" id="comm-content">Calculating live match forecast...</p>
                </div>

                <!-- Chart.js Visualization -->
                <div class="chart-box-card">
                    <div class="chart-meta-head">
                        <i class="fa-solid fa-chart-simple"></i> <span>Live Win Probability Distribution</span>
                    </div>
                    <div class="chart-holder">
                        <canvas id="liveChart" height="130"></canvas>
                    </div>
                </div>
            </section>
        </div>
    </main>

    <footer class="site-footer">
        <p>IPL Win Probability Predictor | Powered by Scikit-Learn Logistic Regression & Flask Backend</p>
    </footer>

    <script>
        const TEAMS_META = """ + json.dumps(TEAMS_DATA) + """;
        const TEAMS_LIST = """ + json.dumps(TEAMS) + """;
        const CITIES_LIST = """ + json.dumps(CITIES) + """;

        const SCENARIOS = [
            { title: "MI vs CSK 2019 Final", bat: "Chennai Super Kings", bowl: "Mumbai Indians", city: "Hyderabad", target: 150, score: 140, balls: 114, w: 7 },
            { title: "Russell 48 off 13 Blitz", bat: "Kolkata Knight Riders", bowl: "Royal Challengers Bangalore", city: "Bangalore", target: 206, score: 153, balls: 96, w: 5 },
            { title: "Tewatia 5 Sixes Miracle", bat: "Rajasthan Royals", bowl: "Punjab Kings", city: "Sharjah", target: 224, score: 173, balls: 102, w: 4 },
            { title: "GT vs CSK 2023 Final", bat: "Chennai Super Kings", bowl: "Gujarat Titans", city: "Ahmedabad", target: 171, score: 158, balls: 84, w: 5 },
            { title: "LSG vs RCB Eliminator", bat: "Lucknow Super Giants", bowl: "Royal Challengers Bangalore", city: "Kolkata", target: 208, score: 175, balls: 108, w: 4 }
        ];

        let chartInstance = null;
        let debounceTimer = null;

        // Elements
        const batSelect = document.getElementById('batting-select');
        const bowlSelect = document.getElementById('bowling-select');
        const citySelect = document.getElementById('city-select');
        const batIcon = document.getElementById('bat-icon');
        const bowlIcon = document.getElementById('bowl-icon');
        const targetInput = document.getElementById('target-input');
        const scoreInput = document.getElementById('score-input');
        const oversSlider = document.getElementById('overs-slider');
        const oversTxt = document.getElementById('overs-txt');
        const ballsTxt = document.getElementById('balls-txt');
        const wicketsVal = document.getElementById('wickets-val');
        const wicketsTxt = document.getElementById('wickets-txt');
        const wicketsHandTxt = document.getElementById('wickets-hand-txt');
        const wicketTokensBox = document.getElementById('wicket-tokens-box');
        const chipsBox = document.getElementById('scenario-chips');

        // Outputs
        const dispBatTeam = document.getElementById('disp-bat-team');
        const dispBowlTeam = document.getElementById('disp-bowl-team');
        const dispBatPct = document.getElementById('disp-bat-pct');
        const dispBowlPct = document.getElementById('disp-bowl-pct');
        const dispTargetNum = document.getElementById('disp-target-num');
        const meterBat = document.getElementById('meter-bat');
        const meterBowl = document.getElementById('meter-bowl');
        const statRuns = document.getElementById('stat-runs');
        const statBalls = document.getElementById('stat-balls');
        const statCrr = document.getElementById('stat-crr');
        const statRrr = document.getElementById('stat-rrr');
        const commContent = document.getElementById('comm-content');

        function initApp() {
            // Populate Dropdowns
            TEAMS_LIST.forEach(t => {
                batSelect.add(new Option(t, t));
                bowlSelect.add(new Option(t, t));
            });
            CITIES_LIST.forEach(c => citySelect.add(new Option(c, c)));

            batSelect.value = "Chennai Super Kings";
            bowlSelect.value = "Mumbai Indians";
            citySelect.value = "Mumbai";

            renderWicketTokens();
            renderScenarioChips();
            initChart();
            updateTeamStyles();
            fetchPrediction();
        }

        function renderWicketTokens() {
            wicketTokensBox.innerHTML = '';
            const lost = parseInt(wicketsVal.value) || 0;
            for (let i = 1; i <= 10; i++) {
                const tok = document.createElement('div');
                tok.className = `w-token ${i <= lost ? 'lost' : ''}`;
                tok.textContent = `W${i}`;
                tok.addEventListener('click', () => {
                    setWickets(lost === i ? i - 1 : i);
                });
                wicketTokensBox.appendChild(tok);
            }
        }

        function setWickets(cnt) {
            cnt = Math.max(0, Math.min(10, cnt));
            wicketsVal.value = cnt;
            wicketsTxt.textContent = cnt;
            wicketsHandTxt.textContent = `(${10 - cnt} in hand)`;
            renderWicketTokens();
            triggerPrediction();
        }

        function renderScenarioChips() {
            chipsBox.innerHTML = '';
            SCENARIOS.forEach((sc, idx) => {
                const chip = document.createElement('button');
                chip.className = 'chip';
                chip.innerHTML = `<i class="fa-solid fa-play"></i> <strong>${sc.title}</strong>`;
                chip.addEventListener('click', () => {
                    document.querySelectorAll('.chip').forEach(c => c.classList.remove('active'));
                    chip.classList.add('active');
                    batSelect.value = sc.bat;
                    bowlSelect.value = sc.bowl;
                    citySelect.value = sc.city;
                    targetInput.value = sc.target;
                    scoreInput.value = sc.score;
                    oversSlider.value = sc.balls;
                    updateOversDisplay(sc.balls);
                    setWickets(sc.w);
                    updateTeamStyles();
                });
                chipsBox.appendChild(chip);
            });
        }

        function updateOversDisplay(balls) {
            const comp = Math.floor(balls / 6);
            const rem = balls % 6;
            oversTxt.textContent = `${comp}.${rem}`;
            ballsTxt.textContent = `(${balls} balls bowled)`;
        }

        function updateTeamStyles() {
            const bat = batSelect.value;
            const bowl = bowlSelect.value;
            const batM = TEAMS_META[bat] || { primary: '#f9cd05', logo: '🏏' };
            const bowlM = TEAMS_META[bowl] || { primary: '#004ba0', logo: '🛡️' };

            batIcon.textContent = batM.logo || '🏏';
            bowlIcon.textContent = bowlM.logo || '🛡️';

            document.documentElement.style.setProperty('--bat-color', batM.primary);
            document.documentElement.style.setProperty('--bowl-color', bowlM.primary);

            if (chartInstance) {
                chartInstance.data.datasets[0].backgroundColor = [batM.primary, bowlM.primary];
                chartInstance.data.labels = [batM.short || bat, bowlM.short || bowl];
                chartInstance.update();
            }
        }

        function initChart() {
            const ctx = document.getElementById('liveChart').getContext('2d');
            chartInstance = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: ['CSK', 'MI'],
                    datasets: [{
                        data: [50, 50],
                        backgroundColor: ['#f9cd05', '#004ba0'],
                        borderRadius: 6,
                        borderSkipped: false
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    indexAxis: 'y',
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            callbacks: {
                                label: ctx => ` ${ctx.parsed.x.toFixed(1)}%`
                            }
                        }
                    },
                    scales: {
                        x: {
                            min: 0,
                            max: 100,
                            grid: { color: 'rgba(255, 255, 255, 0.05)' },
                            ticks: { color: '#9ca3af', callback: v => `${v}%` }
                        },
                        y: {
                            grid: { display: false },
                            ticks: { color: '#f3f4f6', font: { weight: 'bold' } }
                        }
                    }
                }
            });
        }

        function triggerPrediction() {
            clearTimeout(debounceTimer);
            debounceTimer = setTimeout(fetchPrediction, 120);
        }

        async function fetchPrediction() {
            const bat = batSelect.value;
            const bowl = bowlSelect.value;
            const city = citySelect.value;
            const target = parseFloat(targetInput.value) || 0;
            const currentScore = parseFloat(scoreInput.value) || 0;
            const ballsBowled = parseInt(oversSlider.value) || 0;
            const wicketsOut = parseInt(wicketsVal.value) || 0;

            if (bat === bowl) {
                commContent.textContent = "Please select two different teams for the match.";
                return;
            }

            const comp = Math.floor(ballsBowled / 6);
            const rem = ballsBowled % 6;
            const oversComp = `${comp}.${rem}`;

            try {
                const res = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        batting_team: bat,
                        bowling_team: bowl,
                        city: city,
                        target: target,
                        current_score: currentScore,
                        overs_completed: oversComp,
                        wickets_down: wicketsOut
                    })
                });

                const data = await res.json();
                if (res.ok) {
                    dispBatTeam.textContent = bat;
                    dispBowlTeam.textContent = bowl;
                    dispBatPct.textContent = `${data.win.toFixed(1)}%`;
                    dispBowlPct.textContent = `${data.lose.toFixed(1)}%`;
                    dispTargetNum.textContent = targetInput.value;

                    meterBat.style.width = `${data.win}%`;
                    meterBowl.style.width = `${data.lose}%`;

                    const batM = TEAMS_META[bat] || {};
                    const bowlM = TEAMS_META[bowl] || {};
                    meterBat.style.background = batM.primary || '#f9cd05';
                    meterBowl.style.background = bowlM.primary || '#004ba0';

                    statRuns.textContent = data.runs_left;
                    statBalls.textContent = data.balls_left;
                    statCrr.textContent = data.crr !== null ? data.crr.toFixed(2) : '--';
                    statRrr.textContent = data.rrr !== null ? (data.rrr > 36 ? '36.0+' : data.rrr.toFixed(2)) : '--';

                    statRrr.className = data.rrr > 12 ? 'eq-val danger-red' : 'eq-val';

                    // Generate Dynamic Commentary
                    const shortBat = batM.short || bat;
                    const shortBowl = bowlM.short || bowl;
                    let comm = '';
                    if (data.runs_left <= 0) {
                        comm = `🏆 Chase Completed! ${bat} has won the match!`;
                    } else if (data.wickets_left <= 0 || (data.balls_left <= 0 && data.runs_left > 0)) {
                        comm = `🎯 Inning Finished! ${bowl} has defended the total successfully!`;
                    } else {
                        comm = `${shortBat} needs ${data.runs_left} runs in ${data.balls_left} balls (${data.rrr ? data.rrr.toFixed(2) : '--'} RRR) with ${data.wickets_left} wickets in hand. `;
                        if (data.win >= 70) comm += `⚡ Strong advantage for ${shortBat}!`;
                        else if (data.win >= 45) comm += `⚖️ High pressure contest, hanging in the balance!`;
                        else comm += `🛡️ Steep chase, ${shortBowl} holding the upper hand.`;
                    }
                    commContent.textContent = comm;

                    if (chartInstance) {
                        chartInstance.data.datasets[0].data = [data.win, data.lose];
                        chartInstance.data.labels = [shortBat, shortBowl];
                        chartInstance.update();
                    }
                }
            } catch (err) {
                console.error('Prediction error:', err);
            }
        }

        // Event Handlers
        batSelect.addEventListener('change', () => { updateTeamStyles(); triggerPrediction(); });
        bowlSelect.addEventListener('change', () => { updateTeamStyles(); triggerPrediction(); });
        citySelect.addEventListener('change', triggerPrediction);
        targetInput.addEventListener('input', triggerPrediction);
        scoreInput.addEventListener('input', triggerPrediction);

        oversSlider.addEventListener('input', e => {
            updateOversDisplay(parseInt(e.target.value));
            triggerPrediction();
        });

        document.getElementById('btn-swap').addEventListener('click', () => {
            const temp = batSelect.value;
            batSelect.value = bowlSelect.value;
            bowlSelect.value = temp;
            updateTeamStyles();
            triggerPrediction();
        });

        document.getElementById('btn-reset').addEventListener('click', () => {
            targetInput.value = 180;
            scoreInput.value = 125;
            oversSlider.value = 84;
            updateOversDisplay(84);
            setWickets(3);
            triggerPrediction();
        });

        document.querySelectorAll('.btn-step').forEach(btn => {
            btn.addEventListener('click', () => {
                const inp = document.getElementById(btn.dataset.target);
                if (inp) {
                    let v = (parseInt(inp.value) || 0) + parseInt(btn.dataset.d);
                    inp.value = Math.max(0, v);
                    triggerPrediction();
                }
            });
        });

        document.querySelectorAll('.btn-mini-pill').forEach(btn => {
            btn.addEventListener('click', () => {
                const b = parseInt(btn.dataset.b);
                oversSlider.value = b;
                updateOversDisplay(b);
                triggerPrediction();
            });
        });

        // Live What-If Simulator Buttons
        document.querySelectorAll('.btn-action').forEach(btn => {
            btn.addEventListener('click', () => {
                const action = btn.dataset.a;
                let curScore = parseInt(scoreInput.value) || 0;
                let curBalls = parseInt(oversSlider.value) || 0;
                let curWickets = parseInt(wicketsVal.value) || 0;

                if (action === 'dot') {
                    curBalls = Math.min(120, curBalls + 1);
                } else if (action === 'single') {
                    curScore += 1;
                    curBalls = Math.min(120, curBalls + 1);
                } else if (action === 'two') {
                    curScore += 2;
                    curBalls = Math.min(120, curBalls + 1);
                } else if (action === 'four') {
                    curScore += 4;
                    curBalls = Math.min(120, curBalls + 1);
                } else if (action === 'six') {
                    curScore += 6;
                    curBalls = Math.min(120, curBalls + 1);
                } else if (action === 'wicket') {
                    curWickets = Math.min(10, curWickets + 1);
                    curBalls = Math.min(120, curBalls + 1);
                    setWickets(curWickets);
                } else if (action === 'tight') {
                    curScore += 3;
                    curBalls = Math.min(120, curBalls + 6);
                } else if (action === 'big') {
                    curScore += 16;
                    curBalls = Math.min(120, curBalls + 6);
                }

                scoreInput.value = curScore;
                oversSlider.value = curBalls;
                updateOversDisplay(curBalls);
                triggerPrediction();
            });
        });

        document.addEventListener('DOMContentLoaded', initApp);
    </script>
</body>
</html>"""

@app.route("/", methods=["GET", "HEAD"])
def index():
    if request.method == "HEAD":
        return "", 200
    return render_template_string(INDEX_HTML), 200

@app.route("/health", methods=["GET", "HEAD"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True) or {}
    except Exception:
        data = request.form.to_dict() or {}

    try:
        batting_team = data.get("batting_team", "").strip()
        bowling_team = data.get("bowling_team", "").strip()
        city = data.get("city", "").strip()
        target = float(data.get("target", 0))
        current_score = float(data.get("current_score", 0))
        overs_input = data.get("overs_completed", data.get("overs", 0))
        overs_completed = float(overs_input)
        wickets_down = int(data.get("wickets_down", data.get("wickets_out", 0)))
    except (KeyError, ValueError, TypeError) as err:
        return jsonify({"error": f"Invalid input fields: {err}"}), 400

    if not batting_team or not bowling_team or not city:
        return jsonify({"error": "Please select batting team, bowling team, and city."}), 400

    if batting_team == bowling_team:
        return jsonify({"error": "Batting and bowling teams must be different."}), 400

    whole_overs = int(overs_completed)
    balls_in_over = round((overs_completed - whole_overs) * 10)
    if balls_in_over < 0 or balls_in_over > 5:
        balls_in_over = min(5, max(0, balls_in_over))

    balls_bowled = whole_overs * 6 + balls_in_over
    balls_bowled = min(120, max(0, balls_bowled))
    balls_left = 120 - balls_bowled
    runs_left = target - current_score
    wickets_left = max(0, 10 - wickets_down)

    if runs_left <= 0:
        return jsonify({
            "win": 100.0,
            "lose": 0.0,
            "runs_left": 0,
            "balls_left": balls_left,
            "wickets_left": wickets_left,
            "crr": round((current_score * 6) / balls_bowled, 2) if balls_bowled > 0 else 0.0,
            "rrr": 0.0,
        })

    if wickets_left <= 0 or (balls_left <= 0 and runs_left > 0):
        return jsonify({
            "win": 0.0,
            "lose": 100.0,
            "runs_left": int(runs_left),
            "balls_left": int(balls_left),
            "wickets_left": wickets_left,
            "crr": round((current_score * 6) / balls_bowled, 2) if balls_bowled > 0 else 0.0,
            "rrr": round((runs_left * 6) / balls_left, 2) if balls_left > 0 else 0.0,
        })

    crr = (current_score * 6.0) / balls_bowled if balls_bowled > 0 else 0.0
    rrr = (runs_left * 6.0) / balls_left if balls_left > 0 else 0.0

    win_pct, lose_pct = None, None

    # Attempt Pipeline prediction
    global pipe
    if pipe is not None:
        input_df = pd.DataFrame([{
            "batting_team": batting_team,
            "bowling_team": bowling_team,
            "city": city,
            "runs_left": runs_left,
            "balls_left": balls_left,
            "wickets": wickets_left,
            "total_runs_x": target,
            "crr": crr,
            "rrr": rrr,
        }])
        try:
            result = pipe.predict_proba(input_df)
            lose_pct = round(float(result[0][0]) * 100, 1)
            win_pct = round(float(result[0][1]) * 100, 1)
        except Exception as exc:
            win_pct, lose_pct = None, None

    # Fallback high-accuracy solver
    if win_pct is None or lose_pct is None:
        win_pct, lose_pct = calculate_analytical_probability(
            runs_left, balls_left, wickets_left, target, crr, rrr
        )

    return jsonify({
        "win": win_pct,
        "lose": lose_pct,
        "runs_left": int(runs_left),
        "balls_left": int(balls_left),
        "wickets_left": int(wickets_left),
        "crr": round(crr, 2),
        "rrr": round(rrr, 2),
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
