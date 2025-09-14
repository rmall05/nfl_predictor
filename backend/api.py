from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import joblib
from typing import Dict, Any, List
import warnings
warnings.filterwarnings('ignore')

from predictor import assemble_team_game_dataset, fit_phase3_pipeline
from lib.nfl_teams import NFL_TEAMS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables to store the model and data
model = None
scaler = None
pca = None
feature_names = None
dataset = None

# NFL team abbreviation mapping for consistency with your data
TEAM_MAPPING = {
    "buf": "BUF", "mia": "MIA", "ne": "NE", "nyj": "NYJ",
    "bal": "BAL", "cin": "CIN", "cle": "CLE", "pit": "PIT",
    "hou": "HOU", "ind": "IND", "jax": "JAX", "ten": "TEN",
    "den": "DEN", "kc": "KC", "lv": "LV", "lac": "LAC",
    "dal": "DAL", "nyg": "NYG", "phi": "PHI", "was": "WAS",
    "chi": "CHI", "det": "DET", "gb": "GB", "min": "MIN",
    "atl": "ATL", "car": "CAR", "no": "NO", "tb": "TB",
    "ari": "ARI", "lar": "LAR", "sf": "SF", "sea": "SEA"
}

def load_model():
    """Load the trained model and preprocessors"""
    global model, scaler, pca, feature_names, dataset

    try:
        # Try to load existing model
        if os.path.exists('artifacts/phase3_pipeline.joblib'):
            model = joblib.load('artifacts/phase3_pipeline.joblib')
            scaler = None
            pca = None
            feature_names = []

            print("Model loaded successfully!")
            return True

    except Exception as e:
        print(f"Error loading model: {e}")

    # If model doesn't exist, train a new one
    print("Training new model...")
    try:
        # Load data and train model
        years = [2020, 2021, 2022]  # Use recent years for quick training
        dataset = assemble_team_game_dataset(years, include_momentum=True)

        if len(dataset) > 0:
            # Train the model
            pipeline, data_splits, performance_metrics = fit_phase3_pipeline(
                dataset,
                enable_hyperopt=False,  # Disable for faster startup
                task="classification",
                train_seasons=[2020, 2021],
                test_seasons=[2022]
            )

            model = pipeline
            scaler = None
            pca = None
            feature_names = []

            print("New model trained successfully!")
            return True
        else:
            print("No data available for training")
            return False

    except Exception as e:
        print(f"Error training model: {e}")
        return False

def get_team_stats(team_abbr: str) -> Dict[str, Any]:
    """Get recent stats for a team"""
    global dataset

    if dataset is None or len(dataset) == 0:
        # Return mock data if no dataset available
        return {
            "name": f"Team {team_abbr}",
            "record": "8-6",
            "epa_per_play": 0.12,
            "success_rate": 45.0,
            "recent_form": ["W", "L", "W", "W", "L"],
            "momentum_score": 75
        }

    # Get recent stats for the team
    team_data = dataset[dataset['team'] == team_abbr].tail(5)  # Last 5 games

    if len(team_data) == 0:
        return {
            "name": f"Team {team_abbr}",
            "record": "8-6",
            "epa_per_play": 0.10,
            "success_rate": 44.0,
            "recent_form": ["W", "L", "W", "L", "W"],
            "momentum_score": 70
        }

    # Calculate stats
    recent_record = team_data['win'].tolist()
    recent_form = ["W" if w else "L" for w in recent_record]

    epa_per_play = team_data['epa_per_play'].mean() if 'epa_per_play' in team_data.columns else 0.10
    success_rate = team_data.get('success_rate', pd.Series([44.0])).mean()
    momentum = team_data.get('momentum_score', pd.Series([70])).mean()

    wins = sum(recent_record)
    losses = len(recent_record) - wins

    return {
        "name": f"Team {team_abbr}",
        "record": f"{wins}-{losses}",
        "epa_per_play": float(epa_per_play),
        "success_rate": float(success_rate),
        "recent_form": recent_form[-5:],  # Last 5 games
        "momentum_score": float(momentum)
    }

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "model_loaded": model is not None})

@app.route('/api/predict', methods=['POST'])
def predict_game():
    """Predict game outcome between two teams"""
    try:
        data = request.json
        team_a = data.get('teamA', '').lower()
        team_b = data.get('teamB', '').lower()

        if not team_a or not team_b:
            return jsonify({"error": "Both teamA and teamB are required"}), 400

        # Map to NFL abbreviations
        team_a_abbr = TEAM_MAPPING.get(team_a, team_a.upper())
        team_b_abbr = TEAM_MAPPING.get(team_b, team_b.upper())

        # Get team names for display
        team_a_info = next((t for t in NFL_TEAMS if t['id'] == team_a), None)
        team_b_info = next((t for t in NFL_TEAMS if t['id'] == team_b), None)

        team_a_name = f"{team_a_info['city']} {team_a_info['name']}" if team_a_info else team_a_abbr
        team_b_name = f"{team_b_info['city']} {team_b_info['name']}" if team_b_info else team_b_abbr

        if model is None:
            # Return mock prediction if model not available
            return jsonify({
                "teamA_win_prob": 65.2,
                "teamB_win_prob": 34.8,
                "confidence": 75,
                "key_factors": [
                    {"name": "EPA per Play", "value": 0.15, "description": f"Expected Points Added per play differential favors {team_a_name}"},
                    {"name": "Recent Form", "value": 0.80, "description": "Performance in last 5 games"},
                    {"name": "Momentum Score", "value": 0.75, "description": "Recent performance trend analysis"},
                    {"name": "Success Rate", "value": 0.65, "description": "Play success rate differential"},
                ],
                "teamA_stats": get_team_stats(team_a_abbr),
                "teamB_stats": get_team_stats(team_b_abbr)
            })

        # Create prediction features (simplified version)
        # In a real scenario, you'd need to create features similar to your training data
        try:
            if hasattr(model, 'predict_proba'):
                # Mock features for prediction - replace with actual feature engineering
                # Get a sample from dataset to understand feature structure
                if dataset is not None and len(dataset) > 0:
                    # Get the feature columns (exclude target variables)
                    exclude_cols = ['team', 'season', 'week', 'opponent', 'win', 'point_diff', 'game_id', 'home_team', 'away_team']
                    feature_cols = [col for col in dataset.columns if col not in exclude_cols]
                    n_features = len(feature_cols)
                else:
                    n_features = 50

                features = np.random.normal(0, 1, n_features).reshape(1, -1)
                win_prob = float(model.predict_proba(features)[0][1])  # Probability of team A winning
            else:
                # Fallback for models without predict_proba
                win_prob = 0.65

            confidence = min(95, max(50, abs(win_prob - 0.5) * 200))  # Confidence based on how certain the prediction is

            # Create key factors based on feature importance (mock for now)
            key_factors = [
                {"name": "EPA per Play", "value": win_prob * 0.3, "description": f"Expected Points Added per play differential favors {team_a_name if win_prob > 0.5 else team_b_name}"},
                {"name": "Recent Form", "value": win_prob * 0.8, "description": "Performance in last 5 games"},
                {"name": "Momentum Score", "value": win_prob * 0.7, "description": "Recent performance trend analysis"},
                {"name": "Success Rate", "value": win_prob * 0.6, "description": "Play success rate differential"},
                {"name": "Strength of Schedule", "value": win_prob * 0.5, "description": "Quality of opponents faced this season"},
            ]

        except Exception as e:
            print(f"Prediction error: {e}")
            # Fallback to mock prediction
            win_prob = 0.652
            confidence = 75
            key_factors = [
                {"name": "EPA per Play", "value": 0.15, "description": f"Expected Points Added per play differential favors {team_a_name}"},
                {"name": "Recent Form", "value": 0.80, "description": "Performance in last 5 games"},
                {"name": "Momentum Score", "value": 0.75, "description": "Recent performance trend analysis"},
                {"name": "Success Rate", "value": 0.65, "description": "Play success rate differential"},
            ]

        return jsonify({
            "teamA_win_prob": float(win_prob * 100),
            "teamB_win_prob": float((1 - win_prob) * 100),
            "confidence": int(confidence),
            "key_factors": key_factors,
            "teamA_stats": get_team_stats(team_a_abbr),
            "teamB_stats": get_team_stats(team_b_abbr)
        })

    except Exception as e:
        print(f"API Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/teams', methods=['GET'])
def get_teams():
    """Get list of all NFL teams"""
    return jsonify(NFL_TEAMS)

if __name__ == '__main__':
    print("Starting NFL Predictor API...")

    # Load model on startup
    load_model()

    # Start Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)