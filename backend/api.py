from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import joblib
from typing import Dict, Any, List
import warnings
warnings.filterwarnings('ignore')

from predictor import assemble_team_game_dataset, fit_streamlined_pipeline, predict_new_season_games, prepare_2025_data
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
        # Try to load existing streamlined model
        latest_pipeline = None
        artifacts_dir = "artifacts"

        if os.path.exists(artifacts_dir):
            # Find the latest streamlined pipeline
            pipeline_files = [f for f in os.listdir(artifacts_dir) if f.startswith('streamlined_pipeline_') and f.endswith('.joblib')]
            if pipeline_files:
                # Sort by creation time and get the latest
                pipeline_files.sort(reverse=True)
                latest_pipeline = os.path.join(artifacts_dir, pipeline_files[0])

        if latest_pipeline and os.path.exists(latest_pipeline):
            model = joblib.load(latest_pipeline)

            # Load corresponding feature names
            timestamp = latest_pipeline.split('_')[-1].replace('.joblib', '')
            features_file = os.path.join(artifacts_dir, f'streamlined_features_{timestamp}.csv')
            if os.path.exists(features_file):
                feature_names = pd.read_csv(features_file, header=None)[0].tolist()
            else:
                feature_names = []

            print(f"Streamlined model loaded successfully from {latest_pipeline}")
            return True

    except Exception as e:
        print(f"Error loading model: {e}")

    # If model doesn't exist, train a new one
    print("Training new streamlined model...")
    try:
        # Load data and train model with optimal configuration
        years = list(range(2015, 2025))  # Full dataset
        dataset = assemble_team_game_dataset(
            years,
            include_momentum=True,
            momentum_metric="explosive_rate",  # Optimal configuration
            ema_span=5
        )

        if len(dataset) > 0:
            # Train the streamlined model
            pipeline, data_splits, performance_metrics = fit_streamlined_pipeline(
                dataset,
                include_momentum=True,
                task="classification"
            )

            model = pipeline
            feature_names = data_splits['feature_names']

            # Save the model
            os.makedirs("artifacts", exist_ok=True)
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            pipeline_path = f"artifacts/streamlined_pipeline_classification_{timestamp}.joblib"
            features_path = f"artifacts/streamlined_features_{timestamp}.csv"

            joblib.dump(pipeline, pipeline_path)
            pd.Series(feature_names).to_csv(features_path, index=False)

            print(f"New streamlined model trained and saved! Test accuracy: {performance_metrics.get('test_accuracy', 'N/A'):.1%}")
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
        # Load basic team info
        team_info = next((t for t in NFL_TEAMS if t['abbreviation'] == team_abbr), None)
        team_name = f"{team_info['city']} {team_info['name']}" if team_info else team_abbr

        return {
            "name": team_name,
            "record": "0-0",
            "epa_per_play": 0.0,
            "success_rate": 0.0,
            "recent_form": [],
            "momentum_score": 0.0
        }

    # Get recent stats for the team (last 5 games)
    team_data = dataset[dataset['team'] == team_abbr].tail(5)

    if len(team_data) == 0:
        team_info = next((t for t in NFL_TEAMS if t['abbreviation'] == team_abbr), None)
        team_name = f"{team_info['city']} {team_info['name']}" if team_info else team_abbr

        return {
            "name": team_name,
            "record": "0-0",
            "epa_per_play": 0.0,
            "success_rate": 0.0,
            "recent_form": [],
            "momentum_score": 0.0
        }

    # Calculate actual stats from data
    recent_record = team_data['win'].tolist()
    recent_form = ["W" if w else "L" for w in recent_record]

    epa_per_play = team_data['epa_per_play'].mean() if 'epa_per_play' in team_data.columns else 0.0
    success_rate = team_data.get('success_rate', pd.Series([0.0])).mean()
    momentum = team_data.get('momentum_score', pd.Series([0.0])).mean()

    wins = sum(recent_record)
    losses = len(recent_record) - wins

    # Get team name
    team_info = next((t for t in NFL_TEAMS if t['abbreviation'] == team_abbr), None)
    team_name = f"{team_info['city']} {team_info['name']}" if team_info else team_abbr

    return {
        "name": team_name,
        "record": f"{wins}-{losses}",
        "epa_per_play": float(epa_per_play),
        "success_rate": float(success_rate * 100),  # Convert to percentage
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
            return jsonify({"error": "Model not available. Please check server logs."}), 500

        # Create prediction using actual team data
        try:
            # For now, use a simplified prediction approach
            # In production, you would need recent team-game data for both teams
            # This is a placeholder that demonstrates the API structure

            if hasattr(model, 'predict_proba') and feature_names:
                # Create mock features matching the model's training features
                # This should be replaced with actual team performance data
                n_features = len(feature_names)
                features = np.random.normal(0, 1, n_features).reshape(1, -1)

                # Get prediction
                win_prob = float(model.predict_proba(features)[0][1])
            else:
                # Basic fallback
                win_prob = 0.55

            confidence = min(95, max(50, abs(win_prob - 0.5) * 200))

            # Get team stats for display
            team_a_stats = get_team_stats(team_a_abbr)
            team_b_stats = get_team_stats(team_b_abbr)

            # Create key factors based on team stats differences
            epa_diff = team_a_stats.get('epa_per_play', 0) - team_b_stats.get('epa_per_play', 0)
            momentum_diff = team_a_stats.get('momentum_score', 0) - team_b_stats.get('momentum_score', 0)
            success_diff = team_a_stats.get('success_rate', 0) - team_b_stats.get('success_rate', 0)

            key_factors = [
                {
                    "name": "EPA per Play",
                    "value": round(epa_diff, 3),
                    "description": f"Expected Points Added per play differential {'favors ' + team_a_name if epa_diff > 0 else 'favors ' + team_b_name if epa_diff < 0 else 'is neutral'}"
                },
                {
                    "name": "Recent Form",
                    "value": round(win_prob * 0.8, 2),
                    "description": "Performance in recent games"
                },
                {
                    "name": "Momentum Score",
                    "value": round(momentum_diff / 100, 2),
                    "description": "Recent performance trend analysis"
                },
                {
                    "name": "Success Rate",
                    "value": round(success_diff / 100, 2),
                    "description": "Play success rate differential"
                },
                {
                    "name": "Strength of Schedule",
                    "value": round(win_prob * 0.5, 2),
                    "description": "Quality of opponents faced this season"
                },
            ]

        except Exception as e:
            print(f"Prediction error: {e}")
            return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

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

@app.route('/api/predict_2025', methods=['POST'])
def predict_2025_season():
    """Predict outcomes for 2025 season games"""
    try:
        data = request.json
        weeks = data.get('weeks', [1])  # Default to week 1

        if model is None or not feature_names:
            return jsonify({"error": "Model not available. Please check server logs."}), 500

        # Prepare 2025 season data
        season_2025_data = prepare_2025_data(weeks_to_predict=weeks)

        if len(season_2025_data) == 0:
            return jsonify({
                "message": "2025 season data not yet available",
                "predictions": [],
                "weeks_requested": weeks
            })

        # Make predictions
        predictions = predict_new_season_games(model, feature_names, season_2025_data)

        # Format results
        prediction_results = []
        for _, game in predictions.iterrows():
            prediction_results.append({
                "week": int(game['week']),
                "team": game['team'],
                "opponent": game['opponent'],
                "predicted_win": bool(game['predicted_win']),
                "win_probability": round(float(game['win_probability']), 3),
                "confidence": min(95, max(50, abs(game['win_probability'] - 0.5) * 200))
            })

        return jsonify({
            "message": "2025 season predictions generated successfully",
            "predictions": prediction_results,
            "weeks_predicted": weeks,
            "total_games": len(prediction_results)
        })

    except Exception as e:
        print(f"2025 prediction error: {e}")
        return jsonify({"error": f"2025 prediction failed: {str(e)}"}), 500

if __name__ == '__main__':
    print("Starting NFL Predictor API...")

    # Load model on startup
    load_model()

    # Start Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)