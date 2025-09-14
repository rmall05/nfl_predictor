from flask import Flask, request, jsonify
from flask_cors import CORS
import random
import warnings
warnings.filterwarnings('ignore')

from lib.nfl_teams import NFL_TEAMS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Simple team strength ratings (mock data based on general NFL knowledge)
TEAM_STRENGTH = {
    "kc": 95, "buf": 92, "sf": 89, "dal": 88, "phi": 87, "bal": 86, "cin": 85,
    "gb": 84, "min": 83, "det": 82, "mia": 81, "ten": 80, "lac": 79, "jax": 78,
    "nyj": 77, "lv": 76, "pit": 75, "cle": 74, "ind": 73, "hou": 72, "ne": 71,
    "was": 70, "tb": 69, "atl": 68, "no": 67, "sea": 66, "lar": 65, "den": 64,
    "nyg": 63, "car": 62, "chi": 61, "ari": 60
}

def calculate_win_probability(team_a: str, team_b: str) -> float:
    """Calculate win probability based on team strength"""
    strength_a = TEAM_STRENGTH.get(team_a, 75)
    strength_b = TEAM_STRENGTH.get(team_b, 75)

    # Simple logistic function for win probability
    diff = strength_a - strength_b
    # Convert difference to probability using a scaling factor
    win_prob = 0.5 + (diff / 100.0) * 0.4  # Scale difference to probability

    # Add some randomness to make predictions more realistic
    win_prob += random.uniform(-0.05, 0.05)

    # Ensure probability is between 0.15 and 0.85 (no guaranteed wins)
    win_prob = max(0.15, min(0.85, win_prob))

    return win_prob

def get_team_stats(team_abbr: str) -> dict:
    """Generate realistic team stats"""
    # Get team strength
    strength = TEAM_STRENGTH.get(team_abbr.lower(), 75)

    # Generate stats based on strength
    epa_per_play = 0.05 + (strength - 50) * 0.003
    success_rate = 40.0 + (strength - 50) * 0.15
    momentum_score = max(30, min(95, strength + random.randint(-10, 10)))

    # Generate record based on strength
    expected_wins = max(2, min(14, int((strength - 40) * 0.3)))
    wins = expected_wins + random.randint(-2, 2)
    losses = 16 - wins
    wins = max(0, min(16, wins))
    losses = max(0, min(16, 16 - wins))

    # Generate recent form
    win_prob = max(0.2, min(0.8, strength / 100.0))
    recent_form = []
    for _ in range(5):
        recent_form.append("W" if random.random() < win_prob else "L")

    # Get team info
    team_info = next((t for t in NFL_TEAMS if t['abbreviation'].lower() == team_abbr.lower()), None)
    team_name = f"{team_info['city']} {team_info['name']}" if team_info else team_abbr

    return {
        "name": team_name,
        "record": f"{wins}-{losses}",
        "epa_per_play": round(epa_per_play, 3),
        "success_rate": round(success_rate, 1),
        "recent_form": recent_form,
        "momentum_score": momentum_score
    }

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "model_loaded": True})

@app.route('/api/predict', methods=['POST'])
def predict_game():
    """Predict game outcome between two teams"""
    try:
        data = request.json
        team_a = data.get('teamA', '').lower()
        team_b = data.get('teamB', '').lower()

        if not team_a or not team_b:
            return jsonify({"error": "Both teamA and teamB are required"}), 400

        if team_a == team_b:
            return jsonify({"error": "Teams must be different"}), 400

        # Get team info for display names
        team_a_info = next((t for t in NFL_TEAMS if t['id'] == team_a), None)
        team_b_info = next((t for t in NFL_TEAMS if t['id'] == team_b), None)

        if not team_a_info or not team_b_info:
            return jsonify({"error": "Invalid team ID"}), 400

        team_a_name = f"{team_a_info['city']} {team_a_info['name']}"
        team_b_name = f"{team_b_info['city']} {team_b_info['name']}"

        # Calculate win probabilities
        win_prob_a = calculate_win_probability(team_a, team_b)
        win_prob_b = 1.0 - win_prob_a

        # Calculate confidence based on how certain the prediction is
        confidence = int(abs(win_prob_a - 0.5) * 200)
        confidence = max(60, min(95, confidence))

        # Get team stats
        team_a_stats = get_team_stats(team_a_info['abbreviation'])
        team_b_stats = get_team_stats(team_b_info['abbreviation'])

        # Create key factors
        epa_diff = team_a_stats['epa_per_play'] - team_b_stats['epa_per_play']
        momentum_diff = team_a_stats['momentum_score'] - team_b_stats['momentum_score']
        success_diff = team_a_stats['success_rate'] - team_b_stats['success_rate']

        key_factors = [
            {
                "name": "EPA per Play",
                "value": round(epa_diff, 3),
                "description": f"Expected Points Added per play differential {'favors ' + team_a_name if epa_diff > 0 else 'favors ' + team_b_name}"
            },
            {
                "name": "Momentum Score",
                "value": round(momentum_diff / 100, 2),
                "description": "Recent performance and trend analysis"
            },
            {
                "name": "Success Rate",
                "value": round(success_diff / 100, 2),
                "description": "Play success rate differential"
            },
            {
                "name": "Recent Form",
                "value": round(win_prob_a * 0.8, 2),
                "description": "Performance in last 5 games"
            },
            {
                "name": "Team Strength",
                "value": round((TEAM_STRENGTH.get(team_a, 75) - TEAM_STRENGTH.get(team_b, 75)) / 100, 2),
                "description": "Overall team strength comparison"
            }
        ]

        return jsonify({
            "teamA_win_prob": round(win_prob_a * 100, 1),
            "teamB_win_prob": round(win_prob_b * 100, 1),
            "confidence": confidence,
            "key_factors": key_factors,
            "teamA_stats": team_a_stats,
            "teamB_stats": team_b_stats
        })

    except Exception as e:
        print(f"API Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/teams', methods=['GET'])
def get_teams():
    """Get list of all NFL teams"""
    return jsonify(NFL_TEAMS)

if __name__ == '__main__':
    print("Starting NFL Predictor API (Simple Mode)...")
    print("Model: Mock predictions based on team strength ratings")
    app.run(debug=True, host='0.0.0.0', port=5000)