# NFL Predictor Project - Development Instructions

## Project Overview
This is an NFL game prediction system using machine learning (XGBoost) with a Flask backend and Bootstrap frontend. The system predicts game outcomes based on team statistics, momentum, and historical performance.

## Current Status (2025-09-27)
- ✅ Individual game predictions working
- ✅ Team colors implemented (replacing green/red with NFL team colors)
- ✅ Logarithmic year weighting implemented
- ✅ Weekly predictions feature fully implemented and tested
- ✅ Schedule parser for NFL 2025 season complete

## Server Setup & Running
```bash
# Navigate to backend directory
cd /Users/rakendumalladi/Desktop/projects/nfl/backend

# Activate virtual environment
source ../venv/bin/activate

# Install dependencies (if needed)
pip install pandas numpy scikit-learn xgboost flask

# Run server
python api.py
# Server runs on http://127.0.0.1:5001 (port 5001 to avoid macOS AirPlay conflict)
```

## File Structure & Key Files

### Backend Files
- **`api.py`** - Main Flask application with all API endpoints
- **`schedule_data.py`** - NFL 2025 schedule parser and team mapping
- **`lib/nfl_teams.py`** - NFL team data with official colors and info
- **`predictor.py`** - Machine learning model utilities
- **`nfl-2025.csv`** - NFL 2025 season schedule data

### Frontend Files
- **`templates/layout.html`** - Base template with navigation
- **`templates/weekly.html`** - Weekly predictions page
- **`templates/dashboard.html`** - Main dashboard (individual predictions)
- **`static/js/main.js`** - All JavaScript functionality
- **`static/css/style.css`** - Custom styling

## API Endpoints

### Core Prediction Endpoints
- **`POST /api/predict`** - Individual game prediction
  ```json
  {"teamA": "buf", "teamB": "bal"}
  ```

- **`POST /api/debug_prediction`** - Debug endpoint with detailed features

### Weekly Prediction Endpoints (NEW)
- **`GET /api/weeks`** - Get available weeks (1-18)
- **`GET /api/schedule/{week}`** - Get all games for specific week
- **`POST /api/predict_week/{week}`** - Predict all games in a week

### Other Endpoints
- **`GET /`** - Dashboard page
- **`GET /weekly`** - Weekly predictions page
- **`GET /performance`** - Model performance page
- **`GET /api/model/performance`** - Performance metrics JSON

## Recent Major Changes

### 1. Team Colors Implementation
**Problem Solved**: Changed from generic green/red metrics to NFL team-specific colors

**Files Modified**:
- `lib/nfl_teams.py`: Added `primary_color` and `secondary_color` for all 32 teams
- `api.py`: Updated prediction responses to include team colors
- `static/js/main.js`: Modified `displayDetailedMetrics()` to use team colors

**Key Code Addition**:
```javascript
// In main.js - displayDetailedMetrics function
if (teamAColors && teamBColors) {
    metricElement.style.color = isTeamAHigher ? teamAColors.primary : teamBColors.primary;
}
```

### 2. Logarithmic Year Weighting
**Problem Solved**: Switched from linear (6x, 4x, 3x) to logarithmic weighting for recent years

**Files Modified**:
- `api.py`: Added `calculate_year_weight()` function and updated all weighting functions

**Key Code**:
```python
def calculate_year_weight(season):
    current_year = 2025
    years_back = current_year - season
    base_weight = 1.0
    log_factor = 2.0
    weight = base_weight + log_factor * math.log(years_since_oldest + 1)
    return weight
```

### 3. Weekly Predictions Feature (MAJOR NEW FEATURE)
**Problem Solved**: Added ability to predict all games in a selected NFL week

**New Files**:
- `schedule_data.py`: Complete NFL schedule parser with team mapping
- `templates/weekly.html`: Weekly predictions page template

**Files Modified**:
- `api.py`: Added 3 new endpoints and weekly prediction logic
- `templates/layout.html`: Added weekly predictions navigation
- `static/js/main.js`: Added comprehensive weekly page functionality

**Key Features**:
- Week selector (1-18)
- Batch predictions for entire weeks
- Team colors in game cards
- Collapsible detailed metrics
- Week statistics summary

## Team Mapping & Data Handling

### Team Abbreviations
The system uses lowercase team IDs internally (`buf`, `dal`, `phi`) but the ML model expects uppercase (`BUF`, `DAL`, `PHI`).

**Critical Mapping in `api.py`**:
```python
TEAM_MAPPING = {
    "buf": "BUF", "mia": "MIA", "ne": "NE", "nyj": "NYJ",
    "bal": "BAL", "cin": "CIN", "cle": "CLE", "pit": "PIT",
    # ... all 32 teams
}
```

### Schedule Data Format
The NFL 2025 schedule (`nfl-2025.csv`) contains:
- **Columns**: Match Number, Week, Date, Location, Home Team, Away Team, Result
- **Date Formats**: Mixed formats handled by parser (DD/MM/YYYY, MM/DD/YY, etc.)
- **Team Names**: Full names converted to abbreviations via mapping

## Common Issues & Solutions

### 1. Feature Generation Errors
**Symptom**: "Found array with 0 feature(s)" error
**Cause**: Empty `feature_names` list or model not properly loaded
**Solution**: Implemented fallback prediction logic in weekly predictions

### 2. Team Name Mismatches
**Symptom**: Teams not found or predictions failing
**Solution**: Always use `TEAM_MAPPING.get(team_id, team_id.upper())` for conversions

### 3. Port Conflicts
**Issue**: Port 5000 conflicts with macOS AirPlay
**Solution**: Server runs on port 5001

## Frontend JavaScript Structure

### Main Functions in `main.js`
- **`initializeWeeklyPage()`** - Sets up weekly predictions page
- **`handleWeekSelection()`** - Handles week dropdown changes
- **`handlePredictWeek()`** - Generates predictions for selected week
- **`displayWeeklyPredictions()`** - Renders game cards with predictions
- **`createGameCard()`** - Creates individual game card elements
- **`displayDetailedMetrics()`** - Shows team comparison metrics with colors

### Page Routing
```javascript
// Auto-detects current page and initializes appropriate functionality
if (window.location.pathname === '/weekly') {
    initializeWeeklyPage();
} else if (window.location.pathname === '/performance') {
    initializePerformancePage();
} else {
    initializeDashboardPage();
}
```

## Database & Model Info
- **Model Type**: XGBoost Classification Pipeline with StandardScaler
- **Model File**: `artifacts/streamlined_pipeline_classification_20250924_165904.joblib`
- **Dataset Cache**: `artifacts/team_data_cache.pkl` (5422 team-game records, 2015-2024)
- **Features**: EPA, Success Rate, Momentum Score, Opponent Strength, etc.

## Future Development Notes

### Planned Features (Not Yet Implemented)
1. **Actual vs Predicted Comparison**: For completed weeks, show model accuracy
2. **Confidence Thresholds**: Highlight high-confidence vs close predictions
3. **Historical Week Analysis**: Compare predictions against actual results

### Development Workflow
1. **Backend Changes**: Modify `api.py` endpoints or add new ones
2. **Frontend Updates**: Update templates and `main.js` functionality
3. **Testing**: Use curl commands to test API endpoints
4. **Debugging**: Check server logs for errors, use debug endpoints

### Key Dependencies
- **Python**: pandas, numpy, scikit-learn, xgboost, flask
- **Frontend**: Bootstrap 5.3.0, Chart.js
- **Browser**: Modern browsers with ES6+ support

## Command Reference

### Testing API Endpoints
```bash
# Test individual prediction
curl -X POST -H "Content-Type: application/json" \
  -d '{"teamA": "buf", "teamB": "bal"}' \
  http://127.0.0.1:5001/api/predict

# Test weekly predictions
curl -X POST http://127.0.0.1:5001/api/predict_week/1

# Get available weeks
curl http://127.0.0.1:5001/api/weeks

# Get week schedule
curl http://127.0.0.1:5001/api/schedule/1
```

### Useful Debugging
```bash
# Monitor server logs
tail -f logs/output.log

# Check server status in browser
http://127.0.0.1:5001/

# Test specific team predictions
http://127.0.0.1:5001/ (use dashboard form)
```

This documentation should help any future AI assistant or developer understand the current state and continue development efficiently.