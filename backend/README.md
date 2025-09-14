# NFL Predictor Backend

Flask API server for NFL game predictions.

## Setup

1. Create virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Server

### Simple API (Mock Predictions)
```bash
python simple_api.py
```

### Full ML API (Requires Trained Model)
```bash
python api.py
```

The API will be available at `http://localhost:5000`

## Endpoints

- `GET /api/health` - Health check
- `POST /api/predict` - Predict game outcome
- `GET /api/teams` - Get NFL teams list

## Example Request

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"teamA": "kc", "teamB": "buf"}'
```