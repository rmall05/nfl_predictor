# NFL Game Predictor

A full-stack application for predicting NFL game outcomes using machine learning and advanced analytics.

## Project Structure

```
nfl_predictor/
├── backend/          # Python Flask API
│   ├── api.py       # Full ML API (with trained model)
│   ├── simple_api.py # Simple API (mock predictions)
│   ├── main.py      # Data processing pipeline
│   ├── predictor.py # ML model training
│   ├── clustering.py # Team clustering analysis
│   └── lib/         # Shared utilities
├── frontend/         # Next.js React web app
│   ├── app/         # Next.js app router pages
│   ├── components/  # React components
│   ├── lib/         # Frontend utilities
│   └── types/       # TypeScript types
├── artifacts/        # Model artifacts and outputs
├── venv/            # Python virtual environment
└── CLAUDE.md        # Project documentation
```

## Quick Start

### 1. Backend Setup
```bash
cd backend
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
python simple_api.py  # Start simple API
```

### 2. Frontend Setup
```bash
cd frontend
npm install
npm run dev  # Start Next.js app
```

### 3. Access the Application
- Frontend: http://localhost:3000
- Backend API: http://localhost:5000

## Features

### Backend
- **Flask REST API** with CORS support
- **Real-time predictions** based on team strength ratings
- **NFL team data** with proper IDs and metadata
- **Comprehensive stats** including EPA, momentum, and recent form
- **Extensible architecture** for integrating ML models

### Frontend
- **Modern React UI** with Next.js 14
- **Real-time API integration** with error handling
- **Interactive team selection** with all 32 NFL teams
- **Win probability visualization** with animated gauges
- **Key factors analysis** with detailed tooltips
- **Team comparison** with performance metrics
- **Responsive design** with Tailwind CSS

## API Endpoints

- `GET /api/health` - Health check and model status
- `POST /api/predict` - Predict game outcome between two teams
- `GET /api/teams` - Get list of all NFL teams

## Machine Learning Pipeline

The project includes a sophisticated ML pipeline:

1. **Data Processing** (`main.py`)
   - NFL play-by-play data from `nfl_data_py`
   - Offensive and defensive statistics
   - Momentum calculations with EMA

2. **Feature Engineering** (`predictor.py`)
   - EPA-based metrics
   - Strength of schedule analysis
   - PCA dimensionality reduction
   - Advanced model ensembles

3. **Model Training**
   - Random Forest, XGBoost, Neural Networks
   - Hyperparameter optimization
   - Time-series cross-validation

## Development

The project is set up for easy development with:
- Hot reloading for both frontend and backend
- TypeScript for type safety
- Comprehensive error handling
- Modular, maintainable code structure

See individual README files in `backend/` and `frontend/` directories for detailed setup instructions.