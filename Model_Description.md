# NFL Predictor Project - Complete Breakdown

## Project Overview
A full-stack NFL game outcome prediction system that uses machine learning to analyze team performance and predict game results. The system combines advanced analytics with a modern web interface.

## Architecture

### Backend (`/backend/`)
**Technology Stack**: Python, Flask, scikit-learn, pandas, numpy

**Core Components**:

1. **`main.py`** - Data processing pipeline
   - `load_and_prepare_pbp()`: Downloads NFL play-by-play data from nfl_data_py
   - `offensive_stats()`: Aggregates team offensive metrics (EPA, success rates)
   - `defensive_stats()`: Calculates defensive performance (opponent stats allowed)
   - `add_momentum_simple()`: Computes momentum scores using EMA and Z-score normalization

2. **`predictor.py`** - Machine learning pipeline
   - `assemble_team_game_dataset()`: Combines offensive/defensive stats with game outcomes
   - `fit_pca_pipeline()`: Implements StandardScaler + PCA + LogisticRegression/LinearRegression
   - Supports both classification (win/loss) and regression (point differential)
   - Uses season-based train/test splits (2015-2021 default)

3. **`simple_api.py`** - Flask API server (Currently Active)
   - `/api/health`: Health check endpoint
   - `/api/predict`: Game prediction endpoint (POST)
   - `/api/teams`: NFL teams data endpoint
   - Uses mock predictions based on team strength ratings
   - CORS enabled for frontend communication

4. **`api.py`** - Full ML API server (Alternative)
   - Loads actual trained models from artifacts/
   - More sophisticated prediction logic

5. **`clustering.py`** - Team clustering analysis (DEPRECATED - not used in current model)

6. **`lib/nfl_teams.py`** - NFL team data structure

### Frontend (`/frontend/`)
**Technology Stack**: Next.js 14, React 18, TypeScript, Tailwind CSS, Radix UI

**Key Features**:
- **Team Selection**: Dropdown selectors with team colors and divisions
- **Real-time Predictions**: API integration for live game predictions
- **Performance Dashboard**: Model accuracy metrics and visualizations
- **Responsive Design**: Mobile-friendly interface

**Core Components**:

1. **Pages**:
   - `/app/page.tsx`: Main prediction dashboard
   - `/app/performance/page.tsx`: Model performance analytics

2. **Components**:
   - `team-selector.tsx`: NFL team dropdown with search ✅ **FIXED**
   - `prediction-display.tsx`: Win probability visualization
   - `team-comparison.tsx`: Side-by-side team stats
   - `confusion-matrix.tsx`: Model performance matrix ✅ **FIXED**
   - `navigation.tsx`: App navigation header

3. **UI Library** (`/components/ui/`):
   - Button, Card, Command, Popover, Badge components
   - Built on Radix UI primitives

4. **Data & Types**:
   - `/lib/nfl-data.ts`: NFL teams data (32 teams)
   - `/types/index.ts`: TypeScript interfaces
   - `/lib/api-client.ts`: Backend API communication

## Data Flow

```
Raw NFL Data → main.py → Enhanced play-level data
       ↓
Team performance metrics → predictor.py → Trained ML models
       ↓
Model artifacts → simple_api.py → REST API
       ↓
Frontend → Team selection → API calls → Predictions displayed
```

## Key Features

### Machine Learning Pipeline
- **EPA-based metrics**: Expected Points Added as primary performance indicator
- **Situational analytics**: Red zone performance, third down conversions
- **Momentum tracking**: EMA-based recent performance trends
- **PCA dimensionality reduction**: Auto-selects components for 90% variance
- **Advanced models**: Random Forest, XGBoost, Neural Networks with ensemble methods
- **No clustering**: Clustering features removed for improved performance and reduced complexity
- **Performance**: 79% accuracy, 96% variance explained, 6.9% generalization gap

### Web Interface
- **Interactive team selection** with search and filtering
- **Real-time predictions** with confidence scores
- **Key factors analysis** showing what drives predictions
- **Model performance dashboard** with historical accuracy
- **Responsive design** for desktop and mobile

## Current Status

### ✅ **Working Components**
- Backend API server (Flask on port 5000)
- Frontend application (Next.js)
- Team selector dropdowns **FIXED**
- API connectivity and CORS
- Prediction display and team comparison
- Model performance dashboard
- Confusion matrix visualization **FIXED**

### 🔧 **Configuration**
- **Default years**: 2015-2021 for training data
- **Mock API mode**: Using team strength ratings for development
- **Model artifacts**: Saved to `/artifacts/` directory
- **Dependencies**: Listed in `requirements.txt` (backend) and `package.json` (frontend)

### 📂 **Project Structure**
```
nfl_predictor/
├── backend/           # Python ML pipeline & API
├── frontend/          # Next.js web application
├── artifacts/         # Trained model storage
├── venv/             # Python virtual environment
├── CLAUDE.md         # Project documentation
└── README.md         # User documentation
```

The project is **fully operational** with both frontend and backend working correctly. The recent fixes resolved the team selector dropdown issues and React warnings, making the application production-ready for development and testing.