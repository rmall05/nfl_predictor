# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an NFL game outcome prediction system that uses advanced analytics and machine learning. The project analyzes NFL play-by-play data to generate team performance metrics and predict game outcomes using PCA-based feature engineering.

## Architecture

### Core Components

- **main.py**: Data processing pipeline that handles NFL play-by-play data
  - `load_and_prepare_pbp()`: Downloads and preprocesses play-by-play data from nfl_data_py
  - `offensive_stats()`: Aggregates per-team, per-game offensive metrics (EPA, success rates, etc.)
  - `defensive_stats()`: Aggregates per-team, per-game defensive metrics (opponent stats allowed)
  - `add_momentum_simple()`: Calculates momentum scores using EMA and Z-score normalization

- **predictor.py**: Machine learning pipeline for game outcome prediction
  - `assemble_team_game_dataset()`: Combines offensive/defensive stats with game outcomes
  - `fit_pca_pipeline()`: Implements StandardScaler + PCA + LogisticRegression/LinearRegression
  - Supports both classification (win/loss) and regression (point differential) tasks
  - Uses season-based train/test splits (typically train on early seasons, test on later ones)

### Data Flow

1. Raw NFL play-by-play data → `load_and_prepare_pbp()` → Enhanced play-level data
2. Play-level data → `offensive_stats()` + `defensive_stats()` → Team-game performance metrics
3. Team-game metrics + schedules → `assemble_team_game_dataset()` → Model-ready dataset
4. Model-ready dataset → `fit_pca_pipeline()` → Trained PCA + prediction model

### Key Features

- **EPA-based metrics**: Uses Expected Points Added as primary performance metric
- **Situational analytics**: Red zone performance, third down conversions, explosive plays
- **Momentum tracking**: Short-term vs season-long performance trends using EMA
- **Defensive analytics**: What teams allow on defense (opponent perspective)
- **PCA dimensionality reduction**: Auto-selects components for 90% explained variance

## Development Commands

### Environment Setup
```bash
# Create and activate virtual environment (if not exists)
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Running the Code
```bash
# Run main data processing (testing/development)
python main.py

# Run full prediction pipeline
python predictor.py
```

### Dependencies
- pandas==2.2.3: Data manipulation
- numpy: Numerical operations
- nfl_data_py: NFL data API
- scikit-learn: Machine learning pipeline
- joblib: Model serialization
- matplotlib: Plotting (optional)

## Important Notes

- The project uses `nfl_data_py` which downloads data from the internet on first run
- Models are saved to `artifacts/` directory when pipeline runs
- Default years are 2015-2021 but can be adjusted in predictor.py configuration
- PCA components and feature importance are saved for interpretability
- Train/test splits are season-based to avoid data leakage