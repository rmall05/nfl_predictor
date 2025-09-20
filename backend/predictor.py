import os
import numpy as np
import pandas as pd
import nfl_data_py as nfl
from typing import List, Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from joblib import dump
import xgboost as xgb

from main import (
    load_and_prepare_pbp,
    offensive_stats,
    defensive_stats,
    add_momentum_simple,
)

###############################################################################
# Streamlined Dataset Assembly
###############################################################################

def assemble_team_game_dataset(
    years: List[int],
    include_momentum: bool = True,
    momentum_metric: str = "explosive_rate",  # Optimal from testing
    ema_span: int = 5,  # Optimal from testing
) -> pd.DataFrame:
    """
    Streamlined dataset assembly with fixed optimal momentum configuration.
    """
    # 1) Build base tables
    print(f"Loading play-by-play data for years: {years}")
    season = load_and_prepare_pbp(years)
    print(f"Loaded {len(season)} plays")

    print("Computing offensive stats...")
    off = offensive_stats(season)
    print(f"Generated {len(off)} offensive team-game records")

    print("Computing defensive stats...")
    defe = defensive_stats(season)
    print(f"Generated {len(defe)} defensive team-game records")

    # 2) Import schedules
    print("Loading NFL schedules...")
    sched = nfl.import_schedules(years=years)[
        ["game_id","home_team","away_team","home_score","away_score","season","week"]
    ].dropna(subset=["game_id"])
    print(f"Loaded {len(sched)} scheduled games")

    # 3) Add opponent to offense table
    off = off.drop_duplicates(subset=["season","week","game_id","team"], keep="first")
    off = off.merge(sched[["game_id","home_team","away_team"]], on="game_id", how="left")
    off["opponent"] = np.where(off["team"] == off["home_team"], off["away_team"], off["home_team"])
    off = off.drop(columns=["home_team","away_team"])

    # 4) Prep defense table: rename team→opponent and dedupe
    defe_ren = (
        defe.rename(columns={"team": "opponent"})
            .drop_duplicates(subset=["season","week","game_id","opponent"], keep="first")
    )

    # 5) Check for duplicates and merge offense with opponent defense
    keys = ["season","week","game_id","opponent"]

    # Remove any remaining duplicates
    off = off.drop_duplicates(subset=keys, keep="first")
    defe_ren = defe_ren.drop_duplicates(subset=keys, keep="first")

    df = off.merge(defe_ren, on=keys, how="left", suffixes=("", "_def"))

    # 6) Add labels
    scores = sched.rename(columns={"home_team":"home","away_team":"away"})
    df = df.merge(scores[["game_id","home","away","home_score","away_score"]], on="game_id", how="left")
    df["team_is_home"] = (df["team"] == df["home"]).astype(int)
    df["team_score"] = np.where(df["team_is_home"] == 1, df["home_score"], df["away_score"])
    df["opp_score"] = np.where(df["team_is_home"] == 1, df["away_score"], df["home_score"])
    df["point_diff"] = df["team_score"] - df["opp_score"]
    df["win"] = (df["point_diff"] > 0).astype(int)

    # 7) Add Strength of Schedule Features
    print("Computing strength of schedule features...")
    df = calculate_opponent_strength(df)

    # 8) Add optimal momentum (fixed configuration)
    if include_momentum:
        df = df.sort_values(["team","season","week"]).reset_index(drop=True)
        df = add_momentum_simple(df, metric=momentum_metric, ema_span=ema_span)
        df["momentum_score"] = df["momentum_score"].fillna(0.0).round(4)

    # 9) Clean column order
    base_keys = ["season","week","game_id","team","opponent","win","point_diff"]
    cols = base_keys + [c for c in df.columns if c not in base_keys]
    return df[cols]


def calculate_opponent_strength(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add strength of schedule features.
    """
    df = df.copy().sort_values(["team", "season", "week"]).reset_index(drop=True)
    team_metrics = []

    for season in df["season"].unique():
        season_data = df[df["season"] == season].copy()

        for week in sorted(season_data["week"].unique()):
            # Get data up to (but not including) current week
            historical_data = season_data[season_data["week"] < week]

            if len(historical_data) == 0:
                # No historical data - use neutral values
                week_data = season_data[season_data["week"] == week].copy()
                week_data["opp_season_epa_avg"] = 0.0
                week_data["opp_recent_form_epa"] = 0.0
                week_data["opp_win_pct_to_date"] = 0.5
                week_data["opp_strength_ranking"] = 16.5
            else:
                # Calculate opponent strength metrics
                team_season_stats = historical_data.groupby("team").agg({
                    "epa_per_play": "mean",
                    "win": "mean",
                }).reset_index()

                # Recent form (last 4 games)
                recent_form = []
                for team in team_season_stats["team"].unique():
                    team_historical = historical_data[historical_data["team"] == team]
                    if len(team_historical) >= 4:
                        recent_epa = team_historical.tail(4)["epa_per_play"].mean()
                    elif len(team_historical) > 0:
                        recent_epa = team_historical["epa_per_play"].mean()
                    else:
                        recent_epa = 0.0
                    recent_form.append({"team": team, "recent_form_epa": recent_epa})

                recent_form_df = pd.DataFrame(recent_form)
                team_season_stats = team_season_stats.merge(recent_form_df, on="team", how="left")

                # Calculate strength rankings
                team_season_stats = team_season_stats.sort_values("epa_per_play", ascending=False)
                team_season_stats["strength_ranking"] = range(1, len(team_season_stats) + 1)

                # Get current week data and merge opponent strength
                week_data = season_data[season_data["week"] == week].copy()
                week_data = week_data.merge(
                    team_season_stats[["team", "epa_per_play", "win", "recent_form_epa", "strength_ranking"]],
                    left_on="opponent", right_on="team",
                    how="left", suffixes=("", "_opp")
                )

                # Rename and clean up columns
                week_data = week_data.rename(columns={
                    "epa_per_play_opp": "opp_season_epa_avg",
                    "win_opp": "opp_win_pct_to_date",
                    "recent_form_epa": "opp_recent_form_epa",
                    "strength_ranking": "opp_strength_ranking"
                })
                week_data = week_data.drop(columns=["team_opp"], errors="ignore")

                # Fill missing opponent data with neutral values
                week_data["opp_season_epa_avg"] = week_data["opp_season_epa_avg"].fillna(0.0)
                week_data["opp_recent_form_epa"] = week_data["opp_recent_form_epa"].fillna(0.0)
                week_data["opp_win_pct_to_date"] = week_data["opp_win_pct_to_date"].fillna(0.5)
                week_data["opp_strength_ranking"] = week_data["opp_strength_ranking"].fillna(16.5)

            team_metrics.append(week_data)

    # Combine all weeks
    enhanced_df = pd.concat(team_metrics, ignore_index=True)

    # Add composite strength score
    enhanced_df["opp_composite_strength"] = (
        enhanced_df["opp_season_epa_avg"] * 0.4 +
        enhanced_df["opp_recent_form_epa"] * 0.3 +
        (enhanced_df["opp_win_pct_to_date"] - 0.5) * 0.2 +
        (16.5 - enhanced_df["opp_strength_ranking"]) / 32 * 0.1
    )

    return enhanced_df.sort_values(["season", "week", "game_id", "team"]).reset_index(drop=True)


###############################################################################
# Streamlined Feature Selection
###############################################################################

def select_feature_columns(df: pd.DataFrame, include_momentum: bool = True) -> List[str]:
    """
    Choose numeric feature columns for modeling (exclude IDs and labels).
    """
    exclude = {
        "season", "week", "game_id", "team", "opponent",
        "home", "away", "home_score", "away_score", "team_is_home",
        "team_score", "opp_score", "win", "point_diff"
    }

    if not include_momentum and "momentum_score" in df.columns:
        exclude.add("momentum_score")

    # Keep only numeric, minus exclusions
    feats = [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]
    return feats


###############################################################################
# Streamlined Pipeline with Fixed XGBoost
###############################################################################

def fit_streamlined_pipeline(
    df: pd.DataFrame,
    include_momentum: bool = True,
    task: str = "classification",
    train_seasons: List[int] = None,
    val_seasons: List[int] = None,
    test_seasons: List[int] = None,
    pca_components: int = 15,  # Optimal from testing
) -> Tuple[Pipeline, dict, dict]:
    """
    Streamlined pipeline using only XGBoost with optimal hyperparameters.

    Fixed optimal XGBoost parameters from testing:
    - n_estimators: 100
    - max_depth: 3
    - learning_rate: 0.1
    - subsample: 0.8
    - colsample_bytree: 0.9
    - pca_components: 15
    """
    # Default temporal splits
    available_seasons = sorted(df["season"].unique())
    n_seasons = len(available_seasons)

    if n_seasons < 5:
        raise ValueError(f"Need at least 5 seasons for train/val/test split, got {n_seasons}")

    # Default splits: Train(2015-2020), Val(2021-2022), Test(2023-2024)
    if train_seasons is None:
        train_seasons = available_seasons[:6]
    if val_seasons is None:
        val_seasons = available_seasons[6:8] if n_seasons > 7 else available_seasons[6:7]
    if test_seasons is None:
        test_seasons = available_seasons[8:] if n_seasons > 8 else available_seasons[7:]

    print(f"Streamlined Pipeline Configuration:")
    print(f"Training seasons ({len(train_seasons)}): {train_seasons}")
    print(f"Validation seasons ({len(val_seasons)}): {val_seasons}")
    print(f"Test seasons ({len(test_seasons)}): {test_seasons}")

    # Create masks for splits
    train_mask = df["season"].isin(train_seasons)
    val_mask = df["season"].isin(val_seasons)
    test_mask = df["season"].isin(test_seasons)

    X_cols = select_feature_columns(df, include_momentum=include_momentum)
    print(f"Using {len(X_cols)} features")

    # Prepare data splits
    X_train = df.loc[train_mask, X_cols].fillna(0)
    X_val = df.loc[val_mask, X_cols].fillna(0)
    X_test = df.loc[test_mask, X_cols].fillna(0)

    # Prepare target variables
    if task == "classification":
        y_train = df.loc[train_mask, "win"].values
        y_val = df.loc[val_mask, "win"].values
        y_test = df.loc[test_mask, "win"].values
    else:
        y_train = df.loc[train_mask, "point_diff"].values
        y_val = df.loc[val_mask, "point_diff"].values
        y_test = df.loc[test_mask, "point_diff"].values

    # Build pipeline with optimal components
    print(f"Building XGBoost pipeline with {pca_components} PCA components...")

    scaler = StandardScaler()
    pca = PCA(n_components=pca_components, random_state=42)

    # Fixed optimal XGBoost model
    if task == "classification":
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.9,
            random_state=42,
            eval_metric='logloss'
        )
    else:
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.9,
            random_state=42
        )

    # Fit pipeline
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    print("Training XGBoost model...")
    model.fit(X_train_pca, y_train)

    # Comprehensive evaluation
    performance_metrics = {
        "model_type": "xgboost",
        "pca_components": pca_components,
        "features_used": len(X_cols),
        "explained_variance": pca.explained_variance_ratio_.sum(),
        "include_momentum": include_momentum,
        "optimal_config": True
    }

    # Evaluate on all splits
    for split_name, X_split, y_split in [
        ("train", X_train_pca, y_train),
        ("validation", X_val_pca, y_val),
        ("test", X_test_pca, y_test)
    ]:
        if task == "classification":
            preds = model.predict(X_split)
            probs = model.predict_proba(X_split)[:, 1]
            performance_metrics[f"{split_name}_accuracy"] = accuracy_score(y_split, preds)
            performance_metrics[f"{split_name}_roc_auc"] = roc_auc_score(y_split, probs)
        else:
            preds = model.predict(X_split)
            performance_metrics[f"{split_name}_mae"] = mean_absolute_error(y_split, preds)

    # Calculate generalization gaps
    if task == "classification":
        performance_metrics["train_val_gap_acc"] = performance_metrics["train_accuracy"] - performance_metrics["validation_accuracy"]
        performance_metrics["train_val_gap_auc"] = performance_metrics["train_roc_auc"] - performance_metrics["validation_roc_auc"]
        if len(test_seasons) > 0:
            performance_metrics["val_test_gap_acc"] = performance_metrics["validation_accuracy"] - performance_metrics["test_accuracy"]
            performance_metrics["val_test_gap_auc"] = performance_metrics["validation_roc_auc"] - performance_metrics["test_roc_auc"]

    # Create data splits dictionary
    data_splits = {
        "X_train_pca": X_train_pca,
        "X_val_pca": X_val_pca,
        "X_test_pca": X_test_pca,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "feature_names": X_cols,
        "train_seasons": train_seasons,
        "val_seasons": val_seasons,
        "test_seasons": test_seasons
    }

    # Build final pipeline
    pipeline = Pipeline([
        ("scaler", scaler),
        ("pca", pca),
        ("model", model)
    ])

    return pipeline, data_splits, performance_metrics


###############################################################################
# 2025 Season Prediction Support
###############################################################################

def predict_new_season_games(
    pipeline: Pipeline,
    feature_names: List[str],
    new_season_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Predict outcomes for new season games using trained pipeline.

    Args:
        pipeline: Trained sklearn Pipeline
        feature_names: List of feature column names used in training
        new_season_data: DataFrame with new season team-game data

    Returns:
        DataFrame with predictions added
    """
    # Extract features for prediction
    X_new = new_season_data[feature_names].fillna(0)

    # Get predictions
    predictions = pipeline.predict(X_new)
    probabilities = pipeline.predict_proba(X_new)[:, 1]  # Win probability

    # Add predictions to dataframe
    result = new_season_data.copy()
    result["predicted_win"] = predictions
    result["win_probability"] = probabilities

    return result


def prepare_2025_data(weeks_to_predict: List[int] = None) -> pd.DataFrame:
    """
    Prepare 2025 season data for prediction.

    Args:
        weeks_to_predict: List of week numbers to predict (e.g., [1, 2, 3])

    Returns:
        DataFrame ready for prediction
    """
    try:
        # Load 2025 season data (when available)
        print("Loading 2025 NFL data...")
        data_2025 = assemble_team_game_dataset(
            years=[2025],
            include_momentum=True,
            momentum_metric="explosive_rate",
            ema_span=5
        )

        if weeks_to_predict:
            data_2025 = data_2025[data_2025["week"].isin(weeks_to_predict)]

        print(f"Prepared {len(data_2025)} games for 2025 prediction")
        return data_2025

    except Exception as e:
        print(f"Error loading 2025 data: {e}")
        print("2025 season data may not be available yet")
        return pd.DataFrame()


###############################################################################
# Entry point
###############################################################################

if __name__ == "__main__":
    # Streamlined configuration
    YEARS = list(range(2015, 2025))  # 2015-2024 for training
    INCLUDE_MOMENTUM = True
    TASK = "classification"

    try:
        print(">> Assembling streamlined dataset...")
        print(f"Years: {YEARS}")
        print(f"Task: {TASK}")
        print(f"Include momentum: {INCLUDE_MOMENTUM}")

        # Assemble dataset with optimal momentum config
        data = assemble_team_game_dataset(
            years=YEARS,
            include_momentum=INCLUDE_MOMENTUM,
            momentum_metric="explosive_rate",  # Optimal from testing
            ema_span=5,  # Optimal from testing
        )

        print("Rows:", len(data), "Cols:", len(data.columns))

        if len(data) == 0:
            raise ValueError("Dataset is empty - check data availability")

        print(">> Fitting streamlined XGBoost pipeline...")
        pipeline, data_splits, performance_metrics = fit_streamlined_pipeline(
            data,
            include_momentum=INCLUDE_MOMENTUM,
            task=TASK
        )

        print("\n" + "="*60)
        print("STREAMLINED PIPELINE RESULTS")
        print("="*60)

        # Print performance metrics
        if TASK == "classification":
            print(f"\nPERFORMANCE:")
            print(f"Training:   Accuracy={performance_metrics['train_accuracy']:.4f}, ROC-AUC={performance_metrics['train_roc_auc']:.4f}")
            print(f"Validation: Accuracy={performance_metrics['validation_accuracy']:.4f}, ROC-AUC={performance_metrics['validation_roc_auc']:.4f}")
            if 'test_accuracy' in performance_metrics:
                print(f"Test:       Accuracy={performance_metrics['test_accuracy']:.4f}, ROC-AUC={performance_metrics['test_roc_auc']:.4f}")

        print(f"\nCONFIGURATION:")
        print(f"Model: XGBoost (optimal hyperparameters)")
        print(f"Features: {performance_metrics['features_used']}")
        print(f"PCA components: {performance_metrics['pca_components']}")
        print(f"Explained variance: {performance_metrics['explained_variance']:.3f}")
        print(f"Momentum: explosive_rate (5-game EMA)")

        # Save streamlined artifacts
        os.makedirs("artifacts", exist_ok=True)
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

        pipeline_filename = f"artifacts/streamlined_pipeline_{TASK}_{timestamp}.joblib"
        features_filename = f"artifacts/streamlined_features_{timestamp}.csv"
        components_filename = f"artifacts/streamlined_components_{timestamp}.csv"
        metrics_filename = f"artifacts/streamlined_metrics_{timestamp}.json"

        # Save pipeline
        dump(pipeline, pipeline_filename)

        # Save features
        pd.Series(data_splits['feature_names']).to_csv(features_filename, index=False)

        # Save PCA components
        pca = pipeline.named_steps["pca"]
        comp_df = pd.DataFrame(pca.components_, columns=data_splits['feature_names'])
        comp_df.to_csv(components_filename, index=False)

        # Save metrics
        import json
        metrics_to_save = {}
        for key, value in performance_metrics.items():
            if isinstance(value, (np.integer, np.floating)):
                metrics_to_save[key] = value.item()
            else:
                metrics_to_save[key] = value

        with open(metrics_filename, 'w') as f:
            json.dump(metrics_to_save, f, indent=2)

        print(f"\nARTIFACTS SAVED:")
        print(f"Pipeline: {pipeline_filename}")
        print(f"Features: {features_filename}")
        print(f"Components: {components_filename}")
        print(f"Metrics: {metrics_filename}")

        print(f"\nStreamlined pipeline completed successfully!")
        print(f"Test Accuracy: {performance_metrics.get('test_accuracy', 'N/A'):.1%}")
        print("="*60)

    except Exception as e:
        print(f"Error during pipeline execution: {e}")
        import traceback
        traceback.print_exc()
        exit(1)