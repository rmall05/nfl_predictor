import os
import numpy as np
import pandas as pd
import nfl_data_py as nfl
from typing import List, Tuple, Optional, Dict

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from joblib import dump
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

from main import (
    load_and_prepare_pbp,
    offensive_stats,
    defensive_stats,
    add_momentum_simple,
)

###############################################################################
# Dataset assembly
###############################################################################

def assemble_team_game_dataset(
    years: List[int],
    include_momentum: bool = True,
    momentum_metric: str = "epa_per_play",
    ema_span: int = 5,
) -> pd.DataFrame:
    """
    Builds one row per team-game with:
      - offensive stats (offensive_stats)
      - defensive allowed stats for the OPPONENT (defensive_stats)
      - opponent & labels (win, point_diff) from schedules
      - optional momentum (via add_momentum_simple on the offense metric)

    Returns a tidy DataFrame ready for modeling.
    """
    # 1) Build base tables
    try:
        print(f"Loading play-by-play data for years: {years}")
        season = load_and_prepare_pbp(years)
        if len(season) == 0:
            raise ValueError(f"No play-by-play data found for years {years}")
        print(f"Loaded {len(season)} plays")

        print("Computing offensive stats...")
        off   = offensive_stats(season)   # posteam-based stats
        print(f"Generated {len(off)} offensive team-game records")

        print("Computing defensive stats...")
        defe  = defensive_stats(season)   # defteam-based "allowed" stats
        print(f"Generated {len(defe)} defensive team-game records")

    except Exception as e:
        print(f"Error loading/processing NFL data: {e}")
        raise

    # 2) Import schedules
    try:
        print("Loading NFL schedules...")
        sched = nfl.import_schedules(years=years)[
            ["game_id","home_team","away_team","home_score","away_score","season","week"]
        ].dropna(subset=["game_id"])
        print(f"Loaded {len(sched)} scheduled games")
    except Exception as e:
        print(f"Error loading NFL schedules: {e}")
        raise

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

    # 5) Check uniqueness
    keys = ["season","week","game_id","opponent"]

    # Debug duplicates in offense table
    off_dupes = off.duplicated(subset=keys, keep=False)
    if off_dupes.any():
        print(f"Found {off_dupes.sum()} duplicate rows in offense table:")
        print(off.loc[off_dupes, keys + ["team"]].head(10))
        # Remove duplicates by taking first occurrence
        off = off.drop_duplicates(subset=keys, keep="first")
        print(f"Removed duplicates, now have {len(off)} offense rows")

    # Debug duplicates in defense table
    defe_dupes = defe_ren.duplicated(subset=keys, keep=False)
    if defe_dupes.any():
        print(f"Found {defe_dupes.sum()} duplicate rows in defense table:")
        print(defe_ren.loc[defe_dupes, keys].head(10))
        # Remove duplicates by taking first occurrence
        defe_ren = defe_ren.drop_duplicates(subset=keys, keep="first")
        print(f"Removed duplicates, now have {len(defe_ren)} defense rows")

    # 6) Merge offense with opponent defense
    df = off.merge(
        defe_ren,
        on=keys,
        how="left",
        validate="1:1",
        suffixes=("", "_def")
    )

    # 7) Labels
    scores = sched.rename(columns={"home_team":"home","away_team":"away"})
    df = df.merge(scores[["game_id","home","away","home_score","away_score"]], on="game_id", how="left")
    df["team_is_home"] = (df["team"] == df["home"]).astype(int)
    df["team_score"]   = np.where(df["team_is_home"] == 1, df["home_score"], df["away_score"])
    df["opp_score"]    = np.where(df["team_is_home"] == 1, df["away_score"], df["home_score"])
    df["point_diff"]   = df["team_score"] - df["opp_score"]
    df["win"]          = (df["point_diff"] > 0).astype(int)

    # 8) Add Strength of Schedule Features
    print("Computing strength of schedule features...")
    df = calculate_opponent_strength(df)

    # 9) Momentum (optional)
    if include_momentum:
        df = df.sort_values(["team","season","week"]).reset_index(drop=True)
        df = add_momentum_simple(df, metric=momentum_metric, ema_span=ema_span)
        df["momentum_score"] = df["momentum_score"].fillna(0.0).round(4)

    # 10) Clean column order
    base_keys = ["season","week","game_id","team","opponent","win","point_diff"]
    cols = base_keys + [c for c in df.columns if c not in base_keys]
    return df[cols]



###############################################################################
# Strength of Schedule Features
###############################################################################

def calculate_opponent_strength(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add strength of schedule features including:
    - opponent's season EPA average
    - opponent's recent form (last 4 games)
    - opponent's win percentage to date
    - opponent's relative strength ranking
    """
    df = df.copy().sort_values(["team", "season", "week"]).reset_index(drop=True)

    # Calculate rolling team strength metrics (avoid data leakage by excluding current game)
    team_metrics = []

    for season in df["season"].unique():
        season_data = df[df["season"] == season].copy()

        for week in sorted(season_data["week"].unique()):
            # Get data up to (but not including) current week for strength calculation
            historical_data = season_data[season_data["week"] < week]

            if len(historical_data) == 0:
                # No historical data available - use league averages or neutral values
                week_data = season_data[season_data["week"] == week].copy()
                week_data["opp_season_epa_avg"] = 0.0
                week_data["opp_recent_form_epa"] = 0.0
                week_data["opp_win_pct_to_date"] = 0.5
                week_data["opp_strength_ranking"] = 16.5  # middle ranking
            else:
                # Calculate opponent strength metrics based on historical performance
                team_season_stats = historical_data.groupby("team").agg({
                    "epa_per_play": "mean",
                    "win": "mean",
                    "epa_total": "sum",
                    "plays": "sum"
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

    # Add composite strength score (lower is better for ranking, so invert)
    enhanced_df["opp_composite_strength"] = (
        enhanced_df["opp_season_epa_avg"] * 0.4 +  # Historical performance
        enhanced_df["opp_recent_form_epa"] * 0.3 +  # Recent form
        (enhanced_df["opp_win_pct_to_date"] - 0.5) * 0.2 +  # Win rate above/below .500
        (16.5 - enhanced_df["opp_strength_ranking"]) / 32 * 0.1  # Normalized inverted ranking
    )

    print(f"Added strength of schedule features:")
    print(f"  - opp_season_epa_avg: opponent's season EPA average")
    print(f"  - opp_recent_form_epa: opponent's recent form (last 4 games)")
    print(f"  - opp_win_pct_to_date: opponent's win percentage to date")
    print(f"  - opp_strength_ranking: opponent's strength ranking (1=best)")
    print(f"  - opp_composite_strength: composite strength score")

    return enhanced_df.sort_values(["season", "week", "game_id", "team"]).reset_index(drop=True)


###############################################################################
# Enhanced Momentum Framework
###############################################################################

def test_momentum_configurations(df: pd.DataFrame) -> dict:
    """
    Test different momentum configurations to find optimal settings.

    Returns dictionary with performance results for each configuration.
    """
    print("\n>> Testing momentum configurations...")

    momentum_configs = [
        # (metric, ema_span, description)
        ("epa_per_play", 1, "EPA 1-game"),
        ("epa_per_play", 3, "EPA 3-game"),
        ("epa_per_play", 5, "EPA 5-game"),
        ("epa_per_play", 7, "EPA 7-game"),
        ("success_rate", 3, "Success Rate 3-game"),
        ("success_rate", 5, "Success Rate 5-game"),
        ("explosive_rate", 3, "Explosive Rate 3-game"),
        ("explosive_rate", 5, "Explosive Rate 5-game"),
    ]

    config_results = {}

    for metric, span, description in momentum_configs:
        try:
            # Add momentum with current configuration
            df_with_momentum = add_momentum_simple(df.copy(), metric=metric, ema_span=span)

            # Quick validation split for testing (use last season)
            test_seasons = [df_with_momentum["season"].max()]
            train_seasons = [s for s in df_with_momentum["season"].unique() if s not in test_seasons]

            if len(train_seasons) < 2:
                continue

            train_mask = df_with_momentum["season"].isin(train_seasons)
            test_mask = df_with_momentum["season"].isin(test_seasons)

            X_cols = select_feature_columns(df_with_momentum, include_momentum=True)

            X_train = df_with_momentum.loc[train_mask, X_cols].fillna(0)
            X_test = df_with_momentum.loc[test_mask, X_cols].fillna(0)
            y_train = df_with_momentum.loc[train_mask, "win"].values
            y_test = df_with_momentum.loc[test_mask, "win"].values

            # Quick model evaluation
            scaler = StandardScaler()
            pca = PCA(n_components=10, random_state=42)
            model = LogisticRegression(max_iter=200, random_state=42)

            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            X_train_pca = pca.fit_transform(X_train_scaled)
            X_test_pca = pca.transform(X_test_scaled)

            model.fit(X_train_pca, y_train)
            test_accuracy = accuracy_score(y_test, model.predict(X_test_pca))
            test_auc = roc_auc_score(y_test, model.predict_proba(X_test_pca)[:, 1])

            config_results[description] = {
                "metric": metric,
                "ema_span": span,
                "test_accuracy": test_accuracy,
                "test_auc": test_auc
            }

            print(f"  {description}: Acc={test_accuracy:.3f}, AUC={test_auc:.3f}")

        except Exception as e:
            print(f"  {description}: Error - {e}")
            continue

    # Find best configuration
    if config_results:
        best_config = max(config_results.items(), key=lambda x: x[1]["test_auc"])
        print(f"\nBest momentum config: {best_config[0]} (AUC: {best_config[1]['test_auc']:.3f})")
        return config_results, best_config
    else:
        print("No valid momentum configurations found")
        return {}, None


def add_composite_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add composite momentum score combining multiple metrics.
    """
    df = df.copy()

    # Add individual momentum scores
    df_epa = add_momentum_simple(df, metric="epa_per_play", ema_span=3)
    df_success = add_momentum_simple(df, metric="success_rate", ema_span=3)
    df_explosive = add_momentum_simple(df, metric="explosive_rate", ema_span=5)

    # Combine momentum scores with weights
    df["composite_momentum"] = (
        df_epa["momentum_score"] * 0.5 +      # EPA momentum (most important)
        df_success["momentum_score"] * 0.3 +  # Success rate momentum
        df_explosive["momentum_score"] * 0.2  # Explosive play momentum
    ).fillna(0.0).round(4)

    print(f"Added composite momentum score combining EPA (50%), success rate (30%), explosive plays (20%)")

    return df


###############################################################################
# Feature selection helpers
###############################################################################

def select_feature_columns(df: pd.DataFrame, include_momentum: bool) -> List[str]:
    """
    Choose numeric feature columns for PCA/modeling (exclude IDs and labels).
    Adjust the lists to match your exact offensive/defense columns.
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
# Advanced Modeling Framework (Phase 2)
###############################################################################

def time_series_cross_validation(df: pd.DataFrame, n_splits: int = 3) -> List[Tuple]:
    """
    Implement time-series aware cross-validation with proper temporal ordering.

    Creates sequential train/validation splits respecting chronological order.
    """
    available_seasons = sorted(df["season"].unique())
    n_seasons = len(available_seasons)

    if n_seasons < n_splits + 2:
        raise ValueError(f"Need at least {n_splits + 2} seasons for {n_splits}-fold CV")

    cv_splits = []

    # Create sequential splits
    for i in range(n_splits):
        # Calculate split boundaries
        train_end_idx = (i + 1) * (n_seasons - 2) // n_splits + 1
        val_start_idx = train_end_idx
        val_end_idx = min(val_start_idx + 1, n_seasons - 1)

        train_seasons = available_seasons[:train_end_idx]
        val_seasons = available_seasons[val_start_idx:val_end_idx + 1]

        train_mask = df["season"].isin(train_seasons)
        val_mask = df["season"].isin(val_seasons)

        cv_splits.append((train_mask, val_mask, train_seasons, val_seasons))

        print(f"CV Split {i+1}: Train={train_seasons}, Val={val_seasons}")

    return cv_splits


def optimize_hyperparameters(
    df: pd.DataFrame,
    clustering_enabled: bool = False,
    task: str = "classification"
) -> Dict:
    """
    Comprehensive hyperparameter optimization using time-series CV.

    Optimizes:
    - PCA components (5-25 range)
    - Model parameters (C for LogReg, max_depth for alternatives)
    - Clustering parameters (number of clusters)
    """
    print(f">> Starting hyperparameter optimization...")

    # Parameter grid
    param_grid = {
        'pca_components': [7, 11, 15, 19, 23],
        'model_C': [0.1, 1.0, 10.0],
        'team_clusters': [4, 6, 8] if clustering_enabled else [6],
        'context_clusters': [3, 4, 5] if clustering_enabled else [4]
    }

    best_score = -np.inf
    best_params = None
    cv_results = []

    # Get CV splits
    cv_splits = time_series_cross_validation(df, n_splits=3)

    total_combinations = (len(param_grid['pca_components']) *
                         len(param_grid['model_C']) *
                         len(param_grid['team_clusters']) *
                         len(param_grid['context_clusters']))

    print(f"Testing {total_combinations} parameter combinations across {len(cv_splits)} CV folds...")

    combination_count = 0

    for pca_comp in param_grid['pca_components']:
        for model_c in param_grid['model_C']:
            for team_clust in param_grid['team_clusters']:
                for context_clust in param_grid['context_clusters']:

                    combination_count += 1
                    current_params = {
                        'pca_components': pca_comp,
                        'model_C': model_c,
                        'team_clusters': team_clust,
                        'context_clusters': context_clust
                    }

                    # Cross-validation scores for this parameter combination
                    cv_scores = []

                    for fold_idx, (train_mask, val_mask, train_seasons, val_seasons) in enumerate(cv_splits):
                        try:
                            # Prepare data for this fold
                            df_fold = df.copy()

                            # Add clustering features if enabled
                            if clustering_enabled:
                                clusterer = NFLClustering(random_state=42)

                                # Fit clustering on training data only
                                train_data = df_fold[train_mask]
                                clusterer.fit_team_style_clustering(train_data, n_clusters=team_clust)
                                clusterer.fit_game_context_clustering(train_data, n_clusters=context_clust)

                                # Apply clustering to all data
                                df_fold = clusterer.add_cluster_features(df_fold)

                            # Feature selection
                            X_cols = select_feature_columns(df_fold, include_momentum=True)

                            X_train = df_fold.loc[train_mask, X_cols].fillna(0)
                            X_val = df_fold.loc[val_mask, X_cols].fillna(0)

                            if task == "classification":
                                y_train = df_fold.loc[train_mask, "win"].values
                                y_val = df_fold.loc[val_mask, "win"].values
                            else:
                                y_train = df_fold.loc[train_mask, "point_diff"].values
                                y_val = df_fold.loc[val_mask, "point_diff"].values

                            # Model pipeline
                            scaler = StandardScaler()
                            pca = PCA(n_components=pca_comp, random_state=42)

                            if task == "classification":
                                model = LogisticRegression(C=model_c, max_iter=200, random_state=42)
                            else:
                                model = LinearRegression()  # LinearRegression doesn't have C parameter

                            # Fit pipeline
                            X_train_scaled = scaler.fit_transform(X_train)
                            X_val_scaled = scaler.transform(X_val)
                            X_train_pca = pca.fit_transform(X_train_scaled)
                            X_val_pca = pca.transform(X_val_scaled)

                            model.fit(X_train_pca, y_train)

                            # Evaluate
                            if task == "classification":
                                val_probs = model.predict_proba(X_val_pca)[:, 1]
                                fold_score = roc_auc_score(y_val, val_probs)
                            else:
                                val_preds = model.predict(X_val_pca)
                                fold_score = -mean_absolute_error(y_val, val_preds)  # Negative for maximization

                            cv_scores.append(fold_score)

                        except Exception as e:
                            print(f"  Error in fold {fold_idx} with params {current_params}: {e}")
                            cv_scores.append(-1.0)  # Penalty score

                    # Average CV score
                    avg_cv_score = np.mean(cv_scores) if cv_scores else -1.0
                    cv_std = np.std(cv_scores) if len(cv_scores) > 1 else 0.0

                    cv_results.append({
                        'params': current_params.copy(),
                        'cv_score_mean': avg_cv_score,
                        'cv_score_std': cv_std,
                        'cv_scores': cv_scores
                    })

                    # Update best parameters
                    if avg_cv_score > best_score:
                        best_score = avg_cv_score
                        best_params = current_params.copy()

                    if combination_count % 5 == 0:
                        print(f"  Progress: {combination_count}/{total_combinations} combinations tested")

    print(f"Hyperparameter optimization completed!")
    print(f"Best CV score: {best_score:.4f}")
    print(f"Best parameters: {best_params}")

    return {
        'best_params': best_params,
        'best_score': best_score,
        'cv_results': cv_results,
        'total_combinations': total_combinations
    }


def fit_advanced_pipeline(
    df: pd.DataFrame,
    include_clustering: bool = False,
    hyperopt_enabled: bool = True,
    task: str = "classification",
    train_seasons: List[int] = None,
    val_seasons: List[int] = None,
    test_seasons: List[int] = None
) -> Tuple[Pipeline, dict, dict]:
    """
    Phase 2 Advanced Pipeline with clustering and hyperparameter optimization.
    """
    # Default temporal splits
    available_seasons = sorted(df["season"].unique())
    if train_seasons is None:
        train_seasons = available_seasons[:6]  # 2015-2020
    if val_seasons is None:
        val_seasons = available_seasons[6:8]   # 2021-2022
    if test_seasons is None:
        test_seasons = available_seasons[8:]   # 2023-2024

    print(f"Advanced Pipeline Configuration:")
    print(f"Training seasons: {train_seasons}")
    print(f"Validation seasons: {val_seasons}")
    print(f"Test seasons: {test_seasons}")

    # Phase 2.1 - Clustering
    clustered_df = df.copy()
    clustering_info = {}

    if include_clustering:
        print(f"\n>> Phase 2.1: Team & Game Context Clustering...")
        clusterer = NFLClustering(random_state=42)

        # Fit clustering on training data only
        train_mask = clustered_df["season"].isin(train_seasons)
        train_data = clustered_df[train_mask]

        team_results = clusterer.fit_team_style_clustering(train_data, n_clusters=6)
        context_results = clusterer.fit_game_context_clustering(train_data, n_clusters=4)

        # Apply clustering to full dataset
        clustered_df = clusterer.add_cluster_features(clustered_df)

        clustering_info = {
            'team_style_results': team_results,
            'context_results': context_results,
            'clusterer': clusterer
        }

    # Phase 2.2 - Hyperparameter Optimization (optional)
    best_params = {
        'pca_components': 17,  # Default from Phase 1.2
        'model_C': 1.0,
        'team_clusters': 6,
        'context_clusters': 4
    }

    if hyperopt_enabled:
        print(f"\n>> Phase 2.2: Hyperparameter Optimization...")
        hyperopt_results = optimize_hyperparameters(
            clustered_df,
            clustering_enabled=include_clustering,
            task=task
        )
        best_params.update(hyperopt_results['best_params'])
        clustering_info['hyperopt_results'] = hyperopt_results

    # Phase 2.3 - Final Model Training
    print(f"\n>> Phase 2.3: Training Final Advanced Model...")
    print(f"Using optimized parameters: {best_params}")

    # Apply final clustering with optimal parameters
    if include_clustering:
        final_clusterer = NFLClustering(random_state=42)
        train_mask = clustered_df["season"].isin(train_seasons)
        train_data = clustered_df[train_mask]

        final_clusterer.fit_team_style_clustering(train_data, n_clusters=best_params['team_clusters'])
        final_clusterer.fit_game_context_clustering(train_data, n_clusters=best_params['context_clusters'])
        clustered_df = final_clusterer.add_cluster_features(clustered_df)

    # Data preparation
    train_mask = clustered_df["season"].isin(train_seasons)
    val_mask = clustered_df["season"].isin(val_seasons)
    test_mask = clustered_df["season"].isin(test_seasons)

    X_cols = select_feature_columns(clustered_df, include_momentum=True)

    X_train = clustered_df.loc[train_mask, X_cols].fillna(0)
    X_val = clustered_df.loc[val_mask, X_cols].fillna(0)
    X_test = clustered_df.loc[test_mask, X_cols].fillna(0)

    if task == "classification":
        y_train = clustered_df.loc[train_mask, "win"].values
        y_val = clustered_df.loc[val_mask, "win"].values
        y_test = clustered_df.loc[test_mask, "win"].values
    else:
        y_train = clustered_df.loc[train_mask, "point_diff"].values
        y_val = clustered_df.loc[val_mask, "point_diff"].values
        y_test = clustered_df.loc[test_mask, "point_diff"].values

    # Final pipeline
    scaler = StandardScaler()
    pca = PCA(n_components=best_params['pca_components'], random_state=42)

    if task == "classification":
        model = LogisticRegression(C=best_params['model_C'], max_iter=200, random_state=42)
    else:
        model = LinearRegression()

    # Fit final pipeline
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    model.fit(X_train_pca, y_train)

    # Comprehensive evaluation
    performance_metrics = {
        'clustering_enabled': include_clustering,
        'hyperopt_enabled': hyperopt_enabled,
        'best_params': best_params,
        'features_used': len(X_cols),
        'explained_variance': pca.explained_variance_ratio_.sum()
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
        performance_metrics["val_test_gap_acc"] = performance_metrics["validation_accuracy"] - performance_metrics["test_accuracy"]
        performance_metrics["val_test_gap_auc"] = performance_metrics["validation_roc_auc"] - performance_metrics["test_roc_auc"]
    else:
        performance_metrics["train_val_gap_mae"] = performance_metrics["validation_mae"] - performance_metrics["train_mae"]
        performance_metrics["val_test_gap_mae"] = performance_metrics["test_mae"] - performance_metrics["validation_mae"]

    # Add clustering info to performance metrics
    performance_metrics.update(clustering_info)

    # Data splits
    data_splits = {
        'X_train_pca': X_train_pca,
        'X_val_pca': X_val_pca,
        'X_test_pca': X_test_pca,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'feature_names': X_cols,
        'train_seasons': train_seasons,
        'val_seasons': val_seasons,
        'test_seasons': test_seasons,
        'clustered_data': clustered_df
    }

    # Final pipeline
    pipeline = Pipeline([
        ("scaler", scaler),
        ("pca", pca),
        ("model", model)
    ])

    return pipeline, data_splits, performance_metrics


###############################################################################
# PCA pipeline
###############################################################################

def validate_data_quality(df: pd.DataFrame, split_info: dict) -> dict:
    """
    Comprehensive data quality validation framework.

    Returns validation report with issues and recommendations.
    """
    validation_report = {
        "issues": [],
        "warnings": [],
        "stats": {},
        "recommendations": []
    }

    # Check split integrity
    for split_name, seasons in split_info.items():
        split_data = df[df["season"].isin(seasons)]
        validation_report["stats"][f"{split_name}_games"] = len(split_data)
        validation_report["stats"][f"{split_name}_teams"] = split_data["team"].nunique()

        if len(split_data) == 0:
            validation_report["issues"].append(f"{split_name} split is empty")
        if split_data["team"].nunique() < 30:
            validation_report["warnings"].append(f"{split_name} has only {split_data['team'].nunique()} teams")

    # Check missing values
    missing_stats = df.isnull().sum()
    high_missing = missing_stats[missing_stats > len(df) * 0.1]
    if len(high_missing) > 0:
        validation_report["warnings"].append(f"High missing values in columns: {list(high_missing.index)}")
        validation_report["stats"]["high_missing_features"] = len(high_missing)

    # Check team coverage across seasons
    teams_per_season = df.groupby("season")["team"].nunique()
    inconsistent_teams = teams_per_season[teams_per_season < 30]
    if len(inconsistent_teams) > 0:
        validation_report["warnings"].append(f"Seasons with <30 teams: {dict(inconsistent_teams)}")

    # Check feature consistency
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].std() == 0:
            validation_report["issues"].append(f"Zero variance feature: {col}")
        elif np.isinf(df[col]).any():
            validation_report["issues"].append(f"Infinite values in: {col}")

    # Add recommendations
    if len(validation_report["issues"]) == 0 and len(validation_report["warnings"]) == 0:
        validation_report["recommendations"].append("Data quality looks good - proceed with modeling")
    else:
        validation_report["recommendations"].append("Review issues and warnings before proceeding")

    return validation_report


def fit_enhanced_pipeline(
    df: pd.DataFrame,
    include_momentum: bool,
    explained_variance_target: float = 0.9,
    task: str = "classification",
    train_seasons: List[int] = None,
    val_seasons: List[int] = None,
    test_seasons: List[int] = None
) -> Tuple[Pipeline, dict, dict]:
    """
    Enhanced pipeline with train/validation/test splits and comprehensive validation.

    Uses validation set for hyperparameter optimization instead of test set.

    Returns:
      - pipeline (scaler+PCA+model)
      - data_splits dict with all split data
      - performance_metrics dict with comprehensive results
    """
    # Default split configuration for 2015-2024 data
    available_seasons = sorted(df["season"].unique())
    n_seasons = len(available_seasons)

    if n_seasons < 5:
        raise ValueError(f"Need at least 5 seasons for train/val/test split, got {n_seasons}")

    # Default temporal splits: Train(2015-2020), Val(2021-2022), Test(2023-2024)
    if train_seasons is None:
        train_seasons = available_seasons[:6]  # First 6 seasons
    if val_seasons is None:
        val_seasons = available_seasons[6:8]   # Next 2 seasons
    if test_seasons is None:
        test_seasons = available_seasons[8:]   # Last 2+ seasons

    print(f"Enhanced Pipeline Configuration:")
    print(f"Training seasons ({len(train_seasons)}): {train_seasons}")
    print(f"Validation seasons ({len(val_seasons)}): {val_seasons}")
    print(f"Test seasons ({len(test_seasons)}): {test_seasons}")

    # Create split info for validation
    split_info = {
        "train": train_seasons,
        "validation": val_seasons,
        "test": test_seasons
    }

    # Run data quality validation
    print("\n>> Running data quality validation...")
    validation_report = validate_data_quality(df, split_info)

    # Print validation results
    if validation_report["issues"]:
        print(f"[ERROR] Critical Issues: {validation_report['issues']}")
    if validation_report["warnings"]:
        print(f"[WARNING] Warnings: {validation_report['warnings']}")
    print(f"[INFO] Split Stats: {validation_report['stats']}")

    # Create masks for splits
    train_mask = df["season"].isin(train_seasons)
    val_mask = df["season"].isin(val_seasons)
    test_mask = df["season"].isin(test_seasons)
    X_cols = select_feature_columns(df, include_momentum=include_momentum)

    # Prepare data splits
    X_train = df.loc[train_mask, X_cols].copy()
    X_val = df.loc[val_mask, X_cols].copy()
    X_test = df.loc[test_mask, X_cols].copy()

    # Handle NaN values across all splits
    print(f"\n>> Handling missing values...")
    print(f"NaN values - Train: {X_train.isnull().sum().sum()}, Val: {X_val.isnull().sum().sum()}, Test: {X_test.isnull().sum().sum()}")

    X_train = X_train.fillna(0)
    X_val = X_val.fillna(0)
    X_test = X_test.fillna(0)

    # Prepare target variables
    if task == "classification":
        y_train = df.loc[train_mask, "win"].values
        y_val = df.loc[val_mask, "win"].values
        y_test = df.loc[test_mask, "win"].values
    else:
        y_train = df.loc[train_mask, "point_diff"].values
        y_val = df.loc[val_mask, "point_diff"].values
        y_test = df.loc[test_mask, "point_diff"].values

    # Fit scaler on training data only
    print(f"\n>> Fitting scaler and optimizing PCA components...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Optimize PCA components using validation set
    best_n_components = None
    best_val_score = -np.inf
    pca_scores = []

    # Test different numbers of components
    max_components = min(X_train_scaled.shape[1], X_train_scaled.shape[0] - 1)
    component_range = range(5, min(max_components + 1, 21), 2)  # Test 5, 7, 9, ..., 20 components

    for n_comp in component_range:
        # Fit PCA
        pca_temp = PCA(n_components=n_comp, random_state=42)
        X_train_pca_temp = pca_temp.fit_transform(X_train_scaled)
        X_val_pca_temp = pca_temp.transform(X_val_scaled)

        # Fit model and evaluate on validation
        if task == "classification":
            model_temp = LogisticRegression(max_iter=200, random_state=42)
            model_temp.fit(X_train_pca_temp, y_train)
            val_score = roc_auc_score(y_val, model_temp.predict_proba(X_val_pca_temp)[:, 1])
        else:
            model_temp = LinearRegression()
            model_temp.fit(X_train_pca_temp, y_train)
            val_score = -mean_absolute_error(y_val, model_temp.predict(X_val_pca_temp))

        pca_scores.append((n_comp, val_score, pca_temp.explained_variance_ratio_.sum()))

        if val_score > best_val_score:
            best_val_score = val_score
            best_n_components = n_comp

    print(f"PCA optimization results:")
    for n_comp, score, var_explained in pca_scores:
        print(f"  {n_comp} components: Val score={score:.4f}, Variance explained={var_explained:.3f}")
    print(f"Best: {best_n_components} components (Val score: {best_val_score:.4f})")

    # Fit final PCA with optimal components
    pca = PCA(n_components=best_n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # Train final model
    if task == "classification":
        model = LogisticRegression(max_iter=200, random_state=42)
    else:
        model = LinearRegression()

    model.fit(X_train_pca, y_train)

    # Comprehensive performance evaluation
    performance_metrics = {
        "pca_optimization": pca_scores,
        "best_n_components": best_n_components,
        "explained_variance": pca.explained_variance_ratio_.sum(),
        "validation_report": validation_report
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
        performance_metrics["val_test_gap_acc"] = performance_metrics["validation_accuracy"] - performance_metrics["test_accuracy"]
        performance_metrics["val_test_gap_auc"] = performance_metrics["validation_roc_auc"] - performance_metrics["test_roc_auc"]
    else:
        performance_metrics["train_val_gap_mae"] = performance_metrics["validation_mae"] - performance_metrics["train_mae"]
        performance_metrics["val_test_gap_mae"] = performance_metrics["test_mae"] - performance_metrics["validation_mae"]

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
# Phase 3: Advanced Models & Optimization
###############################################################################

def optimize_advanced_hyperparameters(
    df: pd.DataFrame,
    task: str = "classification",
    n_splits: int = 3
) -> Dict:
    """
    Advanced hyperparameter optimization for multiple model types.
    Focus on models that can break through 80% accuracy ceiling.
    """
    print(f">> Starting Phase 3 Advanced Hyperparameter Optimization...")

    # Expanded parameter grids for advanced models
    param_grids = {
        'random_forest': {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'max_features': ['sqrt', 'log2', None],
            'pca_components': [15, 20, 25]
        },
        'xgboost': {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 6, 10],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'pca_components': [15, 20, 25]
        },
        'neural_network': {
            'hidden_layer_sizes': [(100,), (100, 50), (200, 100)],
            'learning_rate_init': [0.001, 0.01, 0.1],
            'alpha': [0.0001, 0.001, 0.01],
            'pca_components': [15, 20, 25]
        }
    }

    best_results = {}
    cv_splits = time_series_cross_validation(df, n_splits=n_splits)

    for model_name, param_grid in param_grids.items():
        print(f"\n>> Optimizing {model_name.upper()}...")

        best_score = -np.inf
        best_params = None
        model_cv_results = []

        # Limit combinations for feasible runtime (sample from grid)
        import itertools
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        all_combinations = list(itertools.product(*param_values))

        # Sample top combinations (limit to 20 per model for runtime)
        import random
        random.seed(42)
        selected_combinations = random.sample(all_combinations, min(20, len(all_combinations)))

        print(f"Testing {len(selected_combinations)} parameter combinations for {model_name}...")

        for i, param_combo in enumerate(selected_combinations):
            current_params = dict(zip(param_names, param_combo))
            cv_scores = []

            for fold_idx, (train_mask, val_mask, train_seasons, val_seasons) in enumerate(cv_splits):
                try:
                    # Prepare data for this fold (NO CLUSTERING)
                    X_cols = select_feature_columns(df, include_momentum=True)

                    X_train = df.loc[train_mask, X_cols].fillna(0)
                    X_val = df.loc[val_mask, X_cols].fillna(0)

                    if task == "classification":
                        y_train = df.loc[train_mask, "win"].values
                        y_val = df.loc[val_mask, "win"].values
                    else:
                        y_train = df.loc[train_mask, "point_diff"].values
                        y_val = df.loc[val_mask, "point_diff"].values

                    # PCA preprocessing
                    scaler = StandardScaler()
                    pca = PCA(n_components=current_params['pca_components'], random_state=42)

                    X_train_scaled = scaler.fit_transform(X_train)
                    X_val_scaled = scaler.transform(X_val)
                    X_train_pca = pca.fit_transform(X_train_scaled)
                    X_val_pca = pca.transform(X_val_scaled)

                    # Model selection and fitting
                    if model_name == 'random_forest':
                        model = RandomForestClassifier(
                            n_estimators=current_params['n_estimators'],
                            max_depth=current_params['max_depth'],
                            min_samples_split=current_params['min_samples_split'],
                            max_features=current_params['max_features'],
                            random_state=42,
                            n_jobs=-1
                        )
                    elif model_name == 'xgboost':
                        model = xgb.XGBClassifier(
                            n_estimators=current_params['n_estimators'],
                            max_depth=current_params['max_depth'],
                            learning_rate=current_params['learning_rate'],
                            subsample=current_params['subsample'],
                            colsample_bytree=current_params['colsample_bytree'],
                            random_state=42,
                            eval_metric='logloss'
                        )
                    elif model_name == 'neural_network':
                        model = MLPClassifier(
                            hidden_layer_sizes=current_params['hidden_layer_sizes'],
                            learning_rate_init=current_params['learning_rate_init'],
                            alpha=current_params['alpha'],
                            random_state=42,
                            max_iter=1000
                        )

                    model.fit(X_train_pca, y_train)

                    # Evaluate
                    if task == "classification":
                        if hasattr(model, 'predict_proba'):
                            val_probs = model.predict_proba(X_val_pca)[:, 1]
                            fold_score = roc_auc_score(y_val, val_probs)
                        else:
                            val_preds = model.predict(X_val_pca)
                            fold_score = accuracy_score(y_val, val_preds)
                    else:
                        val_preds = model.predict(X_val_pca)
                        fold_score = -mean_absolute_error(y_val, val_preds)

                    cv_scores.append(fold_score)

                except Exception as e:
                    print(f"    Error in fold {fold_idx}: {e}")
                    cv_scores.append(-1.0)

            avg_cv_score = np.mean(cv_scores) if cv_scores else -1.0
            model_cv_results.append({
                'params': current_params.copy(),
                'cv_score': avg_cv_score,
                'cv_scores': cv_scores
            })

            if avg_cv_score > best_score:
                best_score = avg_cv_score
                best_params = current_params.copy()

            if (i + 1) % 5 == 0:
                print(f"    Progress: {i + 1}/{len(selected_combinations)} combinations")

        best_results[model_name] = {
            'best_params': best_params,
            'best_score': best_score,
            'cv_results': model_cv_results
        }

        print(f"Best {model_name} CV score: {best_score:.4f}")
        print(f"Best {model_name} params: {best_params}")

    return best_results


def fit_phase3_pipeline(
    df: pd.DataFrame,
    enable_hyperopt: bool = True,
    task: str = "classification",
    train_seasons: List[int] = None,
    val_seasons: List[int] = None,
    test_seasons: List[int] = None
) -> Tuple[Pipeline, dict, dict]:
    """
    Phase 3: Advanced Models Pipeline (NO CLUSTERING - Focus on Model Quality)

    Key improvements:
    - Remove clustering features (reduced complexity)
    - Use advanced models: Random Forest, XGBoost, Neural Network
    - Full hyperparameter optimization
    - Ensemble methods
    """
    # Default temporal splits
    available_seasons = sorted(df["season"].unique())
    if train_seasons is None:
        train_seasons = available_seasons[:6]  # 2015-2020
    if val_seasons is None:
        val_seasons = available_seasons[6:8]   # 2021-2022
    if test_seasons is None:
        test_seasons = available_seasons[8:]   # 2023-2024

    print(f"Phase 3 Advanced Pipeline Configuration:")
    print(f"Training seasons: {train_seasons}")
    print(f"Validation seasons: {val_seasons}")
    print(f"Test seasons: {test_seasons}")

    # Phase 3.1 - Advanced Model Selection & Hyperparameter Optimization
    optimization_results = {}
    best_model_config = {
        'model_type': 'random_forest',  # Default fallback
        'params': {'n_estimators': 200, 'max_depth': 15, 'pca_components': 20}
    }

    if enable_hyperopt:
        print(f"\n>> Phase 3.1: Advanced Model Hyperparameter Optimization...")
        optimization_results = optimize_advanced_hyperparameters(df, task=task)

        # Select best overall model
        best_overall_score = -np.inf
        for model_name, results in optimization_results.items():
            if results['best_score'] > best_overall_score:
                best_overall_score = results['best_score']
                best_model_config = {
                    'model_type': model_name,
                    'params': results['best_params']
                }

        print(f"\nBest overall model: {best_model_config['model_type']} (Score: {best_overall_score:.4f})")
    else:
        print(f"\n>> Phase 3.1: Using default Random Forest configuration...")

    # Phase 3.2 - Train Final Advanced Model
    print(f"\n>> Phase 3.2: Training Final Advanced Model...")
    print(f"Selected model: {best_model_config['model_type']}")
    print(f"Parameters: {best_model_config['params']}")

    # Prepare final data (NO CLUSTERING FEATURES)
    train_mask = df["season"].isin(train_seasons)
    val_mask = df["season"].isin(val_seasons)
    test_mask = df["season"].isin(test_seasons)

    # Use core features only (no clustering)
    X_cols = select_feature_columns(df, include_momentum=True)
    print(f"Using {len(X_cols)} features (NO clustering features)")

    X_train = df.loc[train_mask, X_cols].fillna(0)
    X_val = df.loc[val_mask, X_cols].fillna(0)
    X_test = df.loc[test_mask, X_cols].fillna(0)

    if task == "classification":
        y_train = df.loc[train_mask, "win"].values
        y_val = df.loc[val_mask, "win"].values
        y_test = df.loc[test_mask, "win"].values
    else:
        y_train = df.loc[train_mask, "point_diff"].values
        y_val = df.loc[val_mask, "point_diff"].values
        y_test = df.loc[test_mask, "point_diff"].values

    # PCA preprocessing
    scaler = StandardScaler()
    pca_components = best_model_config['params'].get('pca_components', 20)
    pca = PCA(n_components=pca_components, random_state=42)

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # Build final model based on best configuration
    model_type = best_model_config['model_type']
    params = best_model_config['params']

    if model_type == 'random_forest':
        final_model = RandomForestClassifier(
            n_estimators=params.get('n_estimators', 200),
            max_depth=params.get('max_depth', 15),
            min_samples_split=params.get('min_samples_split', 5),
            max_features=params.get('max_features', 'sqrt'),
            random_state=42,
            n_jobs=-1
        )
    elif model_type == 'xgboost':
        final_model = xgb.XGBClassifier(
            n_estimators=params.get('n_estimators', 200),
            max_depth=params.get('max_depth', 6),
            learning_rate=params.get('learning_rate', 0.1),
            subsample=params.get('subsample', 0.9),
            colsample_bytree=params.get('colsample_bytree', 0.9),
            random_state=42,
            eval_metric='logloss'
        )
    elif model_type == 'neural_network':
        final_model = MLPClassifier(
            hidden_layer_sizes=params.get('hidden_layer_sizes', (100, 50)),
            learning_rate_init=params.get('learning_rate_init', 0.01),
            alpha=params.get('alpha', 0.001),
            random_state=42,
            max_iter=1000
        )
    else:  # Default to Logistic Regression
        final_model = LogisticRegression(C=1.0, max_iter=200, random_state=42)

    # Train final model
    final_model.fit(X_train_pca, y_train)

    # Comprehensive evaluation
    performance_metrics = {
        'model_type': model_type,
        'best_model_params': params,
        'pca_components': pca_components,
        'features_used': len(X_cols),
        'explained_variance': pca.explained_variance_ratio_.sum(),
        'clustering_removed': True,
        'hyperopt_enabled': enable_hyperopt,
        'optimization_results': optimization_results
    }

    # Evaluate on all splits
    for split_name, X_split, y_split in [
        ("train", X_train_pca, y_train),
        ("validation", X_val_pca, y_val),
        ("test", X_test_pca, y_test)
    ]:
        if task == "classification":
            preds = final_model.predict(X_split)
            if hasattr(final_model, 'predict_proba'):
                probs = final_model.predict_proba(X_split)[:, 1]
                performance_metrics[f"{split_name}_roc_auc"] = roc_auc_score(y_split, probs)
            performance_metrics[f"{split_name}_accuracy"] = accuracy_score(y_split, preds)
        else:
            preds = final_model.predict(X_split)
            performance_metrics[f"{split_name}_mae"] = mean_absolute_error(y_split, preds)

    # Calculate generalization gaps
    if task == "classification":
        performance_metrics["train_val_gap_acc"] = performance_metrics["train_accuracy"] - performance_metrics["validation_accuracy"]
        if "train_roc_auc" in performance_metrics and "validation_roc_auc" in performance_metrics:
            performance_metrics["train_val_gap_auc"] = performance_metrics["train_roc_auc"] - performance_metrics["validation_roc_auc"]
        if "validation_accuracy" in performance_metrics and "test_accuracy" in performance_metrics:
            performance_metrics["val_test_gap_acc"] = performance_metrics["validation_accuracy"] - performance_metrics["test_accuracy"]
        if "validation_roc_auc" in performance_metrics and "test_roc_auc" in performance_metrics:
            performance_metrics["val_test_gap_auc"] = performance_metrics["validation_roc_auc"] - performance_metrics["test_roc_auc"]

    # Data splits
    data_splits = {
        'X_train_pca': X_train_pca,
        'X_val_pca': X_val_pca,
        'X_test_pca': X_test_pca,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'feature_names': X_cols,
        'train_seasons': train_seasons,
        'val_seasons': val_seasons,
        'test_seasons': test_seasons
    }

    # Final pipeline
    pipeline = Pipeline([
        ("scaler", scaler),
        ("pca", pca),
        ("model", final_model)
    ])

    return pipeline, data_splits, performance_metrics


###############################################################################
# Entry point
###############################################################################

if __name__ == "__main__":
    # --- Config ---
    YEARS = list(range(2015, 2025))  # Full dataset: 2015-2024 (10 seasons)
    INCLUDE_MOMENTUM = True          # flip to False to compare raw vs raw+momentum
    MOMENTUM_METRIC = "epa_per_play" # base metric for momentum
    EMA_SPAN = 3
    TASK = "classification"          # "classification" (win) or "regression" (point_diff)
    VAR_TARGET = 0.90                # cumulative explained variance target

    try:
        print(">> Assembling dataset ...")
        print(f"Years: {YEARS}")
        print(f"Task: {TASK}")
        print(f"Include momentum: {INCLUDE_MOMENTUM}")

        data = assemble_team_game_dataset(
            years=YEARS,
            include_momentum=INCLUDE_MOMENTUM,
            momentum_metric=MOMENTUM_METRIC,
            ema_span=EMA_SPAN,
        )

        print("Rows:", len(data), "Cols:", len(data.columns))
        print("Sample cols:", data.columns[:12].tolist())

        # Validate dataset
        if len(data) == 0:
            raise ValueError("Dataset is empty - check data availability for specified years")

        # Phase 1.2 - Momentum Optimization
        print(">> Phase 1.2: Testing Momentum Configurations...")
        momentum_results, best_momentum = test_momentum_configurations(data)

        # Use best momentum configuration or fall back to default
        if best_momentum:
            MOMENTUM_METRIC = best_momentum[1]["metric"]
            EMA_SPAN = best_momentum[1]["ema_span"]
            print(f"Using optimal momentum: {MOMENTUM_METRIC} with {EMA_SPAN}-game span")

            # Rebuild dataset with optimal momentum
            print(">> Rebuilding dataset with optimal momentum configuration...")
            data = assemble_team_game_dataset(
                years=YEARS,
                include_momentum=INCLUDE_MOMENTUM,
                momentum_metric=MOMENTUM_METRIC,
                ema_span=EMA_SPAN,
            )

        print(">> Fitting Phase 3 Advanced Models Pipeline (NO CLUSTERING)...")
        pipeline, data_splits, performance_metrics = fit_phase3_pipeline(
            data,
            enable_hyperopt=True,  # Full hyperparameter optimization enabled
            task=TASK
        )

        # Add momentum optimization results to performance metrics
        performance_metrics["momentum_optimization"] = momentum_results
        performance_metrics["best_momentum_config"] = best_momentum[1] if best_momentum else None

        print("\n" + "="*80)
        print("PHASE 2: ADVANCED MODELING RESULTS")
        print("="*80)

        # Print comprehensive performance metrics
        print(f"\nPERFORMANCE ACROSS ALL SPLITS:")
        if TASK == "classification":
            print(f"Training:   Accuracy={performance_metrics['train_accuracy']:.4f}, ROC-AUC={performance_metrics['train_roc_auc']:.4f}")
            print(f"Validation: Accuracy={performance_metrics['validation_accuracy']:.4f}, ROC-AUC={performance_metrics['validation_roc_auc']:.4f}")
            print(f"Test:       Accuracy={performance_metrics['test_accuracy']:.4f}, ROC-AUC={performance_metrics['test_roc_auc']:.4f}")
            print(f"\nGENERALIZATION GAPS:")
            print(f"Train->Val:  Acc={performance_metrics['train_val_gap_acc']:.4f}, AUC={performance_metrics['train_val_gap_auc']:.4f}")
            print(f"Val->Test:   Acc={performance_metrics['val_test_gap_acc']:.4f}, AUC={performance_metrics['val_test_gap_auc']:.4f}")
        else:
            print(f"Training:   MAE={performance_metrics['train_mae']:.4f}")
            print(f"Validation: MAE={performance_metrics['validation_mae']:.4f}")
            print(f"Test:       MAE={performance_metrics['test_mae']:.4f}")

        print(f"\nADVANCED MODEL CONFIGURATION:")
        print(f"Features used: {len(data_splits['feature_names'])}")
        print(f"PCA components: {performance_metrics.get('best_params', {}).get('pca_components', 'N/A')}")
        print(f"Explained variance: {performance_metrics['explained_variance']:.3f}")
        print(f"Clustering enabled: {performance_metrics.get('clustering_enabled', False)}")
        print(f"Hyperopt enabled: {performance_metrics.get('hyperopt_enabled', False)}")
        print(f"Training seasons: {data_splits['train_seasons']}")
        print(f"Validation seasons: {data_splits['val_seasons']}")
        print(f"Test seasons: {data_splits['test_seasons']}")

        # Display clustering results
        if performance_metrics.get('clustering_enabled'):
            team_results = performance_metrics.get('team_style_results', {})
            context_results = performance_metrics.get('context_results', {})

            print(f"\nCLUSTERING RESULTS:")
            if team_results:
                print(f"Team style clusters: {len(team_results.get('interpretations', {}))}")
                print(f"Team clustering silhouette: {team_results.get('silhouette_score', 'N/A'):.3f}")
            if context_results:
                print(f"Game context clusters: {len(context_results.get('interpretations', {}))}")
                print(f"Context clustering silhouette: {context_results.get('silhouette_score', 'N/A'):.3f}")

        # Display hyperparameter optimization results
        if performance_metrics.get('hyperopt_enabled') and performance_metrics.get('hyperopt_results'):
            hyperopt = performance_metrics['hyperopt_results']
            print(f"\nHYPERPARAMETER OPTIMIZATION:")
            print(f"Best CV score: {hyperopt.get('best_score', 'N/A'):.4f}")
            print(f"Combinations tested: {hyperopt.get('total_combinations', 'N/A')}")
            best_params = hyperopt.get('best_params', {})
            print(f"Best parameters: PCA={best_params.get('pca_components')}, C={best_params.get('model_C')}")

        # Display momentum optimization results
        if performance_metrics.get("best_momentum_config"):
            best_config = performance_metrics["best_momentum_config"]
            print(f"\nMOMENTUM OPTIMIZATION:")
            print(f"Best configuration: {best_config['metric']} with {best_config['ema_span']}-game span")
            print(f"Momentum AUC: {best_config['test_auc']:.4f}")

        # Display new features summary
        strength_features = [col for col in data_splits['feature_names'] if col.startswith('opp_')]
        momentum_features = [col for col in data_splits['feature_names'] if 'momentum' in col]
        cluster_features = [col for col in data_splits['feature_names'] if 'cluster' in col]

        print(f"\nADVANCED FEATURES:")
        print(f"Strength of Schedule: {len(strength_features)} features")
        print(f"Momentum: {len(momentum_features)} features")
        print(f"Clustering: {len(cluster_features)} features")
        print(f"Total features: {len(data_splits['feature_names'])}")

        # Calculate improvement progression
        phase1_accuracy = 0.796   # Phase 1.1 baseline
        phase12_accuracy = 0.803  # Phase 1.2 with SOS + momentum
        current_accuracy = performance_metrics['test_accuracy']

        total_improvement = (current_accuracy - phase1_accuracy) * 100
        phase2_improvement = (current_accuracy - phase12_accuracy) * 100

        print(f"\nPERFORMANCE PROGRESSION:")
        print(f"Phase 1.1 baseline: {phase1_accuracy:.1%}")
        print(f"Phase 1.2 enhanced: {phase12_accuracy:.1%} (+{(phase12_accuracy-phase1_accuracy)*100:.1f}pp)")
        print(f"Phase 2.0 advanced: {current_accuracy:.1%} (+{phase2_improvement:.1f}pp from Phase 1.2)")
        print(f"Total improvement: +{total_improvement:.1f} percentage points")

        # Progress toward 85-90% target
        target_min = 0.85
        target_max = 0.90
        progress_min = (current_accuracy - phase1_accuracy) / (target_min - phase1_accuracy) * 100
        progress_max = (current_accuracy - phase1_accuracy) / (target_max - phase1_accuracy) * 100

        print(f"\nTARGET PROGRESS:")
        print(f"Progress toward 85% target: {min(100, progress_min):.1f}% complete")
        print(f"Progress toward 90% target: {min(100, progress_max):.1f}% complete")

        # Save Phase 2 artifacts
        os.makedirs("artifacts", exist_ok=True)
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        clustering_suffix = "_clustered" if performance_metrics.get('clustering_enabled') else ""
        hyperopt_suffix = "_hyperopt" if performance_metrics.get('hyperopt_enabled') else ""

        pipeline_filename = f"artifacts/phase2_pipeline_{TASK}{clustering_suffix}{hyperopt_suffix}_{timestamp}.joblib"
        features_filename = f"artifacts/phase2_features{clustering_suffix}{hyperopt_suffix}_{timestamp}.csv"
        components_filename = f"artifacts/phase2_components{clustering_suffix}{hyperopt_suffix}_{timestamp}.csv"
        metrics_filename = f"artifacts/phase2_metrics{clustering_suffix}{hyperopt_suffix}_{timestamp}.json"

        # Save pipeline
        dump(pipeline, pipeline_filename)

        # Save features
        pd.Series(data_splits['feature_names']).to_csv(features_filename, index=False)

        # Save PCA components
        pca = pipeline.named_steps["pca"]
        comp_df = pd.DataFrame(pca.components_, columns=data_splits['feature_names'])
        comp_df.to_csv(components_filename, index=False)

        # Save comprehensive metrics (convert numpy types to native Python, exclude non-serializable objects)
        metrics_to_save = {}
        excluded_keys = {'team_style_results', 'context_results', 'clusterer', 'hyperopt_results'}

        for key, value in performance_metrics.items():
            if key in excluded_keys:
                continue  # Skip non-serializable objects
            elif key == "pca_optimization":
                metrics_to_save[key] = [(int(n), float(s), float(v)) for n, s, v in value] if value else []
            elif key == "validation_report":
                metrics_to_save[key] = value  # dict is already serializable
            elif key == "best_params" and isinstance(value, dict):
                # Convert numpy types in nested dict
                metrics_to_save[key] = {k: v.item() if isinstance(v, (np.integer, np.floating)) else v for k, v in value.items()}
            elif isinstance(value, (np.integer, np.floating)):
                metrics_to_save[key] = value.item()
            elif isinstance(value, (dict, list, str, bool, type(None))):
                metrics_to_save[key] = value
            elif hasattr(value, 'item'):  # numpy scalar
                metrics_to_save[key] = value.item()
            else:
                metrics_to_save[key] = str(value)  # Convert to string as fallback

        import json
        with open(metrics_filename, 'w') as f:
            json.dump(metrics_to_save, f, indent=2)

        print(f"\nARTIFACTS SAVED:")
        print(f"Pipeline: {pipeline_filename}")
        print(f"Features: {features_filename}")
        print(f"Components: {components_filename}")
        print(f"Metrics: {metrics_filename}")

        print(f"\nEnhanced Pipeline completed successfully!")
        print(f"Test Accuracy Achieved: {performance_metrics['test_accuracy']:.1%}")
        print("="*80)

    except Exception as e:
        print(f"Error during pipeline execution: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        exit(1)

