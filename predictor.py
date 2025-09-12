import os
import numpy as np
import pandas as pd
import nfl_data_py as nfl
from typing import List, Tuple, Optional

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from joblib import dump

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
    season = load_and_prepare_pbp(years)
    off   = offensive_stats(season)   # posteam-based stats
    defe  = defensive_stats(season)   # defteam-based "allowed" stats

    # 2) Import schedules
    sched = nfl.import_schedules(years=years)[
        ["game_id","home_team","away_team","home_score","away_score","season","week"]
    ].dropna(subset=["game_id"])

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
    assert not off.duplicated(subset=keys).any(), "Offense table has duplicate merge keys"
    assert not defe_ren.duplicated(subset=keys).any(), "Defense table has duplicate merge keys"

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

    # 8) Momentum (optional)
    if include_momentum:
        df = df.sort_values(["team","season","week"]).reset_index(drop=True)
        df = add_momentum_simple(df, metric=momentum_metric, ema_span=ema_span)
        df["momentum_score"] = df["momentum_score"].fillna(0.0).round(4)

    # 9) Clean column order
    base_keys = ["season","week","game_id","team","opponent","win","point_diff"]
    cols = base_keys + [c for c in df.columns if c not in base_keys]
    return df[cols]



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
# PCA pipeline
###############################################################################

def fit_pca_pipeline(
    df: pd.DataFrame,
    include_momentum: bool,
    explained_variance_target: float = 0.9,
    task: str = "classification"  # "classification" -> win; "regression" -> point_diff
) -> Tuple[Pipeline, np.ndarray, np.ndarray, List[str], dict]:
    """
    Builds a train/test split by season, fits StandardScaler+PCA on train only,
    auto-selects n_components to reach `explained_variance_target`, and trains a
    simple downstream model (logistic or linear) for a first-pass evaluation.

    Returns:
      - pipeline (scaler+PCA)
      - X_train_pca, X_test_pca
      - feature_names used
      - metrics dict with quick scores
    """
    # Train/test by season (example: train 2015-2022, test 2023-2024 if present)
    train_mask = df["season"] <= df["season"].min() + 8  # ~first 9 seasons
    X_cols = select_feature_columns(df, include_momentum=include_momentum)

    X_train = df.loc[train_mask, X_cols].copy()
    X_test  = df.loc[~train_mask, X_cols].copy()

    if task == "classification":
        y_train = df.loc[train_mask, "win"].values
        y_test  = df.loc[~train_mask, "win"].values
    else:
        y_train = df.loc[train_mask, "point_diff"].values
        y_test  = df.loc[~train_mask, "point_diff"].values

    # Scale fit on train only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # PCA fit on train only
    pca_full = PCA(n_components=None, random_state=42)
    pca_full.fit(X_train_scaled)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumvar, explained_variance_target) + 1)

    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca  = pca.transform(X_test_scaled)

    # Downstream baseline model
    if task == "classification":
        model = LogisticRegression(max_iter=200, n_jobs=None)
        model.fit(X_train_pca, y_train)
        preds = model.predict(X_test_pca)
        probs = model.predict_proba(X_test_pca)[:, 1]
        metrics = {
            "n_components": n_components,
            "explained_var_90%_idx": n_components,
            "test_acc": float(accuracy_score(y_test, preds)),
            "test_roc_auc": float(roc_auc_score(y_test, probs)),
        }
    else:
        model = LinearRegression()
        model.fit(X_train_pca, y_train)
        preds = model.predict(X_test_pca)
        metrics = {
            "n_components": n_components,
            "explained_var_90%_idx": n_components,
            "test_mae": float(mean_absolute_error(y_test, preds)),
        }

    # Build pipeline object (scaler + PCA) for saving
    pipe = Pipeline([("scaler", scaler), ("pca", pca), ("model", model)])

    return pipe, X_train_pca, X_test_pca, X_cols, metrics


###############################################################################
# Entry point
###############################################################################

if __name__ == "__main__":
    # --- Config ---
    YEARS = list(range(2015, 2022))  # adjust if you want a different span
    INCLUDE_MOMENTUM = True          # flip to False to compare raw vs raw+momentum
    MOMENTUM_METRIC = "epa_per_play" # base metric for momentum
    EMA_SPAN = 3
    TASK = "classification"          # "classification" (win) or "regression" (point_diff)
    VAR_TARGET = 0.90                # cumulative explained variance target

    print("▶ Assembling dataset ...")
    data = assemble_team_game_dataset(
        years=YEARS,
        include_momentum=INCLUDE_MOMENTUM,
        momentum_metric=MOMENTUM_METRIC,
        ema_span=EMA_SPAN,
    )

    print("Rows:", len(data), "Cols:", len(data.columns))
    print("Sample cols:", data.columns[:12].tolist())

    print("▶ Fitting PCA pipeline ...")
    pipeline, Xtr_pca, Xte_pca, feat_names, metrics = fit_pca_pipeline(
        data,
        include_momentum=INCLUDE_MOMENTUM,
        explained_variance_target=VAR_TARGET,
        task=TASK,
    )

    print("Metrics:", metrics)

    # Save artifacts (optional)
    os.makedirs("artifacts", exist_ok=True)
    dump(pipeline, f"artifacts/pca_pipeline_{TASK}_{'withmom' if INCLUDE_MOMENTUM else 'raw'}.joblib")
    pd.Series(feat_names).to_csv(f"artifacts/feature_names_{'withmom' if INCLUDE_MOMENTUM else 'raw'}.csv", index=False)

    # Quick interpretability: top component loadings (optional print)
    pca = pipeline.named_steps["pca"]
    comp_df = pd.DataFrame(pca.components_, columns=feat_names)
    comp_df.to_csv(f"artifacts/pca_components_{'withmom' if INCLUDE_MOMENTUM else 'raw'}.csv", index=False)
    print(f"Saved pipeline and PCA loadings to ./artifacts/")

