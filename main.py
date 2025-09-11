import pandas as pd 
import numpy as np
import nfl_data_py as nfl

def load_and_prepare_pbp(years):
    """
    Downloads PBP once and returns a filtered/enriched per-play DataFrame
    ready for any team-level aggregations (offense or defense).
    """
    season = nfl.import_pbp_data(years, downcast=True)
    season = season[season["play_type"].isin(["run", "pass"]) & season["epa"].notna()].copy()

    # Basic derived columns
    season["turnover"]   = ((season.get("interception", 0).fillna(0) > 0) | (season.get("fumble_lost", 0).fillna(0) > 0)).astype(int)
    season["pass_yards"] = np.where(season["play_type"] == "pass", season["yards_gained"], 0)
    season["rush_yards"] = np.where(season["play_type"] == "run",  season["yards_gained"], 0)

    season["success"]   = (season["epa"] > 0).astype(int)
    season["explosive"] = (
        ((season["play_type"] == "pass") & (season["yards_gained"] >= 20)) |
        ((season["play_type"] == "run")  & (season["yards_gained"] >= 10))
    ).astype(int)

    # Situational flags
    season["is_third_down"]   = (season["down"] == 3).astype(int)
    season["third_down_conv"] = season.get("third_down_converted", 0).fillna(0).astype(int)

    # Red zone (inside opp 20)
    season["in_red_zone"] = (season.get("yardline_100", np.nan) <= 20).fillna(False).astype(int)
    season["redzone_td"]  = (season["in_red_zone"].eq(1) & season.get("touchdown", 0).fillna(0).astype(bool)).astype(int)

    # Pass / rush markers
    season["is_pass"] = season.get("pass", 0).fillna(0).astype(int)
    season["is_rush"] = season.get("rush", 0).fillna(0).astype(int)

    # EPA splits
    season["pass_epa"] = np.where(season["is_pass"] == 1, season["epa"], 0.0)
    season["rush_epa"] = np.where(season["is_rush"] == 1, season["epa"], 0.0)

    # Concepts / formations (guard optional cols)
    if "play_action" in season.columns:
        season["play_action_pass"] = ((season["play_action"].fillna(0) == 1) & (season["is_pass"] == 1)).astype(int)
    else:
        season["play_action_pass"] = 0
    season["shotgun_play"] = season.get("shotgun", 0).fillna(0).astype(int)

    # Dropbacks & sacks (prefer qb_dropback; else approximate with pass)
    if "qb_dropback" in season.columns:
        season["dropback"] = season["qb_dropback"].fillna(0).astype(int)
    else:
        season["dropback"] = season["is_pass"]
    season["sack"] = season.get("sack", 0).fillna(0).astype(int)

    return season

def offensive_stats(season):
    """
    Per-team, per-game OFFENSIVE metrics from prepared PBP.
    Returns offensive_stats dataframe.
    """
    offensive_stats = (
        season.groupby(["season", "week", "game_id", "posteam"], as_index=False)
              .agg(
                  plays=("epa", "size"),
                  pass_plays=("is_pass", "sum"),
                  rush_plays=("is_rush", "sum"),
                  pass_yards=("pass_yards", "sum"),
                  rush_yards=("rush_yards", "sum"),
                  turnovers=("turnover", "sum"),
                  epa_total=("epa", "sum"),

                  # Advanced counts
                  success_plays=("success", "sum"),
                  explosive_plays=("explosive", "sum"),
                  third_down_atts=("is_third_down", "sum"),
                  third_down_convs=("third_down_conv", "sum"),
                  redzone_trips=("in_red_zone", "sum"),
                  redzone_tds=("redzone_td", "sum"),
                  play_action_passes=("play_action_pass", "sum"),
                  shotgun_plays=("shotgun_play", "sum"),
                  dropbacks=("dropback", "sum"),
                  sacks=("sack", "sum"),

                  # EPA splits
                  pass_epa_sum=("pass_epa", "sum"),
                  rush_epa_sum=("rush_epa", "sum"),
              )
    )

    # Efficiencies & rates
    offensive_stats["epa_per_play"]       = offensive_stats["epa_total"] / offensive_stats["plays"].replace(0, np.nan)
    offensive_stats["yards_per_play"]     = (offensive_stats["pass_yards"] + offensive_stats["rush_yards"]) / offensive_stats["plays"].replace(0, np.nan)
    offensive_stats["pass_epa_per_play"]  = offensive_stats["pass_epa_sum"] / offensive_stats["pass_plays"].replace(0, np.nan)
    offensive_stats["rush_epa_per_play"]  = offensive_stats["rush_epa_sum"] / offensive_stats["rush_plays"].replace(0, np.nan)

    offensive_stats["success_rate"]         = offensive_stats["success_plays"]   / offensive_stats["plays"].replace(0, np.nan)
    offensive_stats["explosive_rate"]       = offensive_stats["explosive_plays"] / offensive_stats["plays"].replace(0, np.nan)
    offensive_stats["third_down_conv_rate"] = offensive_stats["third_down_convs"] / offensive_stats["third_down_atts"].replace(0, np.nan)
    offensive_stats["red_zone_td_rate"]     = offensive_stats["redzone_tds"]      / offensive_stats["redzone_trips"].replace(0, np.nan)
    offensive_stats["pass_rate"]            = offensive_stats["pass_plays"]       / offensive_stats["plays"].replace(0, np.nan)
    offensive_stats["play_action_rate"]     = offensive_stats["play_action_passes"] / offensive_stats["pass_plays"].replace(0, np.nan)
    offensive_stats["shotgun_rate"]         = offensive_stats["shotgun_plays"]    / offensive_stats["plays"].replace(0, np.nan)
    offensive_stats["sack_rate"]            = offensive_stats["sacks"]            / offensive_stats["dropbacks"].replace(0, np.nan)

    offensive_stats = offensive_stats.rename(columns={"posteam": "team"})

    keep_cols = [
        "season", "week", "game_id", "team",
        "plays", "pass_yards", "rush_yards", "turnovers", "epa_total", "epa_per_play",
        "yards_per_play", "pass_epa_per_play", "rush_epa_per_play",
        "success_rate", "explosive_rate", "third_down_conv_rate", "red_zone_td_rate",
        "pass_rate", "play_action_rate", "shotgun_rate", "sack_rate",
        "pass_plays", "rush_plays", "dropbacks", "sacks", "redzone_trips", "redzone_tds"
    ]

    return offensive_stats[keep_cols].sort_values(["season", "week", "game_id", "team"]).reset_index(drop=True)

def defensive_stats(season):
    """
    Per-team, per-game DEFENSIVE metrics (what the defense allowed).
    Same structure as offense, but grouped by defteam and prefixed with 'allowed_'.
    """
    df = (
        season.groupby(["season", "week", "game_id", "defteam"], as_index=False)
              .agg(
                  plays_allowed=("epa", "size"),
                  pass_plays_allowed=("is_pass", "sum"),
                  rush_plays_allowed=("is_rush", "sum"),
                  allowed_pass_yards=("pass_yards", "sum"),
                  allowed_rush_yards=("rush_yards", "sum"),
                  takeaways=("turnover", "sum"),  # turnovers by offense => takeaways by defense
                  allowed_epa_total=("epa", "sum"),

                  success_plays_allowed=("success", "sum"),
                  explosive_plays_allowed=("explosive", "sum"),
                  third_down_atts_allowed=("is_third_down", "sum"),
                  third_down_convs_allowed=("third_down_conv", "sum"),
                  redzone_trips_allowed=("in_red_zone", "sum"),
                  redzone_tds_allowed=("redzone_td", "sum"),
                  play_action_passes_allowed=("play_action_pass", "sum"),
                  shotgun_plays_allowed=("shotgun_play", "sum"),
                  dropbacks_allowed=("dropback", "sum"),
                  sacks_made=("sack", "sum"),  # sacks made by defense

                  pass_epa_sum_allowed=("pass_epa", "sum"),
                  rush_epa_sum_allowed=("rush_epa", "sum"),
              )
    )

    # Efficiencies & rates (allowed side)
    df["allowed_epa_per_play"]      = df["allowed_epa_total"] / df["plays_allowed"].replace(0, np.nan)
    df["allowed_yards_per_play"]    = (df["allowed_pass_yards"] + df["allowed_rush_yards"]) / df["plays_allowed"].replace(0, np.nan)
    df["allowed_pass_epa_per_play"] = df["pass_epa_sum_allowed"] / df["pass_plays_allowed"].replace(0, np.nan)
    df["allowed_rush_epa_per_play"] = df["rush_epa_sum_allowed"] / df["rush_plays_allowed"].replace(0, np.nan)

    df["allowed_success_rate"]         = df["success_plays_allowed"]   / df["plays_allowed"].replace(0, np.nan)
    df["allowed_explosive_rate"]       = df["explosive_plays_allowed"] / df["plays_allowed"].replace(0, np.nan)
    df["third_down_conv_rate_allowed"] = df["third_down_convs_allowed"] / df["third_down_atts_allowed"].replace(0, np.nan)
    df["red_zone_td_rate_allowed"]     = df["redzone_tds_allowed"]      / df["redzone_trips_allowed"].replace(0, np.nan)
    df["pass_rate_allowed"]            = df["pass_plays_allowed"]       / df["plays_allowed"].replace(0, np.nan)
    df["play_action_rate_allowed"]     = df["play_action_passes_allowed"] / df["pass_plays_allowed"].replace(0, np.nan)
    df["shotgun_rate_allowed"]         = df["shotgun_plays_allowed"]    / df["plays_allowed"].replace(0, np.nan)
    df["pressure_sack_rate"]           = df["sacks_made"]               / df["dropbacks_allowed"].replace(0, np.nan)

    df = df.rename(columns={"defteam": "team"})

    keep_cols = [
        "season", "week", "game_id", "team",
        "allowed_pass_yards", "allowed_rush_yards", "takeaways",
        "allowed_epa_total", "allowed_epa_per_play", "allowed_yards_per_play",
        "allowed_pass_epa_per_play", "allowed_rush_epa_per_play",
        "allowed_success_rate", "allowed_explosive_rate",
        "third_down_conv_rate_allowed", "red_zone_td_rate_allowed",
        "pass_rate_allowed", "play_action_rate_allowed", "shotgun_rate_allowed",
        "pressure_sack_rate",
        "plays_allowed", "pass_plays_allowed", "rush_plays_allowed",
        "dropbacks_allowed", "sacks_made", "redzone_trips_allowed", "redzone_tds_allowed"
    ]

    return df[keep_cols].sort_values(["season", "week", "game_id", "team"]).reset_index(drop=True)


def add_momentum_simple(df: pd.DataFrame, metric: str, ema_span: int = 5) -> pd.DataFrame:
    """
    Adds a single 'momentum_score' column to `df` using:
      - short-term EMA of the metric (shifted to avoid leakage)
      - minus season-to-date expanding mean (shifted)
      - then per-team expanding Z-score of that delta (shifted)
    Requirements: df has columns ['team', metric]
    """

    out = df.copy()

    g = out.groupby("team")[metric]

    # Season-to-date average (no leakage)
    season_avg = g.apply(lambda s: s.shift(1).expanding().mean()).reset_index(level=0, drop=True)

    # Short-term EMA (no leakage)
    short_ema = g.apply(lambda s: s.shift(1).ewm(span=ema_span, adjust=False).mean()).reset_index(level=0, drop=True)

    # Delta (short-term - season-to-date)
    delta = (short_ema - season_avg).astype(float)

    # Per-team expanding Z-score of the delta (no leakage)
    def _expanding_z(x: pd.Series) -> pd.Series:
        mu = x.shift(1).expanding().mean()
        sd = x.shift(1).expanding().std(ddof=1)
        return (x - mu) / sd.replace(0, np.nan)

    momentum_z = (
        out.assign(_delta=delta)
           .groupby("team")["_delta"]
           .apply(_expanding_z)
           .reset_index(level=0, drop=True)
    )

    out["momentum_score"] = momentum_z.astype(float)
    out["momentum_score"] = out["momentum_score"].fillna(0.0).round(4)
    return out


season = load_and_prepare_pbp([2024])
off = offensive_stats(season)
deffense = defensive_stats(season)

off_with_momentum = add_momentum_simple(off, metric="epa_per_play", ema_span=5)
print(off_with_momentum[["season","week","team","momentum_score"]].tail(5))


