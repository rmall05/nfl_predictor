import pandas as pd
import numpy as np
import nfl_data_py as nfl

def get_expected_team_game_stats(
    years=range(2015, 2026),
    window=4,
    w_off=0.5,
    w_def=0.4,
    w_lg=0.1
):
    """
    Returns a DataFrame with expected stats for each team in each game:
    - exp_pass_yards
    - exp_rush_yards
    - exp_turnovers
    - exp_epa_total
    - exp_epa_per_play
    Uses a simple blend of:
      team recent offense (rolling window),
      opponent recent defense-allowed (rolling window),
      league average.
    """

    # 1) Schedules (to map game <-> home/away/opponent)
    sched = nfl.import_schedules(years=list(years)).rename(columns={"home_team": "home_team_abbr", "away_team": "away_team_abbr"})
    sched = sched[["season", "week", "game_id", "gameday", "home_team_abbr", "away_team_abbr"]].dropna(subset=["game_id"])

    # 2) Play-by-play
    pbp = nfl.import_pbp_data(years=list(years), downcast=True)

    # Keep only regular plays (no penalties-as-plays double counting, etc.)
    # nflfastR flags: 'play_type', 'pass', 'rush', 'epa', 'yards_gained', etc.
    pbp = pbp[pbp["play_type"].isin(["run", "pass"]) & pbp["epa"].notna()].copy()

    # Offensive team (posteam) perspective
    pbp["turnover"] = ((pbp.get("interception", 0).fillna(0) > 0) | (pbp.get("fumble_lost", 0).fillna(0) > 0)).astype(int)

    # Passing and rushing yards from offense perspective
    # nflfastR provides 'yards_gained'; we’ll split by play_type.
    pbp["pass_yards"] = np.where(pbp["play_type"] == "pass", pbp["yards_gained"], 0)
    pbp["rush_yards"] = np.where(pbp["play_type"] == "run", pbp["yards_gained"], 0)

    # 3) Aggregate to per-game, per-team (offense) totals
    off_game = (
        pbp.groupby(["season", "week", "game_id", "posteam", "defteam"], as_index=False)
           .agg(
               plays=("epa", "size"),
               pass_plays=("pass_yards", lambda s: (s != 0).sum()),
               rush_plays=("rush_yards", lambda s: (s != 0).sum()),
               pass_yards=("pass_yards", "sum"),
               rush_yards=("rush_yards", "sum"),
               turnovers=("turnover", "sum"),
               epa_total=("epa", "sum")
           )
    )
    off_game["epa_per_play"] = off_game["epa_total"] / off_game["plays"].replace(0, np.nan)

    off_game = off_game.rename(columns={"posteam": "team", "defteam": "opponent"})

    # 4) Join schedule info (gameday for proper chronological rolling)
    off_game = off_game.merge(
        sched[["game_id", "gameday"]],
        on="game_id",
        how="left"
    )
    off_game["gameday"] = pd.to_datetime(off_game["gameday"])

    # 5) Build a "defense-allowed" view by flipping perspective:
    # What each opponent (defense) allowed to its offenses in each game.
    def_allowed = off_game.copy()
    def_allowed = def_allowed.rename(columns={
        "team": "off_team",
        "opponent": "team"  # now 'team' is the defense
    })
    # These columns are what the defense allowed:
    allowed_cols = ["pass_yards", "rush_yards", "turnovers", "epa_total", "epa_per_play", "plays"]
    def_allowed = def_allowed[["season", "week", "game_id", "team", "off_team", "gameday"] + allowed_cols]
    def_allowed = def_allowed.rename(columns={
        "pass_yards": "allowed_pass_yards",
        "rush_yards": "allowed_rush_yards",
        "turnovers": "allowed_turnovers",
        "epa_total": "allowed_epa_total",
        "epa_per_play": "allowed_epa_per_play",
        "plays": "allowed_plays"
    })

    # 6) Compute league averages per season (simple means)
    league_avg = (
        off_game.groupby("season", as_index=False)[["pass_yards", "rush_yards", "turnovers", "epa_total", "epa_per_play"]]
                .mean()
                .rename(columns={
                    "pass_yards": "lg_pass_yards",
                    "rush_yards": "lg_rush_yards",
                    "turnovers": "lg_turnovers",
                    "epa_total": "lg_epa_total",
                    "epa_per_play": "lg_epa_per_play"
                })
    )

    # 7) Rolling means (recent games), offense view
    off_game = off_game.sort_values(["team", "gameday"]).copy()
    for col in ["pass_yards", "rush_yards", "turnovers", "epa_total", "epa_per_play"]:
        off_game[f"off_{col}_rolling"] = (
            off_game.groupby("team")[col]
                    .apply(lambda s: s.shift(1).rolling(window=window, min_periods=1).mean())
                    .reset_index(level=0, drop=True)
        )

    # 8) Rolling means (recent games), defense-allowed view
    def_allowed = def_allowed.sort_values(["team", "gameday"]).copy()
    for col in ["allowed_pass_yards", "allowed_rush_yards", "allowed_turnovers", "allowed_epa_total", "allowed_epa_per_play"]:
        def_allowed[f"{col}_rolling"] = (
            def_allowed.groupby("team")[col]
                       .apply(lambda s: s.shift(1).rolling(window=window, min_periods=1).mean())
                       .reset_index(level=0, drop=True)
        )

    # 9) Merge offense + defense-allowed + league averages
    df = off_game.merge(
        def_allowed[["season", "week", "game_id", "off_team", "team",
                     "allowed_pass_yards_rolling", "allowed_rush_yards_rolling",
                     "allowed_turnovers_rolling", "allowed_epa_total_rolling",
                     "allowed_epa_per_play_rolling"]],
        left_on=["season", "week", "game_id", "team", "opponent"],
        right_on=["season", "week", "game_id", "off_team", "team"],
        how="left",
        suffixes=("", "_y")
    )

    # Clean merge helper cols
    df = df.drop(columns=["off_team", "team_y"]).rename(columns={"team_x": "team"})

    df = df.merge(league_avg, on="season", how="left")

    # 10) Blended expectations
    def blend(off_recent, def_recent, lg_avg):
        return w_off * off_recent + w_def * def_recent + w_lg * lg_avg

    df["exp_pass_yards"] = blend(df["off_pass_yards_rolling"], df["allowed_pass_yards_rolling"], df["lg_pass_yards"])
    df["exp_rush_yards"] = blend(df["off_rush_yards_rolling"], df["allowed_rush_yards_rolling"], df["lg_rush_yards"])
    df["exp_turnovers"]  = blend(df["off_turnovers_rolling"],  df["allowed_turnovers_rolling"],  df["lg_turnovers"])
    df["exp_epa_total"]  = blend(df["off_epa_total_rolling"],  df["allowed_epa_total_rolling"],  df["lg_epa_total"])
    df["exp_epa_per_play"] = blend(df["off_epa_per_play_rolling"], df["allowed_epa_per_play_rolling"], df["lg_epa_per_play"])

    # Helpful final columns
    keep_cols = [
        "season", "week", "game_id", "gameday", "team", "opponent",
        "exp_pass_yards", "exp_rush_yards", "exp_turnovers", "exp_epa_total", "exp_epa_per_play",
        # Optional: show the components you blended
        "off_pass_yards_rolling", "allowed_pass_yards_rolling", "lg_pass_yards",
        "off_rush_yards_rolling", "allowed_rush_yards_rolling", "lg_rush_yards",
        "off_turnovers_rolling",  "allowed_turnovers_rolling",  "lg_turnovers",
        "off_epa_total_rolling",  "allowed_epa_total_rolling",  "lg_epa_total",
        "off_epa_per_play_rolling", "allowed_epa_per_play_rolling", "lg_epa_per_play"
    ]

    return df[keep_cols].sort_values(["season", "week", "game_id", "team"]).reset_index(drop=True)


# Example usage:
ev = get_expected_team_game_stats(years=range(2015, 2016), window=4)
print(ev.head())

if __name__ == "__main__":
    print("▶ Starting...", flush=True)
    print("✅ Computed EVs. Shape:", ev.shape, flush=True)
