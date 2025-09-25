"""
Process 2025 NFL player stats and convert to team-level metrics for model integration
"""

import pandas as pd
import numpy as np


def aggregate_2025_player_stats_to_teams(csv_path):
    """
    Convert 2025 player statistics to team-level metrics that match our model's expected features

    Args:
        csv_path: Path to the 2025 player stats CSV file

    Returns:
        DataFrame with team-level statistics for 2025 season
    """

    print(f"Loading 2025 player stats from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Group by team to aggregate stats
    team_stats = []

    for team in df['recent_team'].unique():
        team_data = df[df['recent_team'] == team].copy()

        # Calculate games played (use max games any player has played)
        games_played = team_data['games'].max()

        # Offensive Statistics (aggregate QB, RB, WR, TE stats)
        offense = team_data[team_data['position'].isin(['QB', 'RB', 'WR', 'TE', 'FB'])].copy()

        # Passing stats (primarily QB)
        qb_stats = offense[offense['position'] == 'QB']
        total_passing_yards = qb_stats['passing_yards'].sum()
        total_attempts = qb_stats['attempts'].sum()
        total_passing_epa = qb_stats['passing_epa'].sum()
        passing_tds = qb_stats['passing_tds'].sum()
        passing_ints = qb_stats['passing_interceptions'].sum()
        sacks_allowed = qb_stats['sacks_suffered'].sum()
        sack_yards_lost = qb_stats['sack_yards_lost'].sum()

        # Rushing stats (RB + QB + others)
        total_carries = offense['carries'].sum()
        total_rushing_yards = offense['rushing_yards'].sum()
        total_rushing_epa = offense['rushing_epa'].sum()
        rushing_tds = offense['rushing_tds'].sum()
        rushing_fumbles = offense['rushing_fumbles'].sum()

        # Receiving stats
        total_receptions = offense['receptions'].sum()
        total_targets = offense['targets'].sum()
        total_receiving_yards = offense['receiving_yards'].sum()
        total_receiving_epa = offense['receiving_epa'].sum()
        receiving_tds = offense['receiving_tds'].sum()

        # Defensive Statistics
        defense = team_data[team_data['position'].isin(['DE', 'DT', 'LB', 'CB', 'SAF', 'EDGE'])].copy()
        def_sacks = defense['def_sacks'].sum()
        def_ints = defense['def_interceptions'].sum()
        def_fumbles_forced = defense['def_fumbles_forced'].sum()
        def_tds = defense['def_tds'].sum()
        def_tackles = defense['def_tackles_solo'].sum() + defense['def_tackle_assists'].sum()

        # Calculate derived metrics (matching our model's expected features)

        # Estimate total plays (passing attempts + carries, approximate)
        total_plays = max(1, total_attempts + total_carries)  # Avoid division by zero

        # EPA per play (offensive)
        epa_per_play = (total_passing_epa + total_rushing_epa + total_receiving_epa) / total_plays if total_plays > 0 else 0

        # Success rate (approximate from EPA - if EPA > 0, consider it successful)
        successful_plays = len(offense[offense['passing_epa'] > 0]) + len(offense[offense['rushing_epa'] > 0])
        success_rate = successful_plays / total_plays if total_plays > 0 else 0.42  # Default to league average

        # Explosive play rate (approximate from big plays)
        # Use yards per attempt metrics as proxy
        explosive_rate = 0.14  # Default, will improve with more detailed data
        if total_attempts > 0:
            avg_pass_yards = total_passing_yards / total_attempts
            if avg_pass_yards > 8:  # Above average passing
                explosive_rate += 0.02

        # Third down conversion rate (approximate)
        third_down_conv_rate = 0.40  # League average, no direct data available

        # Red zone TD rate (approximate from TD efficiency)
        total_tds = passing_tds + rushing_tds + receiving_tds
        red_zone_td_rate = min(0.8, total_tds / games_played / 3) if games_played > 0 else 0.58

        # Pass rate
        pass_rate = total_attempts / total_plays if total_plays > 0 else 0.60

        # Sack rate
        sack_rate = sacks_allowed / max(1, total_attempts + sacks_allowed) if total_attempts > 0 else 0.07

        # Defensive metrics (approximate what the team allows)
        # Use inverse of defensive performance as proxy for what offense allows
        allowed_epa_per_play = -epa_per_play * 0.8  # Rough estimate
        allowed_success_rate = 1 - success_rate if success_rate < 0.5 else 0.42

        # Defensive pressure rate
        pressure_sack_rate = def_sacks / games_played / 35 if games_played > 0 else 0.07  # ~35 pass attempts per game

        # Create team record
        team_record = {
            'season': 2025,
            'week': games_played,  # Use as proxy for data completeness
            'team': team,
            'games_played': games_played,

            # Offensive metrics
            'epa_per_play': epa_per_play,
            'success_rate': min(1.0, max(0.0, success_rate)),
            'explosive_rate': explosive_rate,
            'third_down_conv_rate': third_down_conv_rate,
            'red_zone_td_rate': red_zone_td_rate,
            'pass_rate': min(1.0, max(0.0, pass_rate)),
            'sack_rate': min(0.2, max(0.0, sack_rate)),

            # Volume stats
            'plays': total_plays,
            'pass_yards': total_passing_yards,
            'rush_yards': total_rushing_yards,
            'turnovers': passing_ints + rushing_fumbles,
            'epa_total': total_passing_epa + total_rushing_epa + total_receiving_epa,

            # Defensive metrics (approximate)
            'allowed_epa_per_play': allowed_epa_per_play,
            'allowed_success_rate': allowed_success_rate,
            'pressure_sack_rate': pressure_sack_rate,
            'takeaways': def_ints + def_fumbles_forced,

            # Derived metrics
            'yards_per_play': (total_passing_yards + total_rushing_yards) / total_plays if total_plays > 0 else 0,
            'passing_yards_per_attempt': total_passing_yards / total_attempts if total_attempts > 0 else 0,
            'rushing_yards_per_carry': total_rushing_yards / total_carries if total_carries > 0 else 0,

            # Quality indicators (for debugging)
            'total_players': len(team_data),
            'qb_count': len(qb_stats),
            'data_quality': 'good' if games_played >= 3 else 'limited'
        }

        team_stats.append(team_record)

    result_df = pd.DataFrame(team_stats)

    print(f"Processed {len(result_df)} teams from 2025 data")
    print(f"Average games played: {result_df['games_played'].mean():.1f}")
    print(f"EPA range: {result_df['epa_per_play'].min():.3f} to {result_df['epa_per_play'].max():.3f}")

    return result_df


def integrate_2025_data_with_model():
    """
    Load 2025 data and integrate it with the existing model dataset
    """
    # Process 2025 player stats
    csv_path = 'C:/Users/Rakendu Malladi/Desktop/stats_player_reg_2025.csv'
    team_2025 = aggregate_2025_player_stats_to_teams(csv_path)

    return team_2025


if __name__ == "__main__":
    # Test the processing
    team_stats_2025 = integrate_2025_data_with_model()

    print("\nSample 2025 team stats:")
    print(team_stats_2025[['team', 'epa_per_play', 'success_rate', 'explosive_rate', 'games_played']].head())

    # Save for integration
    team_stats_2025.to_csv('backend/2025_team_stats.csv', index=False)
    print(f"\n2025 team stats saved to backend/2025_team_stats.csv")