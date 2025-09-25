"""
Pre-compute and cache team statistics to speed up API startup
"""

import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
from predictor import assemble_team_game_dataset

def generate_team_data_cache(years=None):
    """
    Generate cached team statistics data to avoid reprocessing on each startup

    Args:
        years: List of years to process (default: 2015-2024)

    Returns:
        dict: Cached data with team stats and metadata
    """

    if years is None:
        years = list(range(2015, 2025))  # 2015-2024

    print(f"Generating team data cache for years: {years}")

    # Use the existing assemble_team_game_dataset function to generate the complete dataset
    # This ensures we get the exact same data structure as used in predictions
    print("Generating complete team dataset with all features...")
    final_df = assemble_team_game_dataset(
        years=years,
        include_momentum=True,
        momentum_metric="explosive_rate",
        ema_span=5
    )
    print(f"Complete dataset generated: {len(final_df)} team-game records")

    # Step 7: Create cache data structure
    cache_data = {
        'team_dataset': final_df,
        'years_processed': years,
        'cache_generated_at': datetime.now().isoformat(),
        'total_games': len(final_df),
        'total_plays_processed': 'N/A (using cached pipeline)',
        'data_summary': {
            'seasons': len(years),
            'teams_per_season': final_df.groupby('season')['team'].nunique().to_dict(),
            'feature_count': len(final_df.columns)
        }
    }

    return cache_data

def save_cache(cache_data, cache_file='team_data_cache.pkl'):
    """Save cache data to pickle file"""

    cache_path = os.path.join('artifacts', cache_file)

    # Create artifacts directory if it doesn't exist
    os.makedirs('artifacts', exist_ok=True)

    print(f"Saving cache to {cache_path}...")
    with open(cache_path, 'wb') as f:
        pickle.dump(cache_data, f)

    # Also save as CSV for inspection
    csv_path = cache_path.replace('.pkl', '.csv')
    cache_data['team_dataset'].to_csv(csv_path, index=False)

    print(f"Cache saved successfully:")
    print(f"  - Binary cache: {cache_path}")
    print(f"  - CSV backup: {csv_path}")
    print(f"  - {cache_data['total_games']} team-game records")
    print(f"  - Generated from {cache_data['total_plays_processed']} plays")

def load_cache(cache_file='team_data_cache.pkl'):
    """
    Load cached team data

    Returns:
        tuple: (team_dataset_df, cache_metadata) or (None, None) if not found
    """

    cache_path = os.path.join('artifacts', cache_file)

    if not os.path.exists(cache_path):
        print(f"Cache file not found: {cache_path}")
        return None, None

    try:
        print(f"Loading cache from {cache_path}...")
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)

        team_dataset = cache_data['team_dataset']

        print(f"Cache loaded successfully:")
        print(f"  - {len(team_dataset)} team-game records")
        print(f"  - Years: {cache_data['years_processed']}")
        print(f"  - Generated: {cache_data['cache_generated_at']}")

        return team_dataset, cache_data

    except Exception as e:
        print(f"Error loading cache: {e}")
        return None, None

def is_cache_stale(cache_metadata, max_age_days=7):
    """
    Check if cache is stale and needs regeneration

    Args:
        cache_metadata: Metadata from cached data
        max_age_days: Maximum age in days before cache is considered stale

    Returns:
        bool: True if cache should be regenerated
    """

    if not cache_metadata:
        return True

    try:
        cache_date = datetime.fromisoformat(cache_metadata['cache_generated_at'])
        days_old = (datetime.now() - cache_date).days

        print(f"Cache age: {days_old} days (max: {max_age_days})")

        return days_old > max_age_days

    except Exception as e:
        print(f"Error checking cache age: {e}")
        return True

def get_or_generate_cache(force_regenerate=False, years=None):
    """
    Get cached data or generate if needed

    Args:
        force_regenerate: Force cache regeneration even if valid cache exists
        years: Years to process (default: 2015-2024)

    Returns:
        pd.DataFrame: Team dataset ready for predictions
    """

    # Try to load existing cache
    if not force_regenerate:
        team_dataset, cache_metadata = load_cache()

        if team_dataset is not None and not is_cache_stale(cache_metadata):
            print("Using cached team data")
            return team_dataset
        else:
            print("Cache is stale or missing, regenerating...")

    # Generate new cache
    print("Generating new team data cache...")
    cache_data = generate_team_data_cache(years)
    save_cache(cache_data)

    return cache_data['team_dataset']

if __name__ == "__main__":
    # Generate cache for testing
    print("=== NFL Team Data Cache Generator ===\n")

    # Generate cache with default years (2015-2024)
    team_data = get_or_generate_cache(force_regenerate=True)

    print(f"\n=== Cache Generation Complete ===")
    print(f"Team dataset shape: {team_data.shape}")
    print(f"Years covered: {team_data['season'].min()}-{team_data['season'].max()}")
    print(f"Teams: {team_data['team'].nunique()}")
    print(f"Features: {len(team_data.columns)}")

    # Show sample data
    print("\nSample team-game records:")
    print(team_data[['season', 'week', 'team', 'epa_per_play', 'success_rate']].head(10))