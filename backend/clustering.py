import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class NFLClustering:
    """
    Advanced clustering system for NFL teams and game contexts.

    Implements:
    - Team style clustering (6 clusters) based on offensive/defensive play styles
    - Game context clustering (4 clusters) based on situational factors
    - Cluster validation and interpretation
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.team_style_clusterer = None
        self.game_context_clusterer = None
        self.team_style_scaler = StandardScaler()
        self.game_context_scaler = StandardScaler()
        self.cluster_interpretations = {}

    def fit_team_style_clustering(self, df: pd.DataFrame, n_clusters: int = 6) -> Dict:
        """
        Cluster teams by their playing style using offensive and defensive characteristics.

        Features used:
        - pass_rate, rush_rate
        - play_action_rate, shotgun_rate
        - explosive_rate, success_rate
        - pressure_sack_rate (defense)
        - allowed_explosive_rate, allowed_success_rate
        """
        print(f">> Fitting team style clustering with {n_clusters} clusters...")

        # Select team style features
        style_features = [
            'pass_rate', 'explosive_rate', 'success_rate',
            'play_action_rate', 'shotgun_rate', 'sack_rate',
            'allowed_explosive_rate', 'allowed_success_rate',
            'pressure_sack_rate', 'third_down_conv_rate',
            'red_zone_td_rate', 'third_down_conv_rate_allowed'
        ]

        # Filter available features (handle missing columns gracefully)
        available_features = [f for f in style_features if f in df.columns]
        if len(available_features) < 6:
            print(f"Warning: Only {len(available_features)} style features available")

        # Aggregate team-season characteristics
        team_profiles = df.groupby(['season', 'team'])[available_features].mean().reset_index()

        # Remove any rows with all NaN values
        team_profiles = team_profiles.dropna(subset=available_features, how='all')
        team_profiles[available_features] = team_profiles[available_features].fillna(team_profiles[available_features].mean())

        if len(team_profiles) < n_clusters:
            print(f"Warning: Only {len(team_profiles)} team-seasons available for {n_clusters} clusters")
            n_clusters = max(2, len(team_profiles) // 10)  # Adjust cluster count

        # Fit clustering
        X_style = self.team_style_scaler.fit_transform(team_profiles[available_features])
        self.team_style_clusterer = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        team_profiles['team_style_cluster'] = self.team_style_clusterer.fit_predict(X_style)

        # Calculate silhouette score
        silhouette = silhouette_score(X_style, team_profiles['team_style_cluster'])

        # Interpret clusters
        cluster_summary = self._interpret_team_clusters(team_profiles, available_features)
        self.cluster_interpretations['team_style'] = cluster_summary

        print(f"Team style clustering completed - Silhouette score: {silhouette:.3f}")
        for cluster_id, description in cluster_summary.items():
            print(f"  Cluster {cluster_id}: {description}")

        return {
            'team_profiles': team_profiles,
            'features_used': available_features,
            'silhouette_score': silhouette,
            'cluster_centers': self.team_style_clusterer.cluster_centers_,
            'interpretations': cluster_summary
        }

    def fit_game_context_clustering(self, df: pd.DataFrame, n_clusters: int = 4) -> Dict:
        """
        Cluster games by context using situational factors.

        Features used:
        - Home/away advantage
        - Week of season (early/mid/late)
        - Division game indicator
        - Opponent strength metrics
        """
        print(f">> Fitting game context clustering with {n_clusters} clusters...")

        # Create game context features
        context_df = df.copy()

        # Home field advantage
        context_df['is_home'] = (context_df.get('team_is_home', 0) == 1).astype(int)

        # Season timing
        context_df['early_season'] = (context_df['week'] <= 6).astype(int)
        context_df['mid_season'] = ((context_df['week'] > 6) & (context_df['week'] <= 13)).astype(int)
        context_df['late_season'] = (context_df['week'] > 13).astype(int)

        # Opponent strength (use composite if available, otherwise use EPA)
        if 'opp_composite_strength' in context_df.columns:
            context_df['opponent_strength'] = context_df['opp_composite_strength']
        elif 'opp_season_epa_avg' in context_df.columns:
            context_df['opponent_strength'] = context_df['opp_season_epa_avg']
        else:
            context_df['opponent_strength'] = 0.0  # Neutral if no strength data

        # Division games (simplified heuristic - teams that play each other multiple times)
        context_df['division_game'] = 0  # Default to non-division

        # Game difficulty (combine opponent strength and home/away)
        context_df['game_difficulty'] = context_df['opponent_strength'] - (context_df['is_home'] * 0.1)

        context_features = [
            'is_home', 'early_season', 'mid_season', 'late_season',
            'opponent_strength', 'division_game', 'game_difficulty'
        ]

        # Remove missing values
        context_df = context_df.dropna(subset=context_features, how='any')

        if len(context_df) < n_clusters:
            print(f"Warning: Only {len(context_df)} games available for {n_clusters} clusters")
            n_clusters = max(2, len(context_df) // 100)

        # Fit clustering
        X_context = self.game_context_scaler.fit_transform(context_df[context_features])
        self.game_context_clusterer = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        context_df['game_context_cluster'] = self.game_context_clusterer.fit_predict(X_context)

        # Calculate silhouette score
        silhouette = silhouette_score(X_context, context_df['game_context_cluster'])

        # Interpret clusters
        cluster_summary = self._interpret_context_clusters(context_df, context_features)
        self.cluster_interpretations['game_context'] = cluster_summary

        print(f"Game context clustering completed - Silhouette score: {silhouette:.3f}")
        for cluster_id, description in cluster_summary.items():
            print(f"  Context {cluster_id}: {description}")

        return {
            'context_profiles': context_df,
            'features_used': context_features,
            'silhouette_score': silhouette,
            'cluster_centers': self.game_context_clusterer.cluster_centers_,
            'interpretations': cluster_summary
        }

    def predict_team_style_cluster(self, team_data: pd.DataFrame) -> np.ndarray:
        """Predict team style cluster for new data."""
        if self.team_style_clusterer is None:
            raise ValueError("Team style clustering not fitted yet")

        # Use same features as training
        features = self.cluster_interpretations.get('team_style_features', [])
        X = self.team_style_scaler.transform(team_data[features])
        return self.team_style_clusterer.predict(X)

    def predict_game_context_cluster(self, context_data: pd.DataFrame) -> np.ndarray:
        """Predict game context cluster for new data."""
        if self.game_context_clusterer is None:
            raise ValueError("Game context clustering not fitted yet")

        features = self.cluster_interpretations.get('context_features', [])
        X = self.game_context_scaler.transform(context_data[features])
        return self.game_context_clusterer.predict(X)

    def add_cluster_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add both team style and game context cluster features to the dataset.
        """
        df_enhanced = df.copy()

        # Add team style clusters (aggregated by season-team)
        if self.team_style_clusterer is not None:
            try:
                # Get team style features
                style_features = [
                    'pass_rate', 'explosive_rate', 'success_rate',
                    'play_action_rate', 'shotgun_rate', 'sack_rate',
                    'allowed_explosive_rate', 'allowed_success_rate',
                    'pressure_sack_rate', 'third_down_conv_rate',
                    'red_zone_td_rate', 'third_down_conv_rate_allowed'
                ]
                available_style_features = [f for f in style_features if f in df.columns]

                if len(available_style_features) >= 6:
                    # Aggregate team characteristics by season
                    team_profiles = df.groupby(['season', 'team'])[available_style_features].mean().reset_index()
                    team_profiles = team_profiles.fillna(team_profiles[available_style_features].mean())

                    # Predict clusters
                    X_style = self.team_style_scaler.transform(team_profiles[available_style_features])
                    team_profiles['team_style_cluster'] = self.team_style_clusterer.predict(X_style)

                    # Merge back to original dataset
                    df_enhanced = df_enhanced.merge(
                        team_profiles[['season', 'team', 'team_style_cluster']],
                        on=['season', 'team'], how='left'
                    )
                    df_enhanced['team_style_cluster'] = df_enhanced['team_style_cluster'].fillna(-1).astype(int)
                else:
                    df_enhanced['team_style_cluster'] = -1

            except Exception as e:
                print(f"Warning: Could not add team style clusters: {e}")
                df_enhanced['team_style_cluster'] = -1
        else:
            df_enhanced['team_style_cluster'] = -1

        # Add game context clusters
        if self.game_context_clusterer is not None:
            try:
                # Create context features
                context_df = df_enhanced.copy()
                context_df['is_home'] = (context_df.get('team_is_home', 0) == 1).astype(int)
                context_df['early_season'] = (context_df['week'] <= 6).astype(int)
                context_df['mid_season'] = ((context_df['week'] > 6) & (context_df['week'] <= 13)).astype(int)
                context_df['late_season'] = (context_df['week'] > 13).astype(int)

                if 'opp_composite_strength' in context_df.columns:
                    context_df['opponent_strength'] = context_df['opp_composite_strength']
                elif 'opp_season_epa_avg' in context_df.columns:
                    context_df['opponent_strength'] = context_df['opp_season_epa_avg']
                else:
                    context_df['opponent_strength'] = 0.0

                context_df['division_game'] = 0
                context_df['game_difficulty'] = context_df['opponent_strength'] - (context_df['is_home'] * 0.1)

                context_features = [
                    'is_home', 'early_season', 'mid_season', 'late_season',
                    'opponent_strength', 'division_game', 'game_difficulty'
                ]

                # Fill missing values and predict
                context_df[context_features] = context_df[context_features].fillna(0)
                X_context = self.game_context_scaler.transform(context_df[context_features])
                df_enhanced['game_context_cluster'] = self.game_context_clusterer.predict(X_context)

            except Exception as e:
                print(f"Warning: Could not add game context clusters: {e}")
                df_enhanced['game_context_cluster'] = -1
        else:
            df_enhanced['game_context_cluster'] = -1

        print(f"Added cluster features:")
        print(f"  - team_style_cluster: {df_enhanced['team_style_cluster'].nunique()} unique values")
        print(f"  - game_context_cluster: {df_enhanced['game_context_cluster'].nunique()} unique values")

        return df_enhanced

    def _interpret_team_clusters(self, team_profiles: pd.DataFrame, features: List[str]) -> Dict[int, str]:
        """Generate interpretable descriptions for team style clusters."""
        interpretations = {}

        for cluster_id in sorted(team_profiles['team_style_cluster'].unique()):
            cluster_data = team_profiles[team_profiles['team_style_cluster'] == cluster_id]
            cluster_means = cluster_data[features].mean()

            # Interpret based on key characteristics
            pass_rate = cluster_means.get('pass_rate', 0.5)
            explosive_rate = cluster_means.get('explosive_rate', 0.1)
            success_rate = cluster_means.get('success_rate', 0.4)
            pressure_rate = cluster_means.get('pressure_sack_rate', 0.1)

            style_desc = []
            if pass_rate > 0.6:
                style_desc.append("Pass-heavy")
            elif pass_rate < 0.4:
                style_desc.append("Run-heavy")
            else:
                style_desc.append("Balanced")

            if explosive_rate > 0.12:
                style_desc.append("Big-play")
            elif explosive_rate < 0.08:
                style_desc.append("Conservative")

            if pressure_rate > 0.12:
                style_desc.append("High-pressure defense")
            elif pressure_rate < 0.08:
                style_desc.append("Bend-don't-break defense")

            interpretations[cluster_id] = " + ".join(style_desc) if style_desc else f"Style {cluster_id}"

        return interpretations

    def _interpret_context_clusters(self, context_df: pd.DataFrame, features: List[str]) -> Dict[int, str]:
        """Generate interpretable descriptions for game context clusters."""
        interpretations = {}

        for cluster_id in sorted(context_df['game_context_cluster'].unique()):
            cluster_data = context_df[context_df['game_context_cluster'] == cluster_id]
            cluster_means = cluster_data[features].mean()

            # Interpret based on key characteristics
            is_home = cluster_means.get('is_home', 0.5)
            early_season = cluster_means.get('early_season', 0.3)
            late_season = cluster_means.get('late_season', 0.3)
            opponent_strength = cluster_means.get('opponent_strength', 0.0)

            context_desc = []
            if is_home > 0.7:
                context_desc.append("Home games")
            elif is_home < 0.3:
                context_desc.append("Away games")

            if early_season > 0.5:
                context_desc.append("Early season")
            elif late_season > 0.5:
                context_desc.append("Late season")

            if opponent_strength > 0.1:
                context_desc.append("vs Strong opponents")
            elif opponent_strength < -0.1:
                context_desc.append("vs Weak opponents")

            interpretations[cluster_id] = " + ".join(context_desc) if context_desc else f"Context {cluster_id}"

        return interpretations

    def get_cluster_validation_report(self) -> Dict:
        """Generate comprehensive cluster validation report."""
        report = {
            "team_style_clustering": {},
            "game_context_clustering": {}
        }

        if self.team_style_clusterer is not None:
            report["team_style_clustering"] = {
                "n_clusters": self.team_style_clusterer.n_clusters,
                "interpretations": self.cluster_interpretations.get('team_style', {}),
                "fitted": True
            }
        else:
            report["team_style_clustering"]["fitted"] = False

        if self.game_context_clusterer is not None:
            report["game_context_clustering"] = {
                "n_clusters": self.game_context_clusterer.n_clusters,
                "interpretations": self.cluster_interpretations.get('game_context', {}),
                "fitted": True
            }
        else:
            report["game_context_clustering"]["fitted"] = False

        return report