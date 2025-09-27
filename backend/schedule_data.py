"""
NFL 2025 Schedule Data Parser
Handles parsing of the NFL schedule CSV and provides weekly game data
"""

import pandas as pd
import os
from datetime import datetime
from typing import Dict, List, Any
from lib.nfl_teams import NFL_TEAMS

class NFLScheduleParser:
    def __init__(self, csv_path: str = None):
        self.csv_path = csv_path or '../nfl-2025.csv'
        self.schedule_data = {}
        self.team_name_mapping = self._create_team_mapping()

    def _create_team_mapping(self) -> Dict[str, str]:
        """Create mapping from full team names to abbreviations"""
        mapping = {}

        # Create mapping from NFL_TEAMS data
        for team in NFL_TEAMS:
            # Add various formats of team names
            full_name = f"{team['city']} {team['name']}"
            mapping[full_name] = team['id']

            # Handle special cases and alternate names
            city = team['city']
            name = team['name']

            # Add just city name for some cases
            mapping[city] = team['id']

            # Handle specific team name variations
            if team['id'] == 'ne':
                mapping['New England Patriots'] = 'ne'
            elif team['id'] == 'no':
                mapping['New Orleans Saints'] = 'no'
            elif team['id'] == 'tb':
                mapping['Tampa Bay Buccaneers'] = 'tb'
            elif team['id'] == 'gb':
                mapping['Green Bay Packers'] = 'gb'
            elif team['id'] == 'sf':
                mapping['San Francisco 49ers'] = 'sf'
            elif team['id'] == 'lac':
                mapping['Los Angeles Chargers'] = 'lac'
            elif team['id'] == 'lar':
                mapping['Los Angeles Rams'] = 'lar'
            elif team['id'] == 'lv':
                mapping['Las Vegas Raiders'] = 'lv'
            elif team['id'] == 'nyg':
                mapping['New York Giants'] = 'nyg'
            elif team['id'] == 'nyj':
                mapping['New York Jets'] = 'nyj'

        return mapping

    def _normalize_date(self, date_str: str) -> tuple:
        """Normalize various date formats to standard format"""
        try:
            # Handle formats like "5/9/25 0:20" and "14/09/2025 17:00"
            if ' ' in date_str:
                date_part, time_part = date_str.split(' ', 1)
            else:
                date_part = date_str
                time_part = "00:00"

            # Try different date formats
            for fmt in ['%d/%m/%Y', '%m/%d/%y', '%d/%m/%y', '%m/%d/%Y']:
                try:
                    date_obj = datetime.strptime(date_part, fmt)
                    # Convert 2-digit years to 20xx
                    if date_obj.year < 2000:
                        date_obj = date_obj.replace(year=date_obj.year + 2000)

                    # Format time
                    if ':' in time_part:
                        time_formatted = time_part
                    else:
                        time_formatted = "00:00"

                    return date_obj.strftime('%Y-%m-%d'), time_formatted
                except ValueError:
                    continue

            # Fallback
            return "2025-01-01", "00:00"

        except Exception as e:
            print(f"Error parsing date '{date_str}': {e}")
            return "2025-01-01", "00:00"

    def _map_team_name(self, team_name: str) -> str:
        """Map full team name to abbreviation"""
        team_name = team_name.strip()

        if team_name in self.team_name_mapping:
            return self.team_name_mapping[team_name]

        # Try to find partial matches
        for full_name, abbr in self.team_name_mapping.items():
            if team_name.lower() in full_name.lower() or full_name.lower() in team_name.lower():
                return abbr

        print(f"Warning: Could not map team name '{team_name}'")
        return team_name.lower().replace(' ', '')[:3]

    def load_schedule(self) -> Dict[int, List[Dict[str, Any]]]:
        """Load and parse the NFL schedule CSV"""
        try:
            # Check if file exists
            if not os.path.exists(self.csv_path):
                print(f"Schedule file not found: {self.csv_path}")
                return {}

            # Read CSV
            df = pd.read_csv(self.csv_path)

            # Initialize schedule data
            schedule_by_week = {}

            for _, row in df.iterrows():
                try:
                    week = int(row['Week'])
                    home_team = self._map_team_name(row['Home Team'])
                    away_team = self._map_team_name(row['Away Team'])
                    date_str, time_str = self._normalize_date(row['Date'])
                    location = row.get('Location', '')

                    game = {
                        'home_team': home_team,
                        'away_team': away_team,
                        'date': date_str,
                        'time': time_str,
                        'location': location,
                        'week': week,
                        'match_number': row.get('Match Number', 0)
                    }

                    if week not in schedule_by_week:
                        schedule_by_week[week] = []

                    schedule_by_week[week].append(game)

                except Exception as e:
                    print(f"Error processing row: {row.to_dict()}, Error: {e}")
                    continue

            self.schedule_data = schedule_by_week
            print(f"Loaded schedule for {len(schedule_by_week)} weeks")
            return schedule_by_week

        except Exception as e:
            print(f"Error loading schedule: {e}")
            return {}

    def get_week_games(self, week: int) -> List[Dict[str, Any]]:
        """Get all games for a specific week"""
        return self.schedule_data.get(week, [])

    def get_available_weeks(self) -> List[int]:
        """Get list of available weeks"""
        return sorted(self.schedule_data.keys())

    def get_week_summary(self, week: int) -> Dict[str, Any]:
        """Get summary information for a week"""
        games = self.get_week_games(week)
        if not games:
            return {}

        # Get date range for the week
        dates = [game['date'] for game in games if game['date']]
        min_date = min(dates) if dates else None
        max_date = max(dates) if dates else None

        return {
            'week': week,
            'game_count': len(games),
            'date_range': {
                'start': min_date,
                'end': max_date if max_date != min_date else None
            }
        }

# Global schedule parser instance
schedule_parser = None

def get_schedule_parser():
    """Get or create the global schedule parser"""
    global schedule_parser
    if schedule_parser is None:
        schedule_parser = NFLScheduleParser()
        schedule_parser.load_schedule()
    return schedule_parser