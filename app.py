import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
from typing import Dict, List, Optional, Tuple
import json
import io
import base64
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Fantasy Football Draft Analyzer Pro",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded"
)

class FantasyDataAPI:
    """Real API handler with Sleeper, Underdog, and league sync support"""
    
    def __init__(self):
        self.sleeper_base_url = "https://api.sleeper.app/v1"
        # Note: Underdog doesn't have a public API - you'll need to use web scraping or find alternative sources
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Fantasy-Draft-Analyzer/1.0',
            'Accept': 'application/json'
        })
    
    def get_sleeper_players(self) -> Dict:
        """Get all NFL players from Sleeper API"""
        try:
            response = self.session.get(f"{self.sleeper_base_url}/players/nfl", timeout=30)
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Failed to fetch Sleeper players: {response.status_code}")
                return {}
        except Exception as e:
            st.error(f"Error fetching Sleeper players: {e}")
            return {}
    
    def get_sleeper_trending(self, sport: str = "nfl", add_drop: str = "add", hours: int = 24, limit: int = 25) -> List:
        """Get trending players from Sleeper"""
        try:
            response = self.session.get(
                f"{self.sleeper_base_url}/players/{sport}/trending/{add_drop}",
                params={"lookback_hours": hours, "limit": limit},
                timeout=15
            )
            if response.status_code == 200:
                return response.json()
            return []
        except Exception as e:
            st.warning(f"Could not fetch trending data: {e}")
            return []
    
    def get_sleeper_league_info(self, league_id: str) -> Dict:
        """Get league information from Sleeper"""
        if not league_id:
            return {}
        
        try:
            # Get league details
            league_response = self.session.get(f"{self.sleeper_base_url}/league/{league_id}", timeout=15)
            if league_response.status_code != 200:
                st.error(f"League {league_id} not found")
                return {}
            
            league_info = league_response.json()
            
            # Get users in league
            users_response = self.session.get(f"{self.sleeper_base_url}/league/{league_id}/users", timeout=15)
            users = users_response.json() if users_response.status_code == 200 else []
            
            # Get rosters
            rosters_response = self.session.get(f"{self.sleeper_base_url}/league/{league_id}/rosters", timeout=15)
            rosters = rosters_response.json() if rosters_response.status_code == 200 else []
            
            # Get draft info if available
            drafts_response = self.session.get(f"{self.sleeper_base_url}/league/{league_id}/drafts", timeout=15)
            drafts = drafts_response.json() if drafts_response.status_code == 200 else []
            
            return {
                'league': league_info,
                'users': users,
                'rosters': rosters,
                'drafts': drafts
            }
            
        except Exception as e:
            st.error(f"Error fetching league data: {e}")
            return {}
    
    def get_sleeper_draft_picks(self, draft_id: str) -> List:
        """Get draft picks from a Sleeper draft"""
        if not draft_id:
            return []
        
        try:
            response = self.session.get(f"{self.sleeper_base_url}/draft/{draft_id}/picks", timeout=15)
            if response.status_code == 200:
                return response.json()
            return []
        except Exception as e:
            st.warning(f"Could not fetch draft picks: {e}")
            return []
    
    def process_sleeper_data(self, players_dict: Dict, trending_data: List = None) -> pd.DataFrame:
        """Process Sleeper API data into usable DataFrame"""
        processed_players = []
        
        for player_id, player_info in players_dict.items():
            # Only include active NFL players
            if (player_info.get('active') and 
                player_info.get('position') in ['QB', 'RB', 'WR', 'TE'] and
                player_info.get('team')):
                
                # Calculate trending score
                trending_score = 0
                if trending_data:
                    for trend_player in trending_data:
                        if trend_player.get('player_id') == player_id:
                            trending_score = trend_player.get('count', 0)
                            break
                
                processed_players.append({
                    'player_id': player_id,
                    'player_name': f"{player_info.get('first_name', '')} {player_info.get('last_name', '')}".strip(),
                    'position': player_info.get('position'),
                    'team': player_info.get('team'),
                    'age': player_info.get('age'),
                    'years_exp': player_info.get('years_exp', 0),
                    'height': player_info.get('height'),
                    'weight': player_info.get('weight'),
                    'college': player_info.get('college'),
                    'injury_status': player_info.get('injury_status'),
                    'trending_score': trending_score,
                    # These would need to be supplemented from other sources
                    'sleeper_adp': None,  # Sleeper doesn't provide ADP directly
                    'projected_points': 0,
                    'auction_value': 0
                })
        
        return pd.DataFrame(processed_players)
    
    def get_fantasypros_adp(self, scoring: str = "PPR") -> pd.DataFrame:
        """
        Fetch ADP data from FantasyPros (requires web scraping or paid API access)
        
        NOTE: FantasyPros requires either:
        1. Paid API access ($30-50/month)
        2. Web scraping (may violate ToS)
        3. Manual CSV download
        
        This is a framework for integration
        """
        try:
            # This would require FantasyPros API key or web scraping
            # For now, return empty DataFrame with instructions
            st.info("""
            **FantasyPros Integration Setup Required:**
            
            Option 1 (Recommended): FantasyPros API
            - Sign up at fantasypros.com/api
            - Get API key ($30-50/month)
            - Add key to Streamlit secrets
            
            Option 2: Manual CSV Upload
            - Download ADP data from FantasyPros
            - Use CSV import feature below
            
            Option 3: Web Scraping (Advanced)
            - Implement BeautifulSoup scraping
            - May violate terms of service
            """)
            return pd.DataFrame()
            
        except Exception as e:
            st.warning(f"FantasyPros integration not configured: {e}")
            return pd.DataFrame()
    
    def get_enhanced_player_data(self, use_real_apis: bool = True) -> pd.DataFrame:
        """Get comprehensive player data from real APIs when possible"""
        if use_real_apis:
            # Get real Sleeper data
            sleeper_players = self.get_sleeper_players()
            trending_data = self.get_sleeper_trending()
            
            if sleeper_players:
                base_df = self.process_sleeper_data(sleeper_players, trending_data)
                
                # Add FantasyPros ADP if available
                fp_adp = self.get_fantasypros_adp()
                if not fp_adp.empty:
                    base_df = pd.merge(base_df, fp_adp, on='player_name', how='left')
                
                # Fill in missing advanced metrics with estimates
                base_df = self._enhance_with_estimates(base_df)
                
                return base_df
            else:
                st.warning("Using mock data - real API connection failed")
        
        # Fallback to mock data
        return self._get_mock_enhanced_data()
    
    def _enhance_with_estimates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add estimated advanced metrics to real player data"""
        enhanced_df = df.copy()
        
        # Estimate projections based on position and experience
        position_multipliers = {'QB': 280, 'RB': 180, 'WR': 160, 'TE': 140}
        
        for _, row in enhanced_df.iterrows():
            base_projection = position_multipliers.get(row['position'], 100)
            
            # Adjust based on experience and trending
            exp_modifier = min(row['years_exp'] * 0.1, 0.5) if row['years_exp'] else 0
            trending_modifier = row['trending_score'] * 0.01 if row['trending_score'] else 0
            
            enhanced_df.loc[enhanced_df['player_id'] == row['player_id'], 'projected_points'] = (
                base_projection + (base_projection * exp_modifier) + (base_projection * trending_modifier)
            )
            
            # Add other estimates
            enhanced_df.loc[enhanced_df['player_id'] == row['player_id'], 'ceiling'] = enhanced_df.loc[enhanced_df['player_id'] == row['player_id'], 'projected_points'] * 1.4
            enhanced_df.loc[enhanced_df['player_id'] == row['player_id'], 'floor'] = enhanced_df.loc[enhanced_df['player_id'] == row['player_id'], 'projected_points'] * 0.6
            enhanced_df.loc[enhanced_df['player_id'] == row['player_id'], 'injury_risk'] = 0.3 + (row['age'] - 25) * 0.02 if row['age'] else 0.3
            enhanced_df.loc[enhanced_df['player_id'] == row['player_id'], 'auction_value'] = enhanced_df.loc[enhanced_df['player_id'] == row['player_id'], 'projected_points'] * 0.2
        
        return df
    
    def _get_mock_enhanced_data(self) -> pd.DataFrame:
        """Enhanced mock data with all analytics features"""
        np.random.seed(42)  # For reproducible results
        
        players_data = [
            # RBs
            {"player_name": "Christian McCaffrey", "position": "RB", "team": "SF", "age": 27, "sleeper_adp": 1.2, "underdog_adp": 1.1, 
             "targets_2023": 85, "red_zone_touches": 45, "snap_share": 0.78, "injury_risk": 0.6, "ceiling": 350, "floor": 180,
             "strength_of_schedule": 0.52, "projected_points": 280, "auction_value": 65, "dynasty_value": 85},
            {"player_name": "Saquon Barkley", "position": "RB", "team": "PHI", "age": 26, "sleeper_adp": 28.8, "underdog_adp": 29.4,
             "targets_2023": 65, "red_zone_touches": 38, "snap_share": 0.71, "injury_risk": 0.7, "ceiling": 320, "floor": 160,
             "strength_of_schedule": 0.48, "projected_points": 250, "auction_value": 45, "dynasty_value": 82},
            {"player_name": "Derrick Henry", "position": "RB", "team": "BAL", "age": 30, "sleeper_adp": 15.7, "underdog_adp": 16.3,
             "targets_2023": 25, "red_zone_touches": 42, "snap_share": 0.65, "injury_risk": 0.3, "ceiling": 290, "floor": 170,
             "strength_of_schedule": 0.55, "projected_points": 240, "auction_value": 52, "dynasty_value": 65},
            {"player_name": "Austin Ekeler", "position": "RB", "team": "WAS", "age": 29, "sleeper_adp": 34.5, "underdog_adp": 35.1,
             "targets_2023": 75, "red_zone_touches": 28, "snap_share": 0.68, "injury_risk": 0.5, "ceiling": 270, "floor": 140,
             "strength_of_schedule": 0.46, "projected_points": 220, "auction_value": 38, "dynasty_value": 70},
            {"player_name": "Brian Robinson Jr.", "position": "RB", "team": "WAS", "age": 25, "sleeper_adp": 85.2, "underdog_adp": 87.1,
             "targets_2023": 35, "red_zone_touches": 18, "snap_share": 0.55, "injury_risk": 0.3, "ceiling": 200, "floor": 80,
             "strength_of_schedule": 0.46, "projected_points": 150, "auction_value": 15, "dynasty_value": 75},
             
            # WRs
            {"player_name": "Cooper Kupp", "position": "WR", "team": "LAR", "age": 31, "sleeper_adp": 12.3, "underdog_adp": 11.8,
             "targets_2023": 145, "red_zone_touches": 25, "snap_share": 0.82, "injury_risk": 0.6, "ceiling": 310, "floor": 160,
             "strength_of_schedule": 0.51, "projected_points": 265, "auction_value": 55, "dynasty_value": 72},
            {"player_name": "Tyreek Hill", "position": "WR", "team": "MIA", "age": 30, "sleeper_adp": 22.1, "underdog_adp": 21.7,
             "targets_2023": 135, "red_zone_touches": 18, "snap_share": 0.85, "injury_risk": 0.2, "ceiling": 330, "floor": 180,
             "strength_of_schedule": 0.49, "projected_points": 275, "auction_value": 48, "dynasty_value": 78},
            {"player_name": "Stefon Diggs", "position": "WR", "team": "HOU", "age": 30, "sleeper_adp": 25.4, "underdog_adp": 24.9,
             "targets_2023": 140, "red_zone_touches": 22, "snap_share": 0.88, "injury_risk": 0.3, "ceiling": 300, "floor": 170,
             "strength_of_schedule": 0.53, "projected_points": 260, "auction_value": 46, "dynasty_value": 74},
            {"player_name": "Rome Odunze", "position": "WR", "team": "CHI", "age": 22, "sleeper_adp": 95.3, "underdog_adp": 98.2,
             "targets_2023": 0, "red_zone_touches": 0, "snap_share": 0.0, "injury_risk": 0.2, "ceiling": 250, "floor": 60,
             "strength_of_schedule": 0.47, "projected_points": 140, "auction_value": 12, "dynasty_value": 85},
             
            # QBs
            {"player_name": "Josh Allen", "position": "QB", "team": "BUF", "age": 28, "sleeper_adp": 8.5, "underdog_adp": 9.2,
             "targets_2023": 0, "red_zone_touches": 35, "snap_share": 1.0, "injury_risk": 0.4, "ceiling": 380, "floor": 220,
             "strength_of_schedule": 0.50, "projected_points": 320, "auction_value": 58, "dynasty_value": 88},
            {"player_name": "Lamar Jackson", "position": "QB", "team": "BAL", "age": 27, "sleeper_adp": 31.2, "underdog_adp": 30.8,
             "targets_2023": 0, "red_zone_touches": 42, "snap_share": 1.0, "injury_risk": 0.5, "ceiling": 370, "floor": 200,
             "strength_of_schedule": 0.55, "projected_points": 310, "auction_value": 42, "dynasty_value": 90},
            {"player_name": "C.J. Stroud", "position": "QB", "team": "HOU", "age": 22, "sleeper_adp": 65.8, "underdog_adp": 68.2,
             "targets_2023": 0, "red_zone_touches": 18, "snap_share": 1.0, "injury_risk": 0.3, "ceiling": 340, "floor": 180,
             "strength_of_schedule": 0.53, "projected_points": 280, "auction_value": 22, "dynasty_value": 95},
             
            # TEs
            {"player_name": "Travis Kelce", "position": "TE", "team": "KC", "age": 34, "sleeper_adp": 18.9, "underdog_adp": 19.5,
             "targets_2023": 125, "red_zone_touches": 28, "snap_share": 0.75, "injury_risk": 0.4, "ceiling": 280, "floor": 140,
             "strength_of_schedule": 0.48, "projected_points": 230, "auction_value": 50, "dynasty_value": 65},
            {"player_name": "Mark Andrews", "position": "TE", "team": "BAL", "age": 29, "sleeper_adp": 42.1, "underdog_adp": 43.8,
             "targets_2023": 95, "red_zone_touches": 22, "snap_share": 0.68, "injury_risk": 0.6, "ceiling": 250, "floor": 110,
             "strength_of_schedule": 0.55, "projected_points": 200, "auction_value": 35, "dynasty_value": 72},
        ]
        
        df = pd.DataFrame(players_data)
        
        # Add bye weeks
        bye_weeks = {
            "SF": 9, "PHI": 5, "BAL": 14, "WAS": 7, "LAR": 6, 
            "MIA": 12, "HOU": 10, "CHI": 13, "BUF": 11, "KC": 8
        }
        df['bye_week'] = df['team'].map(bye_weeks)
        
        # Add handcuff relationships
        df['handcuff'] = ""
        df.loc[df['player_name'] == "Christian McCaffrey", 'handcuff'] = "Jordan Mason"
        df.loc[df['player_name'] == "Derrick Henry", 'handcuff'] = "Justice Hill"
        
        return df

class CSVDataManager:
    """Handle CSV imports for expert rankings and custom data"""
    
    @staticmethod
    def process_uploaded_csv(uploaded_file, data_type: str) -> pd.DataFrame:
        """Process uploaded CSV files"""
        try:
            df = pd.read_csv(uploaded_file)
            
            if data_type == "expert_rankings":
                return CSVDataManager._process_expert_rankings_csv(df)
            elif data_type == "adp_data":
                return CSVDataManager._process_adp_csv(df)
            elif data_type == "projections":
                return CSVDataManager._process_projections_csv(df)
            else:
                return df
                
        except Exception as e:
            st.error(f"Error processing CSV: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def _process_expert_rankings_csv(df: pd.DataFrame) -> pd.DataFrame:
        """Process expert rankings CSV with flexible column mapping"""
        # Expected columns: player_name, position, expert_name, rank
        # Try to map common column names
        
        column_mapping = {
            'name': 'player_name',
            'player': 'player_name',
            'full_name': 'player_name',
            'pos': 'position',
            'expert': 'expert_name',
            'expert_name': 'expert_name',
            'rank': 'expert_rank',
            'ranking': 'expert_rank',
            'overall_rank': 'expert_rank'
        }
        
        # Apply mappings
        df_processed = df.copy()
        for old_col, new_col in column_mapping.items():
            if old_col in df_processed.columns:
                df_processed = df_processed.rename(columns={old_col: new_col})
        
        # Validate required columns
        required_cols = ['player_name', 'expert_rank']
        missing_cols = [col for col in required_cols if col not in df_processed.columns]
        
        if missing_cols:
            st.error(f"Missing required columns in expert rankings CSV: {missing_cols}")
            return pd.DataFrame()
        
        return df_processed
    
    @staticmethod
    def _process_adp_csv(df: pd.DataFrame) -> pd.DataFrame:
        """Process ADP data CSV"""
        column_mapping = {
            'name': 'player_name',
            'player': 'player_name',
            'adp': 'avg_adp',
            'average_draft_position': 'avg_adp',
            'sleeper_adp': 'sleeper_adp',
            'underdog_adp': 'underdog_adp',
            'pos': 'position'
        }
        
        df_processed = df.copy()
        for old_col, new_col in column_mapping.items():
            if old_col in df_processed.columns:
                df_processed = df_processed.rename(columns={old_col: new_col})
        
        return df_processed
    
    @staticmethod
    def _process_projections_csv(df: pd.DataFrame) -> pd.DataFrame:
        """Process fantasy projections CSV"""
        column_mapping = {
            'name': 'player_name',
            'player': 'player_name',
            'projected_points': 'projected_points',
            'projection': 'projected_points',
            'points': 'projected_points',
            'pos': 'position'
        }
        
        df_processed = df.copy()
        for old_col, new_col in column_mapping.items():
            if old_col in df_processed.columns:
                df_processed = df_processed.rename(columns={old_col: new_col})
        
        return df_processed
    
    @staticmethod
    def get_sample_csv_format(data_type: str) -> str:
        """Generate sample CSV format for downloads"""
        if data_type == "expert_rankings":
            return """player_name,position,expert_name,expert_rank
Christian McCaffrey,RB,FantasyPros,1
Josh Allen,QB,FantasyPros,8
Cooper Kupp,WR,FantasyPros,12"""
        
        elif data_type == "adp_data":
            return """player_name,position,team,sleeper_adp,underdog_adp
Christian McCaffrey,RB,SF,1.2,1.1
Josh Allen,QB,BUF,8.5,9.2
Cooper Kupp,WR,LAR,12.3,11.8"""
        
        elif data_type == "projections":
            return """player_name,position,team,projected_points,ceiling,floor
Christian McCaffrey,RB,SF,280,350,180
Josh Allen,QB,BUF,320,380,220
Cooper Kupp,WR,LAR,265,310,160"""
        
        return ""

class LeagueSyncManager:
    """Manage league synchronization with fantasy platforms"""
    
    def __init__(self, api_handler: FantasyDataAPI):
        self.api = api_handler
        
    def sync_sleeper_league(self, league_id: str) -> Dict:
        """Sync with Sleeper league"""
        league_data = self.api.get_sleeper_league_info(league_id)
        
        if not league_data:
            return {"error": "Could not sync with league"}
        
        # Process league information
        league_info = league_data.get('league', {})
        users = league_data.get('users', [])
        rosters = league_data.get('rosters', [])
        drafts = league_data.get('drafts', [])
        
        # Build user mapping
        user_mapping = {user['user_id']: user['display_name'] for user in users}
        
        # Process rosters
        processed_rosters = []
        for roster in rosters:
            roster_info = {
                'roster_id': roster.get('roster_id'),
                'owner': user_mapping.get(roster.get('owner_id'), 'Unknown'),
                'players': roster.get('players', []),
                'wins': roster.get('settings', {}).get('wins', 0),
                'losses': roster.get('settings', {}).get('losses', 0),
                'points_for': roster.get('settings', {}).get('fpts', 0),
                'points_against': roster.get('settings', {}).get('fpts_against', 0)
            }
            processed_rosters.append(roster_info)
        
        # Process draft information
        processed_drafts = []
        for draft in drafts:
            draft_picks = self.api.get_sleeper_draft_picks(draft.get('draft_id'))
            processed_drafts.append({
                'draft_id': draft.get('draft_id'),
                'status': draft.get('status'),
                'type': draft.get('type'),
                'picks': draft_picks
            })
        
        return {
            'league_name': league_info.get('name'),
            'season': league_info.get('season'),
            'total_rosters': league_info.get('total_rosters'),
            'scoring_settings': league_info.get('scoring_settings', {}),
            'roster_positions': league_info.get('roster_positions', []),
            'rosters': processed_rosters,
            'drafts': processed_drafts,
            'users': user_mapping
        }
    
    def get_league_standings(self, sync_data: Dict) -> pd.DataFrame:
        """Create standings from synced league data"""
        if not sync_data.get('rosters'):
            return pd.DataFrame()
        
        standings_data = []
        for roster in sync_data['rosters']:
            standings_data.append({
                'team_name': roster['owner'],
                'wins': roster['wins'],
                'losses': roster['losses'],
                'points_for': roster['points_for'],
                'points_against': roster['points_against'],
                'win_percentage': roster['wins'] / (roster['wins'] + roster['losses']) if (roster['wins'] + roster['losses']) > 0 else 0
            })
        
        standings_df = pd.DataFrame(standings_data)
        return standings_df.sort_values('win_percentage', ascending=False)
    
    def analyze_league_trends(self, sync_data: Dict, players_df: pd.DataFrame) -> Dict:
        """Analyze league-wide trends"""
        if not sync_data.get('rosters'):
            return {}
        
        all_rostered_players = []
        for roster in sync_data['rosters']:
            all_rostered_players.extend(roster['players'])
        
        # Get player info for rostered players
        rostered_df = players_df[players_df['player_id'].isin(all_rostered_players)]
        
        analysis = {
            'most_rostered_positions': rostered_df['position'].value_counts().to_dict(),
            'average_team_age': rostered_df.groupby('position')['age'].mean().to_dict(),
            'rookie_adoption_rate': len(rostered_df[rostered_df['years_exp'] == 0]) / len(rostered_df),
            'total_rostered_players': len(set(all_rostered_players))
        }
        
        return analysis

class AdvancedAnalytics:
    """Advanced analytics and predictive modeling"""
    
    @staticmethod
    def calculate_positional_scarcity(df: pd.DataFrame) -> Dict[str, Dict]:
        """Calculate positional scarcity metrics"""
        scarcity_data = {}
        
        for position in ['RB', 'WR', 'QB', 'TE']:
            pos_players = df[df['position'] == position].copy()
            if pos_players.empty:
                continue
                
            pos_players = pos_players.sort_values('projected_points', ascending=False)
            
            # Calculate drop-off between tiers
            tier_1 = pos_players.head(6)['projected_points'].mean()
            tier_2 = pos_players.iloc[6:12]['projected_points'].mean() if len(pos_players) > 6 else 0
            tier_3 = pos_players.iloc[12:18]['projected_points'].mean() if len(pos_players) > 12 else 0
            
            scarcity_data[position] = {
                'tier_1_avg': tier_1,
                'tier_2_avg': tier_2,
                'tier_3_avg': tier_3,
                'tier_1_to_2_drop': tier_1 - tier_2,
                'tier_2_to_3_drop': tier_2 - tier_3,
                'total_players': len(pos_players),
                'scarcity_score': (tier_1 - tier_3) / tier_1 if tier_3 > 0 else 1.0
            }
            
        return scarcity_data
    
    @staticmethod
    def identify_breakout_candidates(df: pd.DataFrame) -> pd.DataFrame:
        """Identify potential breakout candidates"""
        breakout_factors = df.copy()
        
        # Age factor (younger is better for breakouts)
        breakout_factors['age_score'] = np.where(breakout_factors['age'] <= 25, 1.0,
                                               np.where(breakout_factors['age'] <= 27, 0.7, 0.3))
        
        # Opportunity score (targets, snap share)
        breakout_factors['opportunity_score'] = (
            (breakout_factors['targets_2023'] / breakout_factors['targets_2023'].max()) * 0.6 +
            breakout_factors['snap_share'] * 0.4
        )
        
        # ADP vs projection value
        breakout_factors['adp_avg'] = (breakout_factors['sleeper_adp'] + breakout_factors['underdog_adp']) / 2
        breakout_factors['value_ratio'] = breakout_factors['projected_points'] / breakout_factors['adp_avg']
        
        # Team situation (strength of schedule)
        breakout_factors['situation_score'] = 1 - breakout_factors['strength_of_schedule']
        
        # Combined breakout score
        breakout_factors['breakout_score'] = (
            breakout_factors['age_score'] * 0.25 +
            breakout_factors['opportunity_score'] * 0.35 +
            breakout_factors['value_ratio'] * 0.25 +
            breakout_factors['situation_score'] * 0.15
        )
        
        # Focus on players drafted after round 8
        late_round_players = breakout_factors[breakout_factors['adp_avg'] > 96]
        return late_round_players.nlargest(10, 'breakout_score')
    
    @staticmethod
    def calculate_bust_risk(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate bust risk for highly drafted players"""
        bust_analysis = df.copy()
        
        # Age penalty (older players more likely to decline)
        bust_analysis['age_penalty'] = np.where(bust_analysis['age'] >= 30, 0.2,
                                              np.where(bust_analysis['age'] >= 28, 0.1, 0))
        
        # Injury risk factor
        bust_analysis['injury_penalty'] = bust_analysis['injury_risk'] * 0.3
        
        # Previous year regression (simplified)
        bust_analysis['regression_risk'] = np.random.uniform(0, 0.2, len(bust_analysis))
        
        # Team situation decline
        bust_analysis['team_penalty'] = np.where(bust_analysis['strength_of_schedule'] > 0.55, 0.15, 0)
        
        # Combined bust risk
        bust_analysis['bust_risk_score'] = (
            bust_analysis['age_penalty'] +
            bust_analysis['injury_penalty'] +
            bust_analysis['regression_risk'] +
            bust_analysis['team_penalty']
        )
        
        # Focus on early round picks (high ADP)
        bust_analysis['adp_avg'] = (bust_analysis['sleeper_adp'] + bust_analysis['underdog_adp']) / 2
        early_picks = bust_analysis[bust_analysis['adp_avg'] <= 72]  # First 6 rounds
        
        return early_picks.nlargest(10, 'bust_risk_score')
    
    @staticmethod
    def calculate_opportunity_cost(df: pd.DataFrame, player_a: str, player_b: str) -> Dict:
        """Calculate opportunity cost between two players"""
        player_a_data = df[df['player_name'] == player_a].iloc[0] if not df[df['player_name'] == player_a].empty else None
        player_b_data = df[df['player_name'] == player_b].iloc[0] if not df[df['player_name'] == player_b].empty else None
        
        if player_a_data is None or player_b_data is None:
            return {"error": "One or both players not found"}
        
        return {
            "points_difference": player_a_data['projected_points'] - player_b_data['projected_points'],
            "adp_difference": player_b_data['sleeper_adp'] - player_a_data['sleeper_adp'],
            "ceiling_difference": player_a_data['ceiling'] - player_b_data['ceiling'],
            "floor_difference": player_a_data['floor'] - player_b_data['floor'],
            "injury_risk_difference": player_b_data['injury_risk'] - player_a_data['injury_risk'],
            "recommendation": "Take " + (player_a if player_a_data['projected_points'] > player_b_data['projected_points'] else player_b)
        }

class DraftDayTools:
    """Live draft management tools"""
    
    def __init__(self):
        if 'draft_board' not in st.session_state:
            st.session_state.draft_board = []
        if 'my_team' not in st.session_state:
            st.session_state.my_team = []
        if 'draft_round' not in st.session_state:
            st.session_state.draft_round = 1
        if 'draft_pick' not in st.session_state:
            st.session_state.draft_pick = 1
    
    def draft_player(self, player_name: str, team_name: str = "My Team"):
        """Record a drafted player"""
        pick_info = {
            'round': st.session_state.draft_round,
            'pick': st.session_state.draft_pick,
            'player': player_name,
            'team': team_name,
            'timestamp': datetime.now()
        }
        st.session_state.draft_board.append(pick_info)
        
        if team_name == "My Team":
            st.session_state.my_team.append(player_name)
        
        # Advance pick counter
        st.session_state.draft_pick += 1
        if st.session_state.draft_pick > 12:  # Assuming 12-team league
            st.session_state.draft_pick = 1
            st.session_state.draft_round += 1
    
    def get_available_players(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get players still available"""
        drafted_players = [pick['player'] for pick in st.session_state.draft_board]
        return df[~df['player_name'].isin(drafted_players)]
    
    def analyze_team_needs(self, df: pd.DataFrame) -> Dict:
        """Analyze current team construction"""
        my_players = df[df['player_name'].isin(st.session_state.my_team)]
        
        position_counts = my_players['position'].value_counts().to_dict()
        
        needs = {
            'RB': max(0, 3 - position_counts.get('RB', 0)),
            'WR': max(0, 4 - position_counts.get('WR', 0)),
            'QB': max(0, 2 - position_counts.get('QB', 0)),
            'TE': max(0, 2 - position_counts.get('TE', 0))
        }
        
        total_points = my_players['projected_points'].sum()
        bye_weeks = my_players['bye_week'].value_counts()
        
        return {
            'position_counts': position_counts,
            'needs': needs,
            'total_projected_points': total_points,
            'bye_week_conflicts': bye_weeks[bye_weeks > 2].to_dict() if not bye_weeks.empty else {},
            'team_strength': 'Strong' if total_points > 2000 else 'Average' if total_points > 1500 else 'Needs Work'
        }
    
    def calculate_auction_values(self, df: pd.DataFrame, budget: int = 200) -> pd.DataFrame:
        """Calculate auction values based on projected points"""
        auction_df = df.copy()
        
        # Calculate total points for all relevant players
        relevant_players = auction_df.nlargest(200, 'projected_points')  # Top 200 players
        total_points = relevant_players['projected_points'].sum()
        
        # Baseline: worst starter at each position
        baseline_points = {
            'QB': relevant_players[relevant_players['position'] == 'QB']['projected_points'].iloc[11] if len(relevant_players[relevant_players['position'] == 'QB']) > 11 else 200,
            'RB': relevant_players[relevant_players['position'] == 'RB']['projected_points'].iloc[23] if len(relevant_players[relevant_players['position'] == 'RB']) > 23 else 150,
            'WR': relevant_players[relevant_players['position'] == 'WR']['projected_points'].iloc[35] if len(relevant_players[relevant_players['position'] == 'WR']) > 35 else 130,
            'TE': relevant_players[relevant_players['position'] == 'TE']['projected_points'].iloc[11] if len(relevant_players[relevant_players['position'] == 'TE']) > 11 else 100
        }
        
        # Calculate value over replacement
        auction_df['baseline'] = auction_df['position'].map(baseline_points)
        auction_df['vor'] = auction_df['projected_points'] - auction_df['baseline']
        auction_df['vor'] = auction_df['vor'].clip(lower=0)
        
        # Convert to auction values
        total_vor = auction_df['vor'].sum()
        available_budget = budget * 12 - (12 * 15)  # Total budget minus $1 for each bench spot
        
        auction_df['calculated_auction_value'] = (auction_df['vor'] / total_vor * available_budget).round().astype(int)
        auction_df['calculated_auction_value'] = auction_df['calculated_auction_value'].clip(lower=1)
        
        return auction_df

class PortfolioAnalyzer:
    """Portfolio analysis and team optimization"""
    
    @staticmethod
    def monte_carlo_simulation(df: pd.DataFrame, team_players: List[str], num_simulations: int = 1000) -> Dict:
        """Run Monte Carlo simulation for team projections"""
        if not team_players:
            return {"error": "No players in team"}
        
        team_data = df[df['player_name'].isin(team_players)]
        
        if team_data.empty:
            return {"error": "No valid players found"}
        
        simulated_scores = []
        
        for _ in range(num_simulations):
            weekly_score = 0
            for _, player in team_data.iterrows():
                # Simulate player performance using normal distribution
                mean_points = player['projected_points'] / 17  # Per game
                std_dev = (player['ceiling'] - player['floor']) / 17 / 4  # Rough estimate
                
                # Account for injury risk
                injury_chance = player['injury_risk']
                if np.random.random() > injury_chance:
                    simulated_points = np.random.normal(mean_points, std_dev)
                    weekly_score += max(0, simulated_points)  # No negative points
            
            simulated_scores.append(weekly_score)
        
        simulated_scores = np.array(simulated_scores)
        
        return {
            "mean_score": np.mean(simulated_scores),
            "median_score": np.median(simulated_scores),
            "std_dev": np.std(simulated_scores),
            "percentile_25": np.percentile(simulated_scores, 25),
            "percentile_75": np.percentile(simulated_scores, 75),
            "championship_probability": np.mean(simulated_scores > np.percentile(simulated_scores, 90)),
            "playoff_probability": np.mean(simulated_scores > np.percentile(simulated_scores, 60))
        }
    
    @staticmethod
    def analyze_correlations(df: pd.DataFrame, team_players: List[str]) -> pd.DataFrame:
        """Analyze player correlations in team"""
        team_data = df[df['player_name'].isin(team_players)]
        
        # Create correlation matrix based on team, position, and bye week
        correlation_data = []
        
        for i, player1 in team_data.iterrows():
            for j, player2 in team_data.iterrows():
                if i >= j:
                    continue
                
                # Calculate correlation score
                correlation = 0
                
                # Same team correlation (positive for QB/WR, negative for RB/WR)
                if player1['team'] == player2['team']:
                    if (player1['position'] == 'QB' and player2['position'] == 'WR') or \
                       (player1['position'] == 'WR' and player2['position'] == 'QB'):
                        correlation += 0.3
                    elif player1['position'] == 'RB' and player2['position'] == 'WR':
                        correlation -= 0.1
                
                # Same bye week (negative correlation)
                if player1['bye_week'] == player2['bye_week']:
                    correlation -= 0.2
                
                # Same position (slight negative for weekly lineup decisions)
                if player1['position'] == player2['position']:
                    correlation -= 0.05
                
                correlation_data.append({
                    'player_1': player1['player_name'],
                    'player_2': player2['player_name'],
                    'correlation': correlation,
                    'same_team': player1['team'] == player2['team'],
                    'same_bye': player1['bye_week'] == player2['bye_week']
                })
        
        return pd.DataFrame(correlation_data)
    
    @staticmethod
    def optimize_lineup(df: pd.DataFrame, team_players: List[str]) -> Dict:
        """Optimize starting lineup from roster"""
        team_data = df[df['player_name'].isin(team_players)]
        
        if team_data.empty:
            return {"error": "No players in team"}
        
        # Standard lineup requirements
        lineup_requirements = {
            'QB': 1,
            'RB': 2,
            'WR': 2,
            'TE': 1,
            'FLEX': 1  # RB, WR, or TE
        }
        
        optimal_lineup = {}
        remaining_players = team_data.copy()
        
        # Fill required positions
        for position, count in lineup_requirements.items():
            if position == 'FLEX':
                continue
                
            pos_players = remaining_players[remaining_players['position'] == position]
            best_players = pos_players.nlargest(count, 'projected_points')
            
            optimal_lineup[position] = best_players['player_name'].tolist()
            remaining_players = remaining_players[~remaining_players['player_name'].isin(best_players['player_name'])]
        
        # Fill FLEX spot
        flex_eligible = remaining_players[remaining_players['position'].isin(['RB', 'WR', 'TE'])]
        if not flex_eligible.empty:
            flex_player = flex_eligible.nlargest(1, 'projected_points')
            optimal_lineup['FLEX'] = flex_player['player_name'].tolist()
        
        # Calculate total projected points
        all_starters = []
        for players in optimal_lineup.values():
            all_starters.extend(players)
        
        total_points = team_data[team_data['player_name'].isin(all_starters)]['projected_points'].sum()
        
        return {
            'lineup': optimal_lineup,
            'total_projected_points': total_points,
            'bench_players': remaining_players[~remaining_players['player_name'].isin(all_starters)]['player_name'].tolist()
        }

def create_advanced_visualizations(df: pd.DataFrame):
    """Create advanced visualization charts"""
    
    # Positional scarcity chart
    scarcity_data = AdvancedAnalytics.calculate_positional_scarcity(df)
    
    scarcity_fig = go.Figure()
    positions = list(scarcity_data.keys())
    scarcity_scores = [scarcity_data[pos]['scarcity_score'] for pos in positions]
    
    scarcity_fig.add_trace(go.Bar(
        x=positions,
        y=scarcity_scores,
        marker_color=['red' if score > 0.3 else 'orange' if score > 0.2 else 'green' for score in scarcity_scores],
        text=[f"{score:.2f}" for score in scarcity_scores],
        textposition='auto'
    ))
    
    scarcity_fig.update_layout(
        title="Positional Scarcity Analysis",
        xaxis_title="Position",
        yaxis_title="Scarcity Score (Higher = More Scarce)",
        height=400
    )
    
    # Risk vs Reward scatter
    risk_reward_fig = go.Figure()
    
    for position in df['position'].unique():
        pos_data = df[df['position'] == position]
        risk_reward_fig.add_trace(go.Scatter(
            x=pos_data['injury_risk'],
            y=pos_data['projected_points'],
            mode='markers+text',
            text=pos_data['player_name'],
            textposition="top center",
            name=position,
            marker=dict(size=10, opacity=0.7)
        ))
    
    risk_reward_fig.update_layout(
        title="Risk vs Reward Analysis",
        xaxis_title="Injury Risk",
        yaxis_title="Projected Points",
        height=500
    )
    
    return scarcity_fig, risk_reward_fig

# Main Streamlit App
def main():
    st.title("🏈 Fantasy Football Draft Analyzer Pro")
    st.markdown("*Advanced analytics with real API integration, league sync, and CSV import*")
    
    # Initialize tools
    draft_tools = DraftDayTools()
    csv_manager = CSVDataManager()
    
    # Sidebar configuration
    st.sidebar.header("🔧 Configuration")
    
    # API Settings
    st.sidebar.subheader("API Settings")
    use_real_apis = st.sidebar.checkbox("Use Real APIs", value=True, help="Uncheck to use mock data for testing")
    
    # League Sync
    st.sidebar.subheader("League Sync")
    league_platform = st.sidebar.selectbox("Platform", ["None", "Sleeper", "ESPN (Coming Soon)", "Yahoo (Coming Soon)"])
    
    league_id = ""
    if league_platform == "Sleeper":
        league_id = st.sidebar.text_input(
            "Sleeper League ID", 
            help="Find this in your Sleeper league URL: sleeper.app/leagues/YOUR_LEAGUE_ID"
        )
        
        if st.sidebar.button("Sync League") and league_id:
            with st.spinner("Syncing with Sleeper league..."):
                api = FantasyDataAPI()
                league_sync = LeagueSyncManager(api)
                sync_result = league_sync.sync_sleeper_league(league_id)
                
                if "error" not in sync_result:
                    st.session_state.league_sync_data = sync_result
                    st.sidebar.success("League synced successfully!")
                else:
                    st.sidebar.error(f"Sync failed: {sync_result['error']}")
    
    # CSV Import Section
    st.sidebar.subheader("📁 CSV Data Import")
    
    csv_upload_type = st.sidebar.selectbox(
        "Data Type",
        ["Expert Rankings", "ADP Data", "Projections"],
        help="Choose what type of data you're uploading"
    )
    
    uploaded_file = st.sidebar.file_uploader(
        f"Upload {csv_upload_type} CSV",
        type=['csv'],
        help=f"Upload a CSV file containing {csv_upload_type.lower()}"
    )
    
    if uploaded_file:
        data_type_mapping = {
            "Expert Rankings": "expert_rankings",
            "ADP Data": "adp_data", 
            "Projections": "projections"
        }
        
        processed_csv = csv_manager.process_uploaded_csv(
            uploaded_file, 
            data_type_mapping[csv_upload_type]
        )
        
        if not processed_csv.empty:
            st.sidebar.success(f"{csv_upload_type} uploaded successfully!")
            st.session_state[f"uploaded_{data_type_mapping[csv_upload_type]}"] = processed_csv
    
    # Download sample CSV formats
    if st.sidebar.button("Download Sample CSV Format"):
        data_type_mapping = {
            "Expert Rankings": "expert_rankings",
            "ADP Data": "adp_data",
            "Projections": "projections"
        }
        
        sample_content = csv_manager.get_sample_csv_format(data_type_mapping[csv_upload_type])
        st.sidebar.download_button(
            f"Download {csv_upload_type} Template",
            sample_content,
            f"{csv_upload_type.lower().replace(' ', '_')}_template.csv",
            "text/csv"
        )
    
    # League settings
    st.sidebar.subheader("League Settings")
    league_size = st.sidebar.selectbox("League Size", [8, 10, 12, 14, 16], index=2)
    scoring_format = st.sidebar.selectbox("Scoring Format", ["PPR", "Half-PPR", "Standard"], index=0)
    draft_type = st.sidebar.selectbox("Draft Type", ["Snake", "Auction", "Best Ball"], index=0)
    
    # Load and merge data
    if st.sidebar.button("Load/Refresh Data") or 'enhanced_data' not in st.session_state:
        with st.spinner("Loading enhanced fantasy football data..."):
            api = FantasyDataAPI()
            base_data = api.get_enhanced_player_data(use_real_apis)
            
            # Merge with uploaded CSV data if available
            merged_data = base_data.copy()
            
            # Merge expert rankings
            if 'uploaded_expert_rankings' in st.session_state:
                expert_csv = st.session_state.uploaded_expert_rankings
                merged_data = pd.merge(merged_data, expert_csv, on='player_name', how='left', suffixes=('', '_csv'))
            
            # Merge ADP data
            if 'uploaded_adp_data' in st.session_state:
                adp_csv = st.session_state.uploaded_adp_data
                merged_data = pd.merge(merged_data, adp_csv, on='player_name', how='left', suffixes=('', '_csv'))
            
            # Merge projections
            if 'uploaded_projections' in st.session_state:
                proj_csv = st.session_state.uploaded_projections
                merged_data = pd.merge(merged_data, proj_csv, on='player_name', how='left', suffixes=('', '_csv'))
                
                # Use CSV projections if available
                if 'projected_points_csv' in merged_data.columns:
                    merged_data['projected_points'] = merged_data['projected_points_csv'].fillna(merged_data['projected_points'])
            
            st.session_state.enhanced_data = merged_data
            
            # Display data source info
            if use_real_apis:
                st.sidebar.info("✅ Using real Sleeper API data")
            else:
                st.sidebar.info("🧪 Using mock data for testing")
    
    if 'enhanced_data' not in st.session_state or st.session_state.enhanced_data.empty:
        st.error("No data available. Please refresh data or check API connections.")
        return
    
    df = st.session_state.enhanced_data
    
    # Display league sync information if available
    if 'league_sync_data' in st.session_state:
        sync_data = st.session_state.league_sync_data
        st.success(f"🔗 Synced with league: **{sync_data.get('league_name', 'Unknown')}** ({sync_data.get('total_rosters', 0)} teams)")
    
    # Main navigation with enhanced tabs - FIXED VERSION
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📊 Advanced Analytics", 
        "🎯 Predictive Models", 
        "🚀 Draft Day Tools", 
        "📈 Portfolio Analysis",
        "💰 Auction Tools",
        "🔄 Trade Analyzer",
        "📋 My Draft Board",
        "🏆 League Sync"
    ])
    
    with tab1:
        st.header("Advanced Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Positional Scarcity")
            scarcity_data = AdvancedAnalytics.calculate_positional_scarcity(df)
            
            # Display scarcity metrics
            for position, data in scarcity_data.items():
                with st.expander(f"{position} Analysis"):
                    st.write(f"**Scarcity Score:** {data['scarcity_score']:.2f}")
                    st.write(f"**Tier 1 → Tier 2 Drop:** {data['tier_1_to_2_drop']:.1f} points")
                    st.write(f"**Tier 2 → Tier 3 Drop:** {data['tier_2_to_3_drop']:.1f} points")
                    st.write(f"**Total Available:** {data['total_players']} players")
            
            # Strength of Schedule
            st.subheader("Strength of Schedule")
            sos_chart = px.bar(
                df.groupby('team')['strength_of_schedule'].first().reset_index(),
                x='team', y='strength_of_schedule',
                title="Team Strength of Schedule",
                color='strength_of_schedule',
                color_continuous_scale='RdYlGn_r'
            )
            st.plotly_chart(sos_chart, use_container_width=True)
        
        with col2:
            st.subheader("Advanced Visualizations")
            scarcity_fig, risk_reward_fig = create_advanced_visualizations(df)
            st.plotly_chart(scarcity_fig, use_container_width=True)
            st.plotly_chart(risk_reward_fig, use_container_width=True)
        
        # Opportunity Cost Calculator
        st.subheader("Opportunity Cost Calculator")
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            player_a = st.selectbox("Player A", df['player_name'].tolist(), key="opp_cost_a")
        with col2:
            player_b = st.selectbox("Player B", df['player_name'].tolist(), key="opp_cost_b")
        with col3:
            if st.button("Calculate Opportunity Cost"):
                cost_analysis = AdvancedAnalytics.calculate_opportunity_cost(df, player_a, player_b)
                if "error" not in cost_analysis:
                    st.write("**Analysis Results:**")
                    for key, value in cost_analysis.items():
                        if key != "recommendation":
                            st.write(f"**{key.replace('_', ' ').title()}:** {value:.2f}")
                    st.success(f"**Recommendation:** {cost_analysis['recommendation']}")
    
    with tab2:
        st.header("Predictive Models")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🚀 Breakout Candidates")
            breakout_candidates = AdvancedAnalytics.identify_breakout_candidates(df)
            
            if not breakout_candidates.empty:
                display_cols = ['player_name', 'position', 'team', 'age', 'breakout_score', 'adp_avg']
                available_cols = [col for col in display_cols if col in breakout_candidates.columns]
                st.dataframe(breakout_candidates[available_cols])
                
                # Breakout visualization
                breakout_fig = px.scatter(
                    breakout_candidates,
                    x='adp_avg',
                    y='breakout_score',
                    color='position',
                    text='player_name',
                    title="Breakout Candidates by ADP"
                )
                breakout_fig.update_traces(textposition="top center")
                st.plotly_chart(breakout_fig, use_container_width=True)
        
        with col2:
            st.subheader("⚠️ Bust Alert System")
            bust_risks = AdvancedAnalytics.calculate_bust_risk(df)
            
            if not bust_risks.empty:
                display_cols = ['player_name', 'position', 'team', 'age', 'bust_risk_score', 'injury_risk']
                available_cols = [col for col in display_cols if col in bust_risks.columns]
                st.dataframe(bust_risks[available_cols])
                
                # Bust risk visualization
                bust_fig = px.scatter(
                    bust_risks,
                    x='adp_avg',
                    y='bust_risk_score',
                    color='position',
                    text='player_name',
                    title="Bust Risk Analysis"
                )
                bust_fig.update_traces(textposition="top center")
                st.plotly_chart(bust_fig, use_container_width=True)
        
        # Sleeper Picks
        st.subheader("💎 Deep Sleeper Recommendations")
        deep_sleepers = df[df['sleeper_adp'] > 120].nlargest(10, 'projected_points')
        if not deep_sleepers.empty:
            st.dataframe(deep_sleepers[['player_name', 'position', 'team', 'sleeper_adp', 'projected_points', 'ceiling']])
        
        # Handcuff Analysis
        st.subheader("🔗 Handcuff Recommendations")
        handcuff_players = df[df['handcuff'] != ""]
        if not handcuff_players.empty:
            for _, player in handcuff_players.iterrows():
                st.write(f"**{player['player_name']}** → Handcuff: **{player['handcuff']}**")
    
    with tab3:
        st.header("Draft Day Tools")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Live Draft Board")
            
            # Draft a player
            col1a, col1b, col1c = st.columns([2, 1, 1])
            with col1a:
                available_players = draft_tools.get_available_players(df)
                selected_player = st.selectbox(
                    "Select Player to Draft", 
                    [""] + available_players['player_name'].tolist()
                )
            with col1b:
                team_name = st.text_input("Team Name", value="My Team")
            with col1c:
                if st.button("Draft Player") and selected_player:
                    draft_tools.draft_player(selected_player, team_name)
                    st.rerun()
            
            # Show draft board
            if st.session_state.draft_board:
                draft_df = pd.DataFrame(st.session_state.draft_board)
                st.dataframe(draft_df[['round', 'pick', 'player', 'team']])
            
            # Available players with recommendations
            st.subheader("Available Players")
            if not available_players.empty:
                # Add draft recommendations
                current_round = st.session_state.draft_round
                round_start = (current_round - 1) * league_size + 1
                round_end = current_round * league_size
                
                available_players['draft_value'] = np.where(
                    (available_players['sleeper_adp'] >= round_start - 6) & 
                    (available_players['sleeper_adp'] <= round_end + 6),
                    "Good Value", "Reach/Wait"
                )
                
                display_cols = ['player_name', 'position', 'team', 'sleeper_adp', 'projected_points', 'draft_value']
                st.dataframe(available_players[display_cols].head(20))
        
        with col2:
            st.subheader("My Team Analysis")
            
            if st.session_state.my_team:
                team_analysis = draft_tools.analyze_team_needs(df)
                
                st.write("**Current Roster:**")
                for player in st.session_state.my_team:
                    player_data = df[df['player_name'] == player].iloc[0]
                    st.write(f"• {player} ({player_data['position']})")
                
                st.write(f"**Projected Points:** {team_analysis['total_projected_points']:.1f}")
                st.write(f"**Team Strength:** {team_analysis['team_strength']}")
                
                st.write("**Position Needs:**")
                for pos, need in team_analysis['needs'].items():
                    if need > 0:
                        st.write(f"• {pos}: Need {need} more")
                
                if team_analysis['bye_week_conflicts']:
                    st.warning("**Bye Week Conflicts:**")
                    for week, count in team_analysis['bye_week_conflicts'].items():
                        st.write(f"• Week {week}: {count} players")
            else:
                st.info("Draft players to see team analysis")
    
    with tab4:
        st.header("Portfolio Analysis")
        
        if st.session_state.my_team:
            portfolio = PortfolioAnalyzer()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Monte Carlo Simulation")
                
                num_simulations = st.slider("Number of Simulations", 100, 5000, 1000)
                
                if st.button("Run Simulation"):
                    with st.spinner("Running Monte Carlo simulation..."):
                        results = portfolio.monte_carlo_simulation(df, st.session_state.my_team, num_simulations)
                    
                    if "error" not in results:
                        st.write("**Simulation Results:**")
                        st.write(f"**Mean Weekly Score:** {results['mean_score']:.1f}")
                        st.write(f"**Median Weekly Score:** {results['median_score']:.1f}")
                        st.write(f"**25th Percentile:** {results['percentile_25']:.1f}")
                        st.write(f"**75th Percentile:** {results['percentile_75']:.1f}")
                        st.write(f"**Championship Probability:** {results['championship_probability']:.1%}")
                        st.write(f"**Playoff Probability:** {results['playoff_probability']:.1%}")
                        
                        # Create distribution chart
                        simulated_scores = np.random.normal(
                            results['mean_score'], 
                            results['std_dev'], 
                            1000
                        )
                        
                        fig = px.histogram(
                            x=simulated_scores,
                            title="Weekly Score Distribution",
                            nbins=30
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Team Correlation Analysis")
                correlations = portfolio.analyze_correlations(df, st.session_state.my_team)
                
                if not correlations.empty:
                    st.dataframe(correlations)
                    
                    # Correlation heatmap
                    if len(correlations) > 0:
                        corr_matrix = correlations.pivot_table(
                            index='player_1', 
                            columns='player_2', 
                            values='correlation'
                        ).fillna(0)
                        
                        fig = px.imshow(
                            corr_matrix,
                            title="Player Correlation Matrix",
                            color_continuous_scale='RdBu'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Optimal Lineup")
                optimal = portfolio.optimize_lineup(df, st.session_state.my_team)
                
                if "error" not in optimal:
                    st.write(f"**Projected Points:** {optimal['total_projected_points']:.1f}")
                    
                    for position, players in optimal['lineup'].items():
                        st.write(f"**{position}:** {', '.join(players)}")
                    
                    if optimal['bench_players']:
                        st.write(f"**Bench:** {', '.join(optimal['bench_players'])}")
        else:
            st.info("Draft players to see portfolio analysis")
    
    with tab5:
        st.header("Auction Tools")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Auction Value Calculator")
            
            budget = st.number_input("Total Budget", value=200, min_value=100, max_value=1000)
            
            if st.button("Calculate Auction Values"):
                draft_tools = DraftDayTools()
                auction_values = draft_tools.calculate_auction_values(df, budget)
                
                st.dataframe(
                    auction_values[['player_name', 'position', 'projected_points', 'calculated_auction_value']]
                    .sort_values('calculated_auction_value', ascending=False)
                    .head(50)
                )
        
        with col2:
            st.subheader("Value Over Replacement")
            
            # Show VOR leaders by position
            for position in ['QB', 'RB', 'WR', 'TE']:
                pos_data = df[df['position'] == position].copy()
                if not pos_data.empty:
                    st.write(f"**{position} VOR Leaders:**")
                    
                    # Calculate simple VOR
                    baseline = pos_data['projected_points'].quantile(0.6)  # Rough baseline
                    pos_data['vor'] = pos_data['projected_points'] - baseline
                    
                    top_vor = pos_data.nlargest(5, 'vor')
                    for _, player in top_vor.iterrows():
                        st.write(f"• {player['player_name']}: {player['vor']:.1f}")
    
    with tab6:
        st.header("Trade Analyzer")
        
        st.subheader("Evaluate Trade Proposals")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Give Away:**")
            give_players = st.multiselect(
                "Select players to trade away",
                df['player_name'].tolist(),
                key="give_players"
            )
        
        with col2:
            st.write("**Receive:**")
            receive_players = st.multiselect(
                "Select players to receive",
                df['player_name'].tolist(),
                key="receive_players"
            )
        
        if st.button("Analyze Trade") and give_players and receive_players:
            give_data = df[df['player_name'].isin(give_players)]
            receive_data = df[df['player_name'].isin(receive_players)]
            
            give_points = give_data['projected_points'].sum()
            receive_points = receive_data['projected_points'].sum()
            
            give_value = give_data['auction_value'].sum()
            receive_value = receive_data['auction_value'].sum()
            
            st.write("**Trade Analysis:**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Points Difference", f"{receive_points - give_points:+.1f}")
            with col2:
                st.metric("Value Difference", f"${receive_value - give_value:+}")
            with col3:
                recommendation = "Accept" if receive_points > give_points else "Decline"
                st.metric("Recommendation", recommendation)
            
            # Detailed breakdown
            st.write("**Giving Away:**")
            st.dataframe(give_data[['player_name', 'position', 'projected_points', 'auction_value']])
            
            st.write("**Receiving:**")
            st.dataframe(receive_data[['player_name', 'position', 'projected_points', 'auction_value']])
    
    with tab7:
        st.header("My Draft Board")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Complete Draft History")
            if st.session_state.draft_board:
                draft_df = pd.DataFrame(st.session_state.draft_board)
                st.dataframe(draft_df)
                
                # Draft position chart
                if len(draft_df) > 0:
                    fig = px.scatter(
                        draft_df,
                        x='round',
                        y='pick',
                        color='team',
                        text='player',
                        title="Draft Board Visualization"
                    )
                    fig.update_traces(textposition="middle center")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No picks recorded yet")
        
        with col2:
            st.subheader("Draft Statistics")
            
            if st.session_state.draft_board:
                draft_df = pd.DataFrame(st.session_state.draft_board)
                
                # Position breakdown
                if 'player' in draft_df.columns:
                    drafted_players = df[df['player_name'].isin(draft_df['player'])]
                    if not drafted_players.empty:
                        pos_counts = drafted_players['position'].value_counts()
                        st.write("**Positions Drafted:**")
                        for pos, count in pos_counts.items():
                            st.write(f"• {pos}: {count}")
                
                st.write(f"**Total Picks:** {len(draft_df)}")
                st.write(f"**Current Round:** {st.session_state.draft_round}")
                st.write(f"**Next Pick:** {st.session_state.draft_pick}")
            
            # Reset draft board
            if st.button("Reset Draft Board", type="secondary"):
                st.session_state.draft_board = []
                st.session_state.my_team = []
                st.session_state.draft_round = 1
                st.session_state.draft_pick = 1
                st.rerun()

    with tab8:
        st.header("League Sync & Management")
        
        if 'league_sync_data' in st.session_state:
            sync_data = st.session_state.league_sync_data
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("League Information")
                st.write(f"**League Name:** {sync_data.get('league_name', 'N/A')}")
                st.write(f"**Season:** {sync_data.get('season', 'N/A')}")
                st.write(f"**Teams:** {sync_data.get('total_rosters', 'N/A')}")
                st.write(f"**Scoring:** {list(sync_data.get('scoring_settings', {}).keys())[:3]}")  # Show first few scoring rules
                
                # League standings if available
                if sync_data.get('rosters'):
                    league_sync = LeagueSyncManager(FantasyDataAPI())
                    standings = league_sync.get_league_standings(sync_data)
                    if not standings.empty:
                        st.subheader("League Standings")
                        st.dataframe(standings)
            
            with col2:
                st.subheader("League Analytics")
                
                if sync_data.get('rosters'):
                    league_sync = LeagueSyncManager(FantasyDataAPI())
                    league_trends = league_sync.analyze_league_trends(sync_data, df)
                    
                    if league_trends:
                        st.write("**League Trends:**")
                        st.write(f"• Total Rostered Players: {league_trends.get('total_rostered_players', 'N/A')}")
                        st.write(f"• Rookie Adoption Rate: {league_trends.get('rookie_adoption_rate', 0):.1%}")
                        
                        if league_trends.get('most_rostered_positions'):
                            st.write("**Position Distribution:**")
                            for pos, count in league_trends['most_rostered_positions'].items():
                                st.write(f"• {pos}: {count}")
                
                # Draft analysis if available
                if sync_data.get('drafts'):
                    st.subheader("Draft Analysis")
                    for draft in sync_data['drafts']:
                        if draft['picks']:
                            st.write(f"**Draft Status:** {draft['status']}")
                            st.write(f"**Total Picks:** {len(draft['picks'])}")
                            
                            # Show recent picks
                            if len(draft['picks']) > 0:
                                recent_picks = draft['picks'][-10:]  # Last 10 picks
                                st.write("**Recent Picks:**")
                                for pick in recent_picks:
                                    st.write(f"Round {pick.get('round', '?')}, Pick {pick.get('draft_slot', '?')}: {pick.get('metadata', {}).get('first_name', '')} {pick.get('metadata', {}).get('last_name', '')}")
        else:
            st.info("Connect a league above to see sync data")
            
            # API Setup Instructions
            st.subheader("🔗 API Setup Instructions")
            
            with st.expander("Sleeper API Setup (Free)", expanded=True):
                st.markdown("""
                **Sleeper Integration is Ready!**
                
                1. **Find Your League ID:**
                   - Go to your Sleeper league
                   - Look at the URL: `sleeper.app/leagues/YOUR_LEAGUE_ID`
                   - Copy the league ID number
                
                2. **Enter League ID:**
                   - Paste it in the sidebar under "League Sync"
                   - Click "Sync League"
                
                **Features Available:**
                - Live roster tracking
                - League standings
                - Draft pick analysis
                - Team composition analytics
                """)
            
            with st.expander("FantasyPros ADP Integration"):
                st.markdown("""
                **Option 1: FantasyPros API (Recommended)**
                
                1. Sign up at: https://www.fantasypros.com/api/
                2. Subscribe to a plan ($30-50/month)
                3. Get your API key
                4. Add to Streamlit secrets:
                ```
                # .streamlit/secrets.toml
                [fantasypros]
                api_key = "your_api_key_here"
                ```
                
                **Option 2: Manual CSV Upload**
                - Download ADP data from FantasyPros
                - Use CSV import feature in sidebar
                
                **Option 3: Web Scraping (Advanced)**
                - Implement with BeautifulSoup/Selenium
                - May violate terms of service
                - Not recommended for production use
                """)
            
            with st.expander("Underdog Fantasy Integration"):
                st.markdown("""
                **Challenge: No Public API**
                
                Underdog Fantasy doesn't provide a public API. Options:
                
                1. **Web Scraping (Advanced Users)**
                   - Use Selenium to scrape ADP data
                   - Requires technical implementation
                   - May be against terms of service
                
                2. **Manual Data Entry**
                   - Download/copy ADP data manually
                   - Upload via CSV import
                
                3. **Alternative Sources**
                   - Use FantasyPros consensus ADP
                   - Include Underdog in consensus calculations
                """)
            
            with st.expander("Other Data Sources"):
                st.markdown("""
                **Free Alternatives:**
                - **NFL.com**: Has fantasy data but limited API
                - **Pro Football Reference**: Great for historical stats
                - **Reddit APIs**: r/fantasyfootball consensus rankings
                - **GitHub Projects**: Community-maintained datasets
                
                **Paid Alternatives:**
                - **The Athletic**: Premium rankings and analysis
                - **4for4**: Advanced projections ($)
                - **Football Outsiders**: Analytics and DVOA data ($)
                """)

if __name__ == "__main__":
    main()
