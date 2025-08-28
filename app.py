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
import warnings
import io
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Fantasy Football Draft Analyzer Pro",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded"
)

class FantasyDataAPI:
    """Real API handler with Sleeper support"""
    
    def __init__(self):
        self.sleeper_base_url = "https://api.sleeper.app/v1"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Fantasy-Draft-Analyzer/1.0',
            'Accept': 'application/json'
        })
    
    def get_sleeper_players(self) -> Dict:
        """Get all players from Sleeper API"""
        try:
            response = self.session.get(f"{self.sleeper_base_url}/players/nfl")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Error fetching Sleeper players: {str(e)}")
            return {}
    
    def get_sleeper_trending(self, sport: str = "nfl", lookback_hours: int = 24) -> Dict:
        """Get trending players from Sleeper API"""
        try:
            response = self.session.get(f"{self.sleeper_base_url}/players/{sport}/trending/add", 
                                      params={"lookback_hours": lookback_hours, "limit": 100})
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.warning(f"Error fetching trending data: {str(e)}")
            return {}
    
    def process_sleeper_data(self, players_dict: Dict, custom_rankings: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Process Sleeper API data into analyzable format"""
        player_list = []
        
        # Position mapping
        position_filter = ['QB', 'RB', 'WR', 'TE', 'K', 'DEF']
        
        for player_id, player_info in players_dict.items():
            if (player_info.get('position') in position_filter and 
                player_info.get('active', True) and
                player_info.get('team')):
                
                # Extract basic player info
                player_data = {
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
                    'status': player_info.get('status', 'Active')
                }
                
                # Add fantasy relevant stats if available
                if 'fantasy_data_id' in player_info:
                    player_data['fantasy_data_id'] = player_info['fantasy_data_id']
                
                # Add mock projections and analytics (in real app, you'd get these from another source)
                player_data.update(self._add_mock_analytics(player_data))
                
                player_list.append(player_data)
        
        df = pd.DataFrame(player_list)
        
        # Merge with custom rankings if provided
        if custom_rankings is not None:
            df = self._merge_custom_rankings(df, custom_rankings)
        
        return df
    
    def _add_mock_analytics(self, player_data: Dict) -> Dict:
        """Add mock analytics data (replace with real projections in production)"""
        np.random.seed(hash(player_data['player_name']) % 2147483647)
        
        position = player_data['position']
        
        # Position-based projections
        if position == 'QB':
            base_points = np.random.normal(250, 50)
            ceiling = base_points * 1.4
            floor = base_points * 0.7
        elif position == 'RB':
            base_points = np.random.normal(180, 40)
            ceiling = base_points * 1.5
            floor = base_points * 0.6
        elif position == 'WR':
            base_points = np.random.normal(160, 35)
            ceiling = base_points * 1.4
            floor = base_points * 0.6
        elif position == 'TE':
            base_points = np.random.normal(120, 30)
            ceiling = base_points * 1.3
            floor = base_points * 0.7
        else:
            base_points = np.random.normal(100, 20)
            ceiling = base_points * 1.2
            floor = base_points * 0.8
        
        return {
            'projected_points': max(0, round(base_points, 1)),
            'ceiling': max(0, round(ceiling, 1)),
            'floor': max(0, round(floor, 1)),
            'sleeper_adp': np.random.uniform(1, 200),
            'targets_projection': np.random.randint(20, 150) if position in ['WR', 'TE', 'RB'] else 0,
            'red_zone_touches': np.random.randint(5, 50) if position in ['RB', 'WR', 'TE'] else np.random.randint(15, 40),
            'snap_share': round(np.random.uniform(0.3, 0.9), 2),
            'injury_risk': round(np.random.uniform(0.1, 0.8), 2),
            'strength_of_schedule': round(np.random.uniform(0.4, 0.6), 2),
            'auction_value': np.random.randint(1, 70),
            'dynasty_value': np.random.randint(50, 95),
            'bye_week': np.random.randint(4, 14)
        }
    
    def _merge_custom_rankings(self, df: pd.DataFrame, custom_rankings: pd.DataFrame) -> pd.DataFrame:
        """Merge custom rankings with player data"""
        try:
            # Normalize column names in custom rankings
            custom_rankings.columns = custom_rankings.columns.str.lower().str.strip()
            
            # Try to find player name column
            name_cols = ['player_name', 'name', 'player', 'full_name']
            name_col = None
            for col in name_cols:
                if col in custom_rankings.columns:
                    name_col = col
                    break
            
            if name_col is None:
                st.warning("Could not find player name column in custom rankings. Expected columns: player_name, name, player, or full_name")
                return df
            
            # Add custom ranking prefix to avoid conflicts
            custom_rankings = custom_rankings.rename(columns={
                col: f"custom_{col}" if col != name_col else "player_name_custom" 
                for col in custom_rankings.columns
            })
            
            # Rename the name column back
            custom_rankings = custom_rankings.rename(columns={"player_name_custom": "player_name"})
            
            # Merge on player name (fuzzy matching could be added here)
            merged_df = df.merge(custom_rankings, on='player_name', how='left', suffixes=('', '_custom'))
            
            st.success(f"Successfully merged custom rankings for {merged_df['custom_rank'].notna().sum() if 'custom_rank' in merged_df.columns else 0} players")
            
            return merged_df
            
        except Exception as e:
            st.error(f"Error merging custom rankings: {str(e)}")
            return df
    
    def get_enhanced_player_data(self, use_real_apis: bool = True, custom_rankings: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Get comprehensive player data"""
        if use_real_apis:
            with st.spinner("Fetching data from Sleeper API..."):
                players_dict = self.get_sleeper_players()
                if players_dict:
                    return self.process_sleeper_data(players_dict, custom_rankings)
                else:
                    st.warning("Failed to fetch real data, using mock data instead")
                    return self._get_mock_enhanced_data(custom_rankings)
        else:
            return self._get_mock_enhanced_data(custom_rankings)
    
    def _get_mock_enhanced_data(self, custom_rankings: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Enhanced mock data with all analytics features"""
        np.random.seed(42)
        
        players_data = [
            # RBs
            {"player_id": "1", "player_name": "Christian McCaffrey", "position": "RB", "team": "SF", "age": 27, "sleeper_adp": 1.2, "underdog_adp": 1.1, 
             "targets_projection": 85, "red_zone_touches": 45, "snap_share": 0.78, "injury_risk": 0.6, "ceiling": 350, "floor": 180,
             "strength_of_schedule": 0.52, "projected_points": 280, "auction_value": 65, "dynasty_value": 85, "years_exp": 6},
            {"player_id": "2", "player_name": "Saquon Barkley", "position": "RB", "team": "PHI", "age": 26, "sleeper_adp": 28.8, "underdog_adp": 29.4,
             "targets_projection": 65, "red_zone_touches": 38, "snap_share": 0.71, "injury_risk": 0.7, "ceiling": 320, "floor": 160,
             "strength_of_schedule": 0.48, "projected_points": 250, "auction_value": 45, "dynasty_value": 82, "years_exp": 5},
            {"player_id": "3", "player_name": "Derrick Henry", "position": "RB", "team": "BAL", "age": 30, "sleeper_adp": 15.7, "underdog_adp": 16.3,
             "targets_projection": 25, "red_zone_touches": 42, "snap_share": 0.65, "injury_risk": 0.3, "ceiling": 290, "floor": 170,
             "strength_of_schedule": 0.55, "projected_points": 240, "auction_value": 52, "dynasty_value": 65, "years_exp": 8},
            {"player_id": "4", "player_name": "Josh Allen", "position": "QB", "team": "BUF", "age": 28, "sleeper_adp": 8.5, "underdog_adp": 9.2,
             "targets_projection": 0, "red_zone_touches": 35, "snap_share": 1.0, "injury_risk": 0.4, "ceiling": 380, "floor": 220,
             "strength_of_schedule": 0.50, "projected_points": 320, "auction_value": 58, "dynasty_value": 88, "years_exp": 6},
            {"player_id": "5", "player_name": "Cooper Kupp", "position": "WR", "team": "LAR", "age": 31, "sleeper_adp": 12.3, "underdog_adp": 11.8,
             "targets_projection": 145, "red_zone_touches": 25, "snap_share": 0.82, "injury_risk": 0.6, "ceiling": 310, "floor": 160,
             "strength_of_schedule": 0.51, "projected_points": 265, "auction_value": 55, "dynasty_value": 72, "years_exp": 7},
            {"player_id": "6", "player_name": "Travis Kelce", "position": "TE", "team": "KC", "age": 34, "sleeper_adp": 18.9, "underdog_adp": 19.5,
             "targets_projection": 125, "red_zone_touches": 28, "snap_share": 0.75, "injury_risk": 0.4, "ceiling": 280, "floor": 140,
             "strength_of_schedule": 0.48, "projected_points": 230, "auction_value": 50, "dynasty_value": 65, "years_exp": 11},
        ]
        
        df = pd.DataFrame(players_data)
        
        # Add bye weeks
        bye_weeks = {"SF": 9, "PHI": 5, "BAL": 14, "BUF": 11, "LAR": 6, "KC": 8}
        df['bye_week'] = df['team'].map(bye_weeks)
        
        # Merge with custom rankings if provided
        if custom_rankings is not None:
            df = self._merge_custom_rankings(df, custom_rankings)
        
        return df

def load_custom_rankings() -> Optional[pd.DataFrame]:
    """Handle CSV upload for custom player rankings"""
    st.sidebar.markdown("---")
    st.sidebar.header("📊 Custom Rankings")
    
    uploaded_file = st.sidebar.file_uploader(
        "Upload your custom player rankings (CSV)", 
        type=['csv'],
        help="CSV should include columns: player_name, rank, and any other custom metrics"
    )
    
    if uploaded_file is not None:
        try:
            # Read the CSV
            custom_df = pd.read_csv(uploaded_file)
            
            # Display preview
            st.sidebar.success(f"✅ Loaded {len(custom_df)} custom rankings")
            
            with st.sidebar.expander("Preview Custom Rankings"):
                st.dataframe(custom_df.head(10))
            
            return custom_df
            
        except Exception as e:
            st.sidebar.error(f"Error loading CSV: {str(e)}")
            return None
    
    return None

def display_player_comparison(df: pd.DataFrame):
    """Enhanced player comparison with custom rankings"""
    st.header("🔍 Player Comparison Tool")
    
    col1, col2 = st.columns(2)
    
    with col1:
        players_to_compare = st.multiselect(
            "Select players to compare:",
            options=df['player_name'].tolist(),
            default=df['player_name'].head(3).tolist() if len(df) >= 3 else df['player_name'].tolist()
        )
    
    with col2:
        comparison_metric = st.selectbox(
            "Primary comparison metric:",
            options=['projected_points', 'sleeper_adp', 'auction_value', 'dynasty_value', 'ceiling', 'floor']
        )
    
    if players_to_compare:
        comparison_df = df[df['player_name'].isin(players_to_compare)].copy()
        
        # Create comparison chart
        fig = px.bar(
            comparison_df, 
            x='player_name', 
            y=comparison_metric,
            color='position',
            title=f"Player Comparison: {comparison_metric.replace('_', ' ').title()}"
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Show detailed comparison table
        comparison_cols = ['player_name', 'position', 'team', 'projected_points', 'sleeper_adp', 'auction_value']
        
        # Add custom ranking columns if they exist
        custom_cols = [col for col in comparison_df.columns if col.startswith('custom_')]
        if custom_cols:
            comparison_cols.extend(custom_cols)
        
        st.dataframe(comparison_df[comparison_cols], use_container_width=True)

def main():
    st.title("🏈 Fantasy Football Draft Analyzer Pro")
    st.markdown("*Advanced analytics with Sleeper API integration and custom rankings*")
    
    # Initialize API
    api = FantasyDataAPI()
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    use_real_apis = st.sidebar.checkbox(
        "Use Sleeper API", 
        value=True, 
        help="Uncheck to use mock data for testing"
    )
    
    # Custom rankings upload
    custom_rankings = load_custom_rankings()
    
    # Data loading section
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Load/Refresh Data", type="primary"):
        with st.spinner("Loading player data..."):
            st.session_state.enhanced_data = api.get_enhanced_player_data(use_real_apis, custom_rankings)
            st.success("Data loaded successfully!")
    
    # Auto-load data on first run
    if 'enhanced_data' not in st.session_state:
        with st.spinner("Loading initial data..."):
            st.session_state.enhanced_data = api.get_enhanced_player_data(use_real_apis, custom_rankings)
    
    if 'enhanced_data' not in st.session_state or st.session_state.enhanced_data.empty:
        st.error("No data available. Please check your connection and try loading data again.")
        return
    
    df = st.session_state.enhanced_data
    
    # Main metrics display
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Players", len(df))
    with col2:
        st.metric("Positions", df['position'].nunique())
    with col3:
        st.metric("Teams", df['team'].nunique())
    with col4:
        custom_count = len([col for col in df.columns if col.startswith('custom_')])
        st.metric("Custom Metrics", custom_count)
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Player Data", 
        "🎯 Analysis", 
        "🚀 Draft Tools", 
        "📈 Charts",
        "💰 Auction",
        "🔄 Trades",
        "📋 My Team"
    ])
    
    with tab1:
        st.header("📊 Player Database")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            position_filter = st.multiselect("Filter by Position:", df['position'].unique(), default=df['position'].unique())
        with col2:
            team_filter = st.multiselect("Filter by Team:", sorted(df['team'].unique()), default=sorted(df['team'].unique()))
        with col3:
            min_points = st.slider("Minimum Projected Points:", 0, int(df['projected_points'].max()), 0)
        
        # Apply filters
        filtered_df = df[
            (df['position'].isin(position_filter)) &
            (df['team'].isin(team_filter)) &
            (df['projected_points'] >= min_points)
        ]
        
        # Display columns selection
        base_cols = ['player_name', 'position', 'team', 'age', 'projected_points', 'sleeper_adp', 'auction_value']
        custom_cols = [col for col in filtered_df.columns if col.startswith('custom_')]
        all_display_cols = base_cols + custom_cols
        
        display_cols = st.multiselect(
            "Select columns to display:",
            options=filtered_df.columns.tolist(),
            default=all_display_cols
        )
        
        if display_cols:
            st.dataframe(filtered_df[display_cols].sort_values('projected_points', ascending=False), 
                        use_container_width=True, height=400)
    
    with tab2:
        st.header("🎯 Advanced Analysis")
        
        # Player comparison tool
        display_player_comparison(df)
        
        st.markdown("---")
        
        # Position analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # Positional breakdown
            pos_counts = df['position'].value_counts()
            fig = px.pie(values=pos_counts.values, names=pos_counts.index, 
                        title="Players by Position")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Top performers by position
            top_n = st.slider("Show top N players per position:", 1, 10, 5)
            
            top_players = []
            for position in df['position'].unique():
                pos_players = df[df['position'] == position].nlargest(top_n, 'projected_points')
                top_players.append(pos_players)
            
            top_df = pd.concat(top_players)
            
            fig = px.bar(top_df, x='player_name', y='projected_points', color='position',
                        title=f"Top {top_n} Players by Position (Projected Points)")
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("🚀 Draft Tools")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_player = st.selectbox("🎯 Analyze a specific player:", 
                                         options=[""] + df['player_name'].tolist())
            
            if selected_player:
                player_data = df[df['player_name'] == selected_player].iloc[0]
                
                # Player info card
                st.markdown(f"""
                ### {selected_player}
                **Position:** {player_data['position']} | **Team:** {player_data['team']} | **Age:** {player_data['age']}
                """)
                
                # Key metrics
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                
                with metric_col1:
                    st.metric("Projected Points", f"{player_data['projected_points']:.1f}")
                with metric_col2:
                    st.metric("Sleeper ADP", f"{player_data['sleeper_adp']:.1f}")
                with metric_col3:
                    st.metric("Auction Value", f"${player_data['auction_value']}")
                with metric_col4:
                    st.metric("Dynasty Value", f"{player_data['dynasty_value']}")
                
                # Risk/Reward analysis
                st.subheader("Risk/Reward Profile")
                risk_col1, risk_col2 = st.columns(2)
                
                with risk_col1:
                    st.metric("Ceiling", f"{player_data['ceiling']:.1f}")
                    st.metric("Injury Risk", f"{player_data['injury_risk']:.2f}")
                    
                with risk_col2:
                    st.metric("Floor", f"{player_data['floor']:.1f}")
                    st.metric("Snap Share", f"{player_data['snap_share']:.1%}")
                
                # Show custom rankings if available
                custom_cols = [col for col in player_data.index if col.startswith('custom_')]
                if custom_cols:
                    st.subheader("Custom Rankings")
                    custom_data = {}
                    for col in custom_cols:
                        if pd.notna(player_data[col]):
                            clean_name = col.replace('custom_', '').replace('_', ' ').title()
                            custom_data[clean_name] = player_data[col]
                    
                    if custom_data:
                        custom_df_display = pd.DataFrame(list(custom_data.items()), 
                                                       columns=['Metric', 'Value'])
                        st.table(custom_df_display)
        
        with col2:
            st.subheader("Quick Actions")
            if st.button("📋 Add to Watch List", disabled=not selected_player):
                st.success(f"Added {selected_player} to watch list!")
            
            if st.button("⭐ Mark as Target", disabled=not selected_player):
                st.success(f"Marked {selected_player} as draft target!")
    
    with tab4:
        st.header("📈 Advanced Charts")
        
        # Chart selection
        chart_type = st.selectbox("Select chart type:", [
            "ADP vs Projections Scatter",
            "Value vs ADP Analysis", 
            "Position Strength Comparison",
            "Age vs Production Analysis",
            "Custom Ranking Comparison"
        ])
        
        if chart_type == "ADP vs Projections Scatter":
            fig = px.scatter(df, x='sleeper_adp', y='projected_points', 
                           color='position', size='auction_value',
                           hover_data=['player_name', 'team'],
                           title="ADP vs Projected Points (Size = Auction Value)")
            fig.update_layout(xaxis_title="Sleeper ADP", yaxis_title="Projected Points")
            st.plotly_chart(fig, use_container_width=True)
        
        elif chart_type == "Custom Ranking Comparison":
            custom_cols = [col for col in df.columns if col.startswith('custom_') and df[col].notna().sum() > 0]
            
            if custom_cols:
                selected_custom_col = st.selectbox("Select custom metric:", custom_cols)
                custom_df = df[df[selected_custom_col].notna()].copy()
                
                fig = px.scatter(custom_df, x=selected_custom_col, y='projected_points',
                               color='position', text='player_name',
                               title=f"Custom {selected_custom_col.replace('custom_', '').title()} vs Projected Points")
                fig.update_traces(textposition="top center")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Upload custom rankings to see this chart.")
    
    with tab5:
        st.header("💰 Auction Strategy")
        
        budget = st.slider("Total Auction Budget:", 100, 300, 200)
        roster_spots = st.slider("Total Roster Spots:", 12, 20, 16)
        
        # Calculate average per player
        avg_per_player = budget / roster_spots
        
        st.metric("Average $ per Player", f"${avg_per_player:.1f}")
        
        # Value analysis
        df_auction = df.copy()
        df_auction['value_score'] = df_auction['projected_points'] / df_auction['auction_value']
        df_auction = df_auction.sort_values('value_score', ascending=False)
        
        st.subheader("Best Values (Points per Dollar)")
        st.dataframe(df_auction[['player_name', 'position', 'team', 'projected_points', 
                                'auction_value', 'value_score']].head(20),
                    use_container_width=True)
    
    with tab6:
        st.header("🔄 Trade Analyzer")
        st.info("Trade analysis functionality coming soon! This will include:")
        st.markdown("""
        - **Trade value calculator**
        - **Roster need analysis** 
        - **Fair trade suggestions**
        - **Dynasty vs Redraft considerations**
        """)
    
    with tab7:
        st.header("📋 My Team Manager")
        st.info("Team management functionality coming soon! This will include:")
        st.markdown("""
        - **Draft tracker**
        - **Roster construction**
        - **Bye week planning**
        - **Waiver wire targets**
        """)

if __name__ == "__main__":
    main()
