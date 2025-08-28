# streamlit run app.py
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
from datetime import datetime


# ----------------------------
# Data Access Layer (Sleeper)
# ----------------------------
class FantasyDataAPI:
    def __init__(self) -> None:
        self.sleeper_base_url = "https://api.sleeper.app/v1"
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "FF Analyzer/1.0"})

    @st.cache_data(ttl=60 * 60)
    def get_sleeper_players(_self) -> Dict:
        """Fetch the full Sleeper NFL players map once per hour."""
        r = _self.session.get(f"{_self.sleeper_base_url}/players/nfl", timeout=30)
        r.raise_for_status()
        return r.json()

    def get_user(self, username_or_id: str) -> Optional[Dict]:
        try:
            r = self.session.get(f"{self.sleeper_base_url}/user/{username_or_id}", timeout=15)
            r.raise_for_status()
            return r.json()
        except Exception:
            return None

    def get_user_leagues(self, user_id: str, season: str, sport: str = "nfl") -> List[Dict]:
        try:
            r = self.session.get(f"{self.sleeper_base_url}/user/{user_id}/leagues/{sport}/{season}", timeout=20)
            r.raise_for_status()
            return r.json()
        except Exception:
            return []

    def get_league_drafts(self, league_id: str) -> List[Dict]:
        try:
            r = self.session.get(f"{self.sleeper_base_url}/league/{league_id}/drafts", timeout=20)
            r.raise_for_status()
            return r.json()
        except Exception:
            return []

    def get_draft_picks(self, draft_id: str) -> List[Dict]:
        try:
            r = self.session.get(f"{self.sleeper_base_url}/draft/{draft_id}/picks", timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception:
            return []

    def _players_dict_to_df(self, players_dict: Dict) -> pd.DataFrame:
        rows: List[Dict] = []
        for pid, info in (players_dict or {}).items():
            pos = info.get("position")
            team = info.get("team")
            if pos in {"QB", "RB", "WR", "TE", "K", "DEF"} and team:
                rows.append(
                    {
                        "player_id": str(pid),
                        "player_name": f"{info.get('first_name','')} {info.get('last_name','')}".strip(),
                        "position": pos,
                        "team": team,
                        "age": info.get("age"),
                        "years_exp": info.get("years_exp", 0),
                        "height": info.get("height"),
                        "weight": info.get("weight"),
                        "college": info.get("college"),
                        "injury_status": info.get("injury_status"),
                        "status": info.get("status", "Active"),
                    }
                )
        base = pd.DataFrame(rows)
        if not base.empty:
            base["name_key"] = base["player_name"].str.strip().str.lower()
        return base

    def compute_adp_from_drafts(self, draft_ids: List[str]) -> pd.DataFrame:
        """Aggregate pick numbers across drafts to compute ADP & count."""
        all_rows: List[Dict] = []
        for did in draft_ids:
            picks = self.get_draft_picks(did) or []
            for p in picks:
                pid = p.get("player_id") or (p.get("metadata") or {}).get("player_id")
                pick_no = p.get("pick_no") or p.get("overall") or p.get("pick")
                if pid and pick_no is not None:
                    try:
                        pick_val = float(pick_no)
                    except Exception:
                        continue
                    all_rows.append({"player_id": str(pid), "pick_no": pick_val})
        if not all_rows:
            return pd.DataFrame(columns=["player_id", "adp", "drafts_count", "adp_rank"])
        adp = (
            pd.DataFrame(all_rows)
            .groupby("player_id")
            .agg(adp=("pick_no", "mean"), drafts_count=("pick_no", "count"))
            .reset_index()
        )
        adp["adp_rank"] = adp["adp"].rank(method="dense").astype(int)
        return adp

    def _read_custom_rankings(self, file) -> pd.DataFrame:
        """
        Accepts a CSV with at least 'player_name' and 'rank' (or similar).
        Tolerates headers like name/player/full_name and rank/expert_rank.
        """
        df = pd.read_csv(file)
        # Normalize column names
        lower_cols = {c.lower(): c for c in df.columns}
        name_col = (
            lower_cols.get("player_name")
            or lower_cols.get("name")
            or lower_cols.get("player")
            or lower_cols.get("full_name")
        )
        rank_col = lower_cols.get("rank") or lower_cols.get("expert_rank") or lower_cols.get("overall")
        if not name_col or not rank_col:
            raise ValueError("CSV must include columns for player name and rank (e.g., 'player_name' and 'rank').")
        df = df.rename(columns={name_col: "player_name", rank_col: "custom_rank"})
        df = df[["player_name", "custom_rank"]].copy()
        df["name_key"] = df["player_name"].astype(str).str.strip().str.lower()
        # Ensure rank is numeric
        df["custom_rank"] = pd.to_numeric(df["custom_rank"], errors="coerce")
        df = df.dropna(subset=["custom_rank"])
        return df

    def get_enhanced_player_data(
        self,
        use_real_apis: bool = True,
        custom_rankings: Optional[pd.DataFrame] = None,
        sleeper_username_or_id: Optional[str] = None,
        season: Optional[str] = None,
        league_ids: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        if not use_real_apis:
            return pd.DataFrame()

        with st.spinner("Fetching data from Sleeper API..."):
            # Players
            players_dict = self.get_sleeper_players()
            base_df = self._players_dict_to_df(players_dict)

            # Season
            if season is None:
                try:
                    r = self.session.get(f"{self.sleeper_base_url}/state/nfl", timeout=10)
                    state = r.json()
                    season = str(state.get("league_season") or state.get("season") or datetime.now().year)
                except Exception:
                    season = str(datetime.now().year)

            # Draft IDs
            draft_ids: List[str] = []
            if league_ids:
                for lg in league_ids:
                    for d in self.get_league_drafts(lg) or []:
                        if d.get("draft_id"):
                            draft_ids.append(d.get("draft_id"))
            elif sleeper_username_or_id:
                user = self.get_user(sleeper_username_or_id)
                if user and user.get("user_id"):
                    leagues = self.get_user_leagues(user["user_id"], season) or []
                    for lg in leagues:
                        for d in self.get_league_drafts(lg.get("league_id")) or []:
                            if d.get("draft_id"):
                                draft_ids.append(d.get("draft_id"))

            adp_df = self.compute_adp_from_drafts(draft_ids) if draft_ids else pd.DataFrame(
                columns=["player_id", "adp", "adp_rank", "drafts_count"]
            )
            df = base_df.merge(adp_df, on="player_id", how="left")

            if "adp" in df.columns:
                df["sleeper_adp"] = df["adp"]

            # Merge expert ranks
            if custom_rankings is not None and not custom_rankings.empty:
                df = df.merge(
                    custom_rankings[["name_key", "custom_rank"]],
                    on="name_key",
                    how="left",
                )

            # Value delta (lower is better rank)
            if "custom_rank" in df.columns:
                df["adp_rank"] = df["adp_rank"].fillna(df["sleeper_adp"].rank(method="dense"))
                df["value_delta"] = (df["adp_rank"] - df["custom_rank"]).astype("Int64")

            return df


# ----------------------------
# Streamlit App
# ----------------------------
def _render_sidebar(api: FantasyDataAPI) -> Tuple[Optional[pd.DataFrame], str, List[str], Optional[str]]:
    st.sidebar.header("🧩 Data Sources")

    # Expert CSV
    uploaded_csv = st.sidebar.file_uploader("Upload expert rankings CSV", type=["csv"], key="csv_upload")
    custom_rankings_df: Optional[pd.DataFrame] = None
    if uploaded_csv is not None:
        try:
            custom_rankings_df = api._read_custom_rankings(uploaded_csv)
            st.sidebar.success(f"Loaded {len(custom_rankings_df)} expert rankings.")
        except Exception as e:
            st.sidebar.error(f"CSV parse error: {e}")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Sleeper (for ADP)")

    sleeper_username = st.sidebar.text_input(
        "Your Sleeper username (or user_id)",
        value="",
        help="Used to pull your leagues & drafts to compute ADP",
        key="sleeper_user",
    )
    season_input = st.sidebar.text_input("Season (blank = auto)", value="", key="season_input")
    league_ids_raw = st.sidebar.text_area(
        "Specific League IDs (optional)",
        help="One per line. If provided, ADP is computed from these leagues only.",
        key="league_ids_raw",
    )
    league_ids = [x.strip() for x in league_ids_raw.splitlines() if x.strip()]

    return custom_rankings_df, sleeper_username, league_ids, (season_input or None)


def _render_value_board(df: pd.DataFrame) -> None:
    st.markdown("## 🧮 Value vs ADP (Experts vs Your Sleeper Room)")
    if df is None or df.empty:
        st.info("Upload a CSV of expert rankings and load data from Sleeper to compute ADP.")
        return

    if "custom_rank" not in df.columns:
        st.info("Upload a CSV of expert rankings in the sidebar. Include 'player_name' and 'rank'.")
        return

    vdf = df.copy()
    if "adp_rank" not in vdf.columns or vdf["adp_rank"].isna().all():
        vdf["adp_rank"] = vdf["sleeper_adp"].rank(method="dense")

    vdf["value_delta"] = (vdf["adp_rank"] - vdf["custom_rank"]).astype("Int64")
    vdf["value_tag"] = np.where(
        vdf["value_delta"] >= 2,
        "✅ Value",
        np.where(vdf["value_delta"] <= -2, "⚠️ Overpriced", "—"),
    )

    cols = [
        "player_name",
        "position",
        "team",
        "custom_rank",
        "adp_rank",
        "sleeper_adp",
        "drafts_count",
        "value_delta",
        "value_tag",
    ]
    cols = [c for c in cols if c in vdf.columns]

    st.subheader("Top Value Targets (Experts higher than market)")
    st.dataframe(
        vdf.sort_values(["value_delta", "custom_rank"], ascending=[False, True])[cols].head(40),
        use_container_width=True,
        height=520,
    )

    st.subheader("Risk Check")
    if "drafts_count" in vdf.columns and vdf["drafts_count"].notna().any():
        max_dc = int(max(1, float(vdf["drafts_count"].max())))
        min_drafts = st.slider("Min drafts in sample", 1, max_dc, 1, key="min_drafts_slider")
        mask = (vdf["drafts_count"] >= min_drafts)
        st.dataframe(
            vdf[mask].sort_values(["value_delta", "custom_rank"], ascending=[False, True])[cols].head(40),
            use_container_width=True,
        )
    else:
        st.caption("No drafts_count available yet.")


def main() -> None:
    st.set_page_config(page_title="Fantasy Football Analyzer", layout="wide")
    st.title("🏈 Fantasy Football Analyzer — Sleeper ADP + Expert Ranks")

    api = FantasyDataAPI()
    custom_rankings_df, sleeper_username, league_ids, season = _render_sidebar(api)

    if st.sidebar.button("🔄 Load/Refresh Data", type="primary", key="sidebar_load_refresh"):
        with st.spinner("Loading player data & computing ADP..."):
            df = api.get_enhanced_player_data(
                use_real_apis=True,
                custom_rankings=custom_rankings_df,
                sleeper_username_or_id=sleeper_username or None,
                season=season,
                league_ids=league_ids or None,
            )
            st.session_state["enhanced_data"] = df
            st.success("Data loaded successfully!")

    df = st.session_state.get("enhanced_data")
    if isinstance(df, pd.DataFrame) and not df.empty:
        st.markdown("### 📊 Player Data (merged)")
        st.dataframe(df.head(200), use_container_width=True, height=420)

    _render_value_board(df if isinstance(df, pd.DataFrame) else pd.DataFrame())


if __name__ == "__main__":
    main()
