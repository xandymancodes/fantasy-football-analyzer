# streamlit run app.py
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st


# ============================
# Sleeper API Data Access
# ============================
class FantasyDataAPI:
    def __init__(self) -> None:
        self.sleeper_base_url = "https://api.sleeper.app/v1"
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "FF Analyzer/1.0"})

    # Cache the heavy players map for an hour
    @st.cache_data(ttl=60 * 60)
    def get_sleeper_players(_self) -> Dict:
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

    # ---------- Helpers ----------
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
        return base

    @staticmethod
    def _clean_name_series(s: pd.Series) -> pd.Series:
        """Normalize names: trim, handle 'Last, First', drop suffixes and periods."""
        s = s.astype(str).str.strip()
        # Handle "Last, First" -> "First Last"
        last_first_mask = s.str.contains(",", regex=False)
        def flip_name(x: str) -> str:
            parts = [p.strip() for p in x.split(",", 1)]
            return f"{parts[1]} {parts[0]}" if len(parts) == 2 else x
        s.loc[last_first_mask] = s.loc[last_first_mask].apply(flip_name)
        # Remove dots and suffixes, squeeze spaces
        s = (
            s.str.replace(r"\.", "", regex=True)
             .str.replace(r"\s+(jr|sr|iii|ii|iv)$", "", regex=True, flags=0)
             .str.replace(r"\s+", " ", regex=True)
             .str.strip()
        )
        return s

    def compute_adp_from_drafts(self, draft_ids: List[str]) -> pd.DataFrame:
        """Aggregate pick numbers across drafts to compute ADP & sample size."""
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

    # Robust merge of uploaded rankings (many header variants supported)
    def merge_custom_rankings(self, df: pd.DataFrame, uploaded_df: pd.DataFrame) -> pd.DataFrame:
        cr = uploaded_df.copy()
        # Normalize headers to snake_case
        cr.columns = cr.columns.str.lower().str.strip().str.replace(r"\s+", "_", regex=True)

        # Find name column
        name_candidates = ["player_name", "name", "player", "full_name", "player_full_name"]
        name_col = next((c for c in name_candidates if c in cr.columns), None)
        if name_col is None:
            st.warning("Could not find a player name column in the uploaded CSV. "
                       "Please include one of: Player, Name, Player_Name, Full Name.")
            return df

        # Map likely rank columns → custom_rank
        rank_map = {
            "rank": "custom_rank",
            "overall": "custom_rank",
            "overall_rank": "custom_rank",
            "ecr": "custom_rank",
        }
        for src, dst in rank_map.items():
            if src in cr.columns and "custom_rank" not in cr.columns:
                cr = cr.rename(columns={src: dst})

        # Coerce custom_rank to numeric if present
        if "custom_rank" in cr.columns:
            cr["custom_rank"] = pd.to_numeric(cr["custom_rank"], errors="coerce")

        # Clean names in both frames
        df = df.copy()
        df["player_name"] = self._clean_name_series(df["player_name"])
        cr[name_col] = self._clean_name_series(cr[name_col])

        # Prefix other uploaded columns with custom_
        renamed = {}
        for col in cr.columns:
            if col in {name_col, "custom_rank"}:
                continue
            renamed[col] = f"custom_{col}"
        if renamed:
            cr = cr.rename(columns=renamed)

        # Align name column name for merge
        if name_col != "player_name":
            cr = cr.rename(columns={name_col: "player_name"})

        merged = df.merge(cr, on="player_name", how="left", suffixes=("", "_dup"))
        matched = int(merged["custom_rank"].notna().sum()) if "custom_rank" in merged.columns else 0
        st.caption(f"🔗 Merged custom rankings for {matched} players")
        return merged

    # Main data method
    def get_enhanced_player_data(
        self,
        custom_rankings: Optional[pd.DataFrame],
        sleeper_username_or_id: Optional[str],
        season: Optional[str],
        league_ids: Optional[List[str]],
    ) -> Tuple[pd.DataFrame, Dict]:
        with st.spinner("Fetching data from Sleeper API..."):
            # Players
            players_dict = self.get_sleeper_players()
            base_df = self._players_dict_to_df(players_dict)

            # Season (auto if blank)
            if not season:
                try:
                    r = self.session.get(f"{self.sleeper_base_url}/state/nfl", timeout=10)
                    state = r.json()
                    season = str(state.get("league_season") or state.get("season") or datetime.now().year)
                except Exception:
                    season = str(datetime.now().year)

            # Draft IDs for ADP
            draft_ids: List[str] = []
            leagues_found = 0
            if league_ids:
                for lg in league_ids:
                    for d in self.get_league_drafts(lg) or []:
                        if d.get("draft_id"):
                            draft_ids.append(d.get("draft_id"))
            elif sleeper_username_or_id:
                user = self.get_user(sleeper_username_or_id)
                if user and user.get("user_id"):
                    leagues = self.get_user_leagues(user["user_id"], season) or []
                    leagues_found = len(leagues)
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

            # Merge expert ranks if provided
            if custom_rankings is not None and not custom_rankings.empty:
                df = self.merge_custom_rankings(df, custom_rankings)

            # Compute value delta when we have ranks
            if "custom_rank" in df.columns:
                # If adp_rank missing, derive from sleeper_adp
                if "adp_rank" not in df.columns or df["adp_rank"].isna().all():
                    df["adp_rank"] = df["sleeper_adp"].rank(method="dense")
                df["value_delta"] = (df["adp_rank"] - df["custom_rank"]).astype("Int64")

            # Diagnostics for user
            picks_sample = 0
            for did in draft_ids[:5]:
                pk = self.get_draft_picks(did) or []
                picks_sample += len(pk)

            diag = {
                "season": season,
                "draft_ids_found": len(draft_ids),
                "leagues_found": leagues_found,
                "sample_picks_in_first_few_drafts": picks_sample,
            }
            return df, diag


# ============================
# CSV Loader (robust)
# ============================
def load_custom_rankings() -> Optional[pd.DataFrame]:
    """Handle CSV upload for custom player rankings with robust parsing."""
    st.sidebar.header("📊 Custom Rankings")
    uploaded_file = st.sidebar.file_uploader(
        "Upload your custom player rankings (CSV)",
        type=["csv"],
        help=("CSV can use commas, semicolons, or tabs. "
              "Include a player name column (e.g., Player, Name, Full_Name) "
              "and a ranking column (e.g., Rank, Overall, ECR)."),
        key="csv_upload",
    )
    if uploaded_file is None:
        return None

    try:
        uploaded_file.seek(0)
        try:
            custom_df = pd.read_csv(uploaded_file)
        except Exception:
            uploaded_file.seek(0)
            custom_df = pd.read_csv(uploaded_file, engine="python", sep=None, on_bad_lines="skip")

        if custom_df.empty:
            st.sidebar.error("The uploaded CSV appears to be empty.")
            return None

        # Normalize headers
        custom_df.columns = custom_df.columns.str.lower().str.strip().str.replace(r"\s+", "_", regex=True)

        # Coerce any common rank-like columns to numeric to aid later mapping
        for c in ["rank", "overall", "overall_rank", "ecr"]:
            if c in custom_df.columns:
                custom_df[c] = pd.to_numeric(custom_df[c], errors="coerce")

        st.sidebar.success(f"Loaded {len(custom_df)} rows from rankings CSV")
        with st.sidebar.expander("Preview Custom Rankings"):
            st.dataframe(custom_df.head(15), use_container_width=True)

        return custom_df

    except Exception as e:
        st.sidebar.error(f"Error loading CSV: {e}")
        return None


# ============================
# UI Helpers
# ============================
def render_sidebar_inputs() -> Tuple[str, Optional[str], List[str]]:
    st.sidebar.header("🧩 Sleeper (for ADP)")

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
    return sleeper_username, (season_input or None), league_ids


def render_value_board(df: pd.DataFrame) -> None:
    st.markdown("## 🧮 Value vs ADP (Experts vs Your Sleeper Room)")
    if df is None or df.empty:
        st.info("Upload a CSV of expert rankings and load data from Sleeper to compute ADP.")
        return

    if "custom_rank" not in df.columns:
        st.info("Upload a CSV of expert rankings in the sidebar. Include 'player_name' and 'rank' (or Overall/ECR).")
        return

    vdf = df.copy()
    if "adp_rank" not in vdf.columns or vdf["adp_rank"].isna().all():
        if "sleeper_adp" in vdf.columns:
            vdf["adp_rank"] = vdf["sleeper_adp"].rank(method="dense")
        else:
            st.warning("No ADP or ranks present to compute value deltas.")
            return

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
        mask = vdf["drafts_count"] >= min_drafts
        st.dataframe(
            vdf[mask].sort_values(["value_delta", "custom_rank"], ascending=[False, True])[cols].head(40),
            use_container_width=True,
        )
    else:
        st.caption("No drafts_count available yet.")


# ============================
# Main App
# ============================
def main() -> None:
    st.set_page_config(page_title="Fantasy Football Analyzer", layout="wide")
    st.title("🏈 Fantasy Football Analyzer — Sleeper ADP + Expert Ranks")

    api = FantasyDataAPI()

    # Sidebar sources
    custom_rankings_df = load_custom_rankings()
    sleeper_username, season, league_ids = render_sidebar_inputs()

    # Load/Refresh button
    if st.sidebar.button("🔄 Load/Refresh Data", type="primary", key="sidebar_load_refresh"):
        df, diag = api.get_enhanced_player_data(
            custom_rankings=custom_rankings_df,
            sleeper_username_or_id=sleeper_username or None,
            season=season,
            league_ids=league_ids or None,
        )
        st.session_state["enhanced_data"] = df
        st.session_state["diag"] = diag
        st.success("Data loaded successfully!")

    # Data preview
    df = st.session_state.get("enhanced_data")
    if isinstance(df, pd.DataFrame) and not df.empty:
        st.markdown("### 📊 Player Data (merged)")
        st.dataframe(df.head(200), use_container_width=True, height=420)
    else:
        st.info("No merged player data yet. Upload a rankings CSV and load Sleeper data.")

    # Diagnostics expander to explain missing ADP
    diag = st.session_state.get("diag", {})
    with st.expander("Sleeper diagnostics"):
        st.write(diag or {"note": "Click 'Load/Refresh Data' after entering your Sleeper info."})

    # Value board
    render_value_board(df if isinstance(df, pd.DataFrame) else pd.DataFrame())


if __name__ == "__main__":
    main()
