import os
import re
import io
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st

# ------------------------------------------------------------------------------------
# Streamlit setup
# ------------------------------------------------------------------------------------

st.set_page_config(
    page_title="Fantasy Football Analyzer (Sleeper + Expert CSV)",
    page_icon="🏈",
    layout="wide",
)

# ------------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------------

def _norm_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    n = name.lower()
    # Remove punctuation, periods, apostrophes, hyphens
    n = re.sub(r"[^\w\s]", " ", n)
    # Collapse whitespace
    n = re.sub(r"\s+", " ", n).strip()
    # Common suffixes
    n = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", n).strip()
    return n

def _coalesce(d: dict, *keys):
    for k in keys:
        if k in d and pd.notna(d[k]):
            return d[k]
    return None

def _find_col(cols: List[str], candidates: List[str]) -> Optional[str]:
    # Return first matching column name from candidates (case-insensitive, fuzzy)
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    # fallback fuzzy contains
    for c in cols:
        for cand in candidates:
            if cand.lower() in c.lower():
                return c
    return None

# ------------------------------------------------------------------------------------
# Sleeper API Client with resilient endpoints + caching
# ------------------------------------------------------------------------------------

@st.cache_data(show_spinner=False, ttl=3600)
def _http_get_json(url: str, params: Optional[dict] = None, headers: Optional[dict] = None):
    try:
        r = requests.get(url, params=params or {}, headers=headers or {}, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.debug(f"GET failed: {url} -> {e}")
        return None

class SleeperClient:
    def __init__(self, season: int, scoring_type: str = "redraft_ppr"):
        self.season = season
        self.scoring_type = scoring_type

    # ---------- Players ----------
    @st.cache_data(show_spinner=False, ttl=86400)
    def get_players(_self) -> Dict[str, dict]:
        # Try both app/com domains (Sleeper has both in the wild)
        data = _http_get_json("https://api.sleeper.com/players/nfl")
        if not data:
            data = _http_get_json("https://api.sleeper.app/v1/players/nfl")
        return data or {}

    # ---------- ADP (sitewide) ----------
    @st.cache_data(show_spinner=False, ttl=3600)
    def get_adp(_self) -> pd.DataFrame:
        # Prefer .com aggregated ADP; fall back to legacy .app if available
        season = _self.season
        adp_df = None

        # Newer aggregate ADP endpoint (observed in the wild)
        # Example: https://api.sleeper.com/adp/nfl/2025?season_type=regular&type=redraft_ppr
        for base in ["https://api.sleeper.com", "https://api.sleeper.app/v1"]:
            url = f"{base}/adp/nfl/{season}"
            params = {"season_type": "regular", "type": _self.scoring_type}
            data = _http_get_json(url, params=params)
            if isinstance(data, list) and len(data):
                try:
                    adp_df = pd.DataFrame(data)
                    break
                except Exception:
                    pass

        # Older pattern (very old) kept as last resort
        if adp_df is None:
            legacy = _http_get_json(f"https://api.sleeper.app/v1/players/nfl/adp")
            if isinstance(legacy, dict) and legacy:
                # legacy may be keyed by player_id -> {adp, rank, ...}
                rows = []
                for pid, row in legacy.items():
                    if isinstance(row, dict):
                        rows.append({"player_id": pid, **row})
                if rows:
                    adp_df = pd.DataFrame(rows)

        if adp_df is None:
            # Return empty frame with expected columns
            return pd.DataFrame(columns=["player_id", "adp", "rank", "name", "position", "team"])

        # Normalize common columns
        # Some returns: {"player_id","adp","name","position","team","count","pos_adp"}
        rename = {}
        for c in adp_df.columns:
            lc = c.lower()
            if lc == "player":
                rename[c] = "player_id"
            if lc == "averagedraftposition" or lc == "adp_overall":
                rename[c] = "adp"
        if rename:
            adp_df = adp_df.rename(columns=rename)

        # Only keep the essentials if available
        keep = [c for c in ["player_id", "adp", "rank", "name", "position", "team", "count", "pos_adp"] if c in adp_df.columns]
        adp_df = adp_df[keep].copy()
        # Ensure types
        if "adp" in adp_df.columns:
            adp_df["adp"] = pd.to_numeric(adp_df["adp"], errors="coerce")
        if "rank" in adp_df.columns:
            adp_df["rank"] = pd.to_numeric(adp_df["rank"], errors="coerce")
        return adp_df

    # ---------- User + Leagues ----------
    @st.cache_data(show_spinner=False, ttl=3600)
    def get_user_id(_self, username: str) -> Optional[str]:
        # Accept both username and numeric id; try lookup only when needed
        if not username:
            return None
        # username to user_id
        for base in ["https://api.sleeper.com", "https://api.sleeper.app/v1"]:
            data = _http_get_json(f"{base}/user/{username}")
            if isinstance(data, dict) and data.get("user_id"):
                return data["user_id"]
        return None

    @st.cache_data(show_spinner=False, ttl=600)
    def get_user_leagues(_self, user_id: str) -> List[dict]:
        leagues = []
        for base in ["https://api.sleeper.com", "https://api.sleeper.app/v1"]:
            data = _http_get_json(f"{base}/user/{user_id}/leagues/nfl/{_self.season}")
            if isinstance(data, list) and data:
                leagues = data
                break
        return leagues or []

    # ---------- Enriched Player Table ----------
    def build_enhanced_table(
        self,
        expert_df: Optional[pd.DataFrame],
        players: Dict[str, dict],
        adp_df: pd.DataFrame,
        keep_positions: Optional[List[str]] = None,
    ) -> pd.DataFrame:

        # Base player table
        rows = []
        for pid, p in (players or {}).items():
            # Filter only active-ish players (optional — keep if status missing)
            pos = p.get("position") or p.get("fantasy_positions", [None])[0]
            if keep_positions and pos and pos not in keep_positions:
                continue
            rows.append(
                {
                    "player_id": pid,
                    "player_name": _coalesce(p, "full_name", "first_name", "last_name", "last_name"),
                    "position": pos,
                    "team": p.get("team"),
                }
            )
        base_df = pd.DataFrame(rows)
        if base_df.empty:
            return pd.DataFrame(columns=[
                "player_id","player_name","position","team",
                "adp","adp_rank","ecr","value_diff"
            ])

        # Join ADP
        adp_slim = adp_df.copy()
        if "player_id" not in adp_slim.columns:
            # Try to join on name as fallback
            # Normalize for join
            adp_slim["__norm_name"] = adp_slim.get("name", "").map(_norm_name) if "name" in adp_slim.columns else ""
            base_df["__norm_name"] = base_df["player_name"].map(_norm_name)
            base_df = base_df.merge(
                adp_slim[[c for c in adp_slim.columns if c in ["__norm_name", "adp", "rank", "pos_adp"]]],
                on="__norm_name",
                how="left",
            )
        else:
            adp_slim = adp_slim[
                [c for c in adp_slim.columns if c in ["player_id", "adp", "rank", "pos_adp"]]
            ].copy()
            base_df = base_df.merge(adp_slim, on="player_id", how="left")

        # Derive adp_rank if not provided
        if "rank" in base_df.columns:
            base_df["adp_rank"] = pd.to_numeric(base_df["rank"], errors="coerce")
        else:
            # Rank by ADP ascending
            base_df["adp_rank"] = base_df["adp"].rank(method="average")

        # Attach Expert Rankings
        if expert_df is not None and not expert_df.empty:
            e = expert_df.copy()
            # Identify columns
            name_col = _find_col(e.columns.tolist(), ["player", "player_name", "name"])
            rank_col = _find_col(e.columns.tolist(), ["ecr", "rank", "overall", "overall_rank", "consensus_rank"])
            pos_col = _find_col(e.columns.tolist(), ["position", "pos"])
            team_col = _find_col(e.columns.tolist(), ["team", "nfl_team"])

            if not name_col or not rank_col:
                st.warning(
                    "Could not find 'Name' and 'Rank' columns in your CSV. "
                    "Please include at least 'Name' and 'Rank' (or ECR/Overall)."
                )
                # Return without ECR attached
            else:
                e = e.rename(
                    columns={name_col: "csv_name", rank_col: "ecr", **({pos_col: "csv_pos"} if pos_col else {}), **({team_col: "csv_team"} if team_col else {})}
                )
                # Normalize for matching
                e["__norm_name"] = e["csv_name"].map(_norm_name)

                # Try exact-name match first
                base_df["__norm_name"] = base_df["player_name"].map(_norm_name)
                base_df = base_df.merge(
                    e[["__norm_name", "ecr", *(["csv_pos"] if "csv_pos" in e.columns else []), *(["csv_team"] if "csv_team" in e.columns else [])]],
                    on="__norm_name",
                    how="left",
                )

                # If still a bunch of NaNs in ecr, try position-aided fallback matching
                if base_df["ecr"].isna().mean() > 0.5 and "csv_pos" in base_df.columns:
                    # Build a small helper index by (name,pos)
                    e2 = e.copy()
                    e2["__key"] = e2["__norm_name"] + "|" + e2.get("csv_pos", "")
                    base_df["__key"] = base_df["__norm_name"] + "|" + base_df["position"].fillna("")
                    base_df = base_df.drop(columns=["ecr"], errors="ignore").merge(
                        e2[["__key", "ecr"]], on="__key", how="left"
                    )

        # Value score (positive means good value vs ADP)
        # Smaller ADP (earlier) means drafted earlier. If ECR=20 and ADP=30 => value of +10 picks later than talent suggests.
        base_df["ecr"] = pd.to_numeric(base_df.get("ecr", pd.Series(index=base_df.index)), errors="coerce")
        base_df["adp"] = pd.to_numeric(base_df.get("adp", pd.Series(index=base_df.index)), errors="coerce")
        # Compute two flavors: by ADP position vs ECR, and by ADP rank vs ECR
        base_df["value_diff"] = base_df["adp"] - base_df["ecr"]

        # Final pretty columns
        pretty_cols = ["player_name", "position", "team", "adp", "ecr", "value_diff"]
        opt_cols = [c for c in ["adp_rank", "pos_adp"] if c in base_df.columns]
        out = base_df[["player_id"] + pretty_cols + opt_cols].copy()

        # Sort good values to top by default
        out = out.sort_values(by=["value_diff"], ascending=[False], na_position="last")
        return out

# ------------------------------------------------------------------------------------
# UI + App logic
# ------------------------------------------------------------------------------------

def main():
    st.title("🏈 Fantasy Football Analyzer")
    st.caption("Sleeper ADP + your Expert CSV to find values for your draft.")

    # Sidebar Controls
    st.sidebar.header("Settings")

    current_year = datetime.now().year
    season = st.sidebar.number_input("Season", min_value=2018, max_value=current_year + 1, value=current_year, step=1, key="season_input")
    scoring = st.sidebar.selectbox(
        "Scoring Type (ADP Dataset)",
        ["redraft_ppr", "redraft_half_ppr", "redraft_standard", "dynasty_ppr"],
        index=0,
        help="Controls which Sleeper aggregated ADP dataset to use (when available).",
        key="scoring_type_select"
    )

    use_leagues = st.sidebar.toggle("Use my Sleeper leagues (optional)", value=False, key="use_leagues_toggle")
    username = st.sidebar.text_input("Sleeper username", placeholder="your_sleeper_username", disabled=not use_leagues, key="sleeper_username_input")

    # Expert CSV upload
    st.sidebar.subheader("Upload Expert Rankings (CSV)")
    csv_file = st.sidebar.file_uploader(
        "CSV with at least Name and Rank/ECR/Overall columns",
        type=["csv"],
        accept_multiple_files=False,
        key="expert_csv_uploader"
    )

    # Data loading
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Load/Refresh Data", type="primary", key="btn_load_refresh"):
        with st.spinner("Loading Sleeper data..."):
            client = SleeperClient(season=season, scoring_type=scoring)
            players = client.get_players()
            adp_df = client.get_adp()

            # Read expert CSV to DataFrame
            expert_df = None
            if csv_file is not None:
                try:
                    expert_df = pd.read_csv(csv_file)
                    st.sidebar.success("CSV loaded.")
                except Exception as e:
                    st.sidebar.error(f"Failed to read CSV: {e}")

            # Filter positions (optional quick filter in main area)
            st.session_state.players = players
            st.session_state.adp_df = adp_df
            st.session_state.expert_df = expert_df
            st.success("Data loaded.")

    # Display guidance if not loaded yet
    if "players" not in st.session_state:
        st.info("Load data from the sidebar to begin.")
        st.stop()

    client = SleeperClient(season=season, scoring_type=scoring)
    players = st.session_state.get("players", {})
    adp_df = st.session_state.get("adp_df", pd.DataFrame())
    expert_df = st.session_state.get("expert_df")

    # Optional: show league list if username provided
    if use_leagues and username:
        with st.expander("My Sleeper Leagues"):
            uid = client.get_user_id(username)
            leagues = client.get_user_leagues(uid) if uid else []
            if uid and leagues:
                lg_df = pd.DataFrame([
                    {
                        "league_id": lg.get("league_id"),
                        "name": lg.get("name"),
                        "draft_rounds": lg.get("draft_rounds"),
                        "roster_positions": ",".join(lg.get("roster_positions", [])) if isinstance(lg.get("roster_positions"), list) else None,
                        "scoring_settings.ppr": (lg.get("scoring_settings") or {}).get("rec", None),
                    }
                    for lg in leagues
                ])
                st.dataframe(lg_df, width="stretch", hide_index=True)
            else:
                st.caption("No leagues found for this username + season yet, or user lookup failed.")

    # Let user pick positions to include
    st.subheader("Build Board")
    keep_positions = st.multiselect(
        "Filter positions to include (optional)",
        ["QB", "RB", "WR", "TE", "K", "DEF"],
        default=["QB", "RB", "WR", "TE"],
        key="pos_filter_multi"
    )

    if st.button("⚙️ Build/Refresh Board", key="btn_build_board"):
        with st.spinner("Computing value board..."):
            table = client.build_enhanced_table(expert_df, players, adp_df, keep_positions=keep_positions if keep_positions else None)
            st.session_state.value_table = table
        st.success("Board ready.")

    if "value_table" not in st.session_state or st.session_state.value_table is None:
        st.stop()

    value_df = st.session_state.value_table

    # KPI area
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Players Loaded", f"{len(players):,}")
    with c2:
        st.metric("ADP Rows", f"{len(adp_df):,}")
    with c3:
        missing = int(value_df["adp"].isna().sum()) if "adp" in value_df.columns else len(value_df)
        st.metric("Missing ADP", f"{missing:,}")
    with c4:
        matched = int(value_df["ecr"].notna().sum())
        st.metric("CSV Matches", f"{matched:,}")

    # Top values & overvalued
    st.subheader("Top Values (higher = better value)")
    top_values = value_df.dropna(subset=["value_diff"]).sort_values("value_diff", ascending=False).head(50)
    st.dataframe(top_values, width="stretch", hide_index=True)

    st.subheader("Most Overvalued (negative = going earlier than ECR)")
    worst_values = value_df.dropna(subset=["value_diff"]).sort_values("value_diff", ascending=True).head(50)
    st.dataframe(worst_values, width="stretch", hide_index=True)

    # Positional views
    with st.expander("Positional Views"):
        for pos in ["QB", "RB", "WR", "TE", "K", "DEF"]:
            sub = value_df[value_df["position"] == pos]
            if not sub.empty:
                st.markdown(f"**{pos}**")
                st.dataframe(sub.sort_values("value_diff", ascending=False), width="stretch", hide_index=True)

    # Download
    @st.cache_data(show_spinner=False)
    def _to_csv_bytes(df: pd.DataFrame) -> bytes:
        return df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download Value Board (CSV)",
        data=_to_csv_bytes(value_df),
        file_name=f"value_board_{season}.csv",
        mime="text/csv",
        key="dl_value_csv"
    )

    st.caption("Note: If Sleeper's ADP endpoint returns no rows for the chosen type/season, you'll still see players but ADP will be missing. Try a different 'Scoring Type (ADP Dataset)' above.")

if __name__ == "__main__":
    main()
