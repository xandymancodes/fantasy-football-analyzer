# streamlit run app.py
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import pandas as pd
import requests
import streamlit as st

# =====================================================
# Streamlit setup
# =====================================================
st.set_page_config(page_title="Fantasy Football Analyzer", page_icon="🏈", layout="wide")
st.title("🏈 Fantasy Football Analyzer — Sleeper ADP + Expert Ranks")

# =====================================================
# Helpers
# =====================================================
def _norm_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    s = name.strip()
    # If "Last, First" -> "First Last"
    if "," in s:
        parts = [p.strip() for p in s.split(",", 1)]
        if len(parts) == 2:
            s = f"{parts[1]} {parts[0]}"
    s = s.replace(".", " ")
    s = " ".join(s.split())
    return s.lower()

def _first(*vals):
    for v in vals:
        if v is not None and v != "":
            return v
    return None

# =====================================================
# Sleeper Client (resilient + cached)
# =====================================================
@st.cache_data(ttl=3600, show_spinner=False)
def _get_json(url: str, params: Optional[dict] = None) -> Optional[dict | list]:
    try:
        r = requests.get(url, params=params or {}, timeout=25)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

class Sleeper:
    def __init__(self, season: int, scoring_type: str) -> None:
        self.season = season
        self.scoring_type = scoring_type

    # ---------- Players ----------
    @st.cache_data(ttl=60*60, show_spinner=False)
    def players(_self) -> Dict[str, dict]:
        # Try new domain first, then legacy
        data = _get_json("https://api.sleeper.com/players/nfl")
        if not data:
            data = _get_json("https://api.sleeper.app/v1/players/nfl")
        return data or {}

    # ---------- Aggregated ADP ----------
    @st.cache_data(ttl=60*60, show_spinner=False)
    def adp_aggregate(_self) -> pd.DataFrame:
        # Try newer aggregate endpoint
        for base in ["https://api.sleeper.com", "https://api.sleeper.app/v1"]:
            data = _get_json(
                f"{base}/adp/nfl/{_self.season}",
                params={"season_type": "regular", "type": _self.scoring_type},
            )
            if isinstance(data, list) and data:
                df = pd.DataFrame(data)
                # Normalize common fields
                rename = {}
                for c in df.columns:
                    lc = c.lower()
                    if lc == "player":
                        rename[c] = "player_id"
                    if lc in ("averagedraftposition", "adp_overall"):
                        rename[c] = "adp"
                if rename:
                    df = df.rename(columns=rename)
                # Keep relevant columns when present
                keep = [c for c in ["player_id", "name", "adp", "rank", "position", "team", "count"] if c in df.columns]
                return df[keep].copy()
        # Legacy fallback (rare)
        legacy = _get_json("https://api.sleeper.app/v1/players/nfl/adp")
        if isinstance(legacy, dict) and legacy:
            rows = []
            for pid, row in legacy.items():
                if isinstance(row, dict):
                    rows.append({"player_id": pid, **row})
            if rows:
                df = pd.DataFrame(rows)
                keep = [c for c in ["player_id", "adp", "rank"] if c in df.columns]
                return df[keep].copy()
        # Empty
        return pd.DataFrame(columns=["player_id", "adp", "rank"])

    # ---------- User/Leagues/Drafts for personal ADP ----------
    @st.cache_data(ttl=6*60, show_spinner=False)
    def user_id(_self, username_or_id: str) -> Optional[str]:
        if not username_or_id:
            return None
        for base in ["https://api.sleeper.com", "https://api.sleeper.app/v1"]:
            data = _get_json(f"{base}/user/{username_or_id}")
            if isinstance(data, dict) and data.get("user_id"):
                return data["user_id"]
        return None

    @st.cache_data(ttl=6*60, show_spinner=False)
    def leagues(_self, user_id: str) -> List[dict]:
        leagues = []
        for base in ["https://api.sleeper.com", "https://api.sleeper.app/v1"]:
            data = _get_json(f"{base}/user/{user_id}/leagues/nfl/{_self.season}")
            if isinstance(data, list) and data:
                leagues = data
                break
        return leagues or []

    @st.cache_data(ttl=6*60, show_spinner=False)
    def league_drafts(_self, league_id: str) -> List[dict]:
        for base in ["https://api.sleeper.com", "https://api.sleeper.app/v1"]:
            data = _get_json(f"{base}/league/{league_id}/drafts")
            if isinstance(data, list):
                return data
        return []

    @st.cache_data(ttl=6*60, show_spinner=False)
    def draft_picks(_self, draft_id: str) -> List[dict]:
        for base in ["https://api.sleeper.com", "https://api.sleeper.app/v1"]:
            data = _get_json(f"{base}/draft/{draft_id}/picks")
            if isinstance(data, list):
                return data
        return []

    @st.cache_data(ttl=10*60, show_spinner=False)
    def adp_from_my_drafts(_self, username_or_id: Optional[str], league_ids: Optional[List[str]]) -> Tuple[pd.DataFrame, dict]:
        diag = {"draft_ids": 0, "picks": 0, "leagues": 0}
        draft_ids: List[str] = []

        if league_ids:
            for lg in league_ids:
                for d in _self.league_drafts(lg) or []:
                    if d.get("draft_id"):
                        draft_ids.append(d["draft_id"])
        elif username_or_id:
            uid = _self.user_id(username_or_id)
            if uid:
                leagues = _self.leagues(uid) or []
                diag["leagues"] = len(leagues)
                for lg in leagues:
                    for d in _self.league_drafts(lg.get("league_id")) or []:
                        if d.get("draft_id"):
                            draft_ids.append(d["draft_id"])

        diag["draft_ids"] = len(draft_ids)

        rows = []
        for did in draft_ids:
            picks = _self.draft_picks(did) or []
            diag["picks"] += len(picks)
            for p in picks:
                pid = p.get("player_id") or (p.get("metadata") or {}).get("player_id")
                pick_no = p.get("pick_no") or p.get("overall") or p.get("pick")
                if pid and pick_no is not None:
                    try:
                        rows.append({"player_id": str(pid), "pick_no": float(pick_no)})
                    except Exception:
                        pass

        if not rows:
            return pd.DataFrame(columns=["player_id", "adp", "rank"]), diag

        df = pd.DataFrame(rows).groupby("player_id").agg(adp=("pick_no", "mean"), count=("pick_no", "count")).reset_index()
        df["rank"] = df["adp"].rank(method="dense")
        return df[["player_id", "adp", "rank", "count"]], diag

# =====================================================
# CSV Handling (robust)
# =====================================================
def load_csv() -> Optional[pd.DataFrame]:
    st.sidebar.subheader("📊 Expert Rankings (CSV)")
    up = st.sidebar.file_uploader("Upload CSV (include Name and Rank/ECR/Overall)", type=["csv"], key="csv_upload")
    if up is None:
        return None
    try:
        up.seek(0)
        try:
            df = pd.read_csv(up)
        except Exception:
            up.seek(0)
            df = pd.read_csv(up, engine="python", sep=None, on_bad_lines="skip")
        if df.empty:
            st.sidebar.error("CSV appears empty.")
            return None
        # Normalize headers
        df.columns = df.columns.str.lower().str.strip().str.replace(r"\s+", "_", regex=True)
        # Coerce common rank columns
        for c in ["rank", "overall", "overall_rank", "ecr"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        with st.sidebar.expander("Preview CSV"):
            st.dataframe(df.head(20))
        return df
    except Exception as e:
        st.sidebar.error(f"Error reading CSV: {e}")
        return None

# =====================================================
# Build tables
# =====================================================
def build_players_table(players: Dict[str, dict], include_positions: Optional[List[str]]) -> pd.DataFrame:
    rows = []
    for pid, p in (players or {}).items():
        pos = _first(p.get("position"), (p.get("fantasy_positions") or [None])[0])
        if include_positions and pos and pos not in include_positions:
            continue
        rows.append({
            "player_id": str(pid),
            "player_name": _first(p.get("full_name"), f"{p.get('first_name','')} {p.get('last_name','')}".strip()),
            "position": pos,
            "team": p.get("team"),
        })
    base = pd.DataFrame(rows)
    return base

def attach_adp(base: pd.DataFrame, adp_df: pd.DataFrame, fallback_name_match: bool = True) -> pd.DataFrame:
    df = base.copy()
    slim = adp_df.copy()
    # Try ID join first
    if "player_id" in slim.columns and "player_id" in df.columns:
        slim = slim[[c for c in slim.columns if c in {"player_id", "adp", "rank", "count"}]]
        df = df.merge(slim, on="player_id", how="left")
    # Fallback name join
    if fallback_name_match and df["adp"].isna().all():
        df["__n"] = df["player_name"].map(_norm_name)
        if "name" in adp_df.columns:
            slim = adp_df[["name", "adp", "rank"]].copy()
            slim["__n"] = slim["name"].map(_norm_name)
            df = df.drop(columns=["adp","rank"], errors="ignore").merge(slim[["__n","adp","rank"]], on="__n", how="left")
    # Derive rank if missing
    if "rank" not in df.columns or df["rank"].isna().all():
        if "adp" in df.columns:
            df["rank"] = df["adp"].rank(method="dense")
    return df

def attach_expert(base: pd.DataFrame, csv_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if csv_df is None or csv_df.empty:
        return base
    df = base.copy()
    e = csv_df.copy()
    # Headers
    name_col = next((c for c in ["player_name","name","player","full_name"] if c in e.columns), None)
    rank_col = next((c for c in ["ecr","rank","overall","overall_rank","consensus_rank"] if c in e.columns), None)
    if not name_col or not rank_col:
        st.warning("CSV must include a player name column and a rank column (e.g., Name + Rank/ECR).")
        return df
    e = e.rename(columns={name_col:"csv_name", rank_col:"ecr"})
    df["__n"] = df["player_name"].map(_norm_name)
    e["__n"] = e["csv_name"].map(_norm_name)
    df = df.merge(e[["__n","ecr"]], on="__n", how="left")
    return df

def build_value_board(players: Dict[str, dict],
                      adp_df: pd.DataFrame,
                      csv_df: Optional[pd.DataFrame],
                      include_positions: Optional[List[str]]) -> pd.DataFrame:
    base = build_players_table(players, include_positions)
    if base.empty:
        return pd.DataFrame()
    df = attach_adp(base, adp_df, fallback_name_match=True)
    df = attach_expert(df, csv_df)
    # Compute value difference: positive means value (going later than expert rank)
    df["ecr"] = pd.to_numeric(df.get("ecr", pd.Series(index=df.index)), errors="coerce")
    df["adp"] = pd.to_numeric(df.get("adp", pd.Series(index=df.index)), errors="coerce")
    df["value_diff"] = df["adp"] - df["ecr"]
    # Order columns
    cols = ["player_name","position","team","adp","rank","ecr","value_diff"]
    cols = [c for c in cols if c in df.columns]
    out = df[cols]
    # Sort best values on top
    out = out.sort_values(["value_diff"], ascending=[False], na_position="last")
    return out

# =====================================================
# Sidebar Controls
# =====================================================
with st.sidebar:
    st.header("Inputs")
    current_year = datetime.now().year
    season = st.number_input("Season", min_value=2018, max_value=current_year+1, value=current_year, step=1, key="season_in")
    scoring_type = st.selectbox(
        "Scoring Type (dataset)",
        ["redraft_ppr","redraft_half_ppr","redraft_standard","dynasty_ppr"],
        index=0, key="score_type"
    )
    use_my_drafts = st.toggle("Prefer ADP from my drafts (if available)", value=False, key="prefer_my_drafts")
    sleeper_user = st.text_input("Sleeper username or user_id", value="", disabled=not use_my_drafts, key="sleeper_user")
    league_ids_raw = st.text_area("Optional: Specific League IDs (one per line)", value="", disabled=not use_my_drafts, key="league_ids_raw")
    league_ids = [s.strip() for s in league_ids_raw.splitlines() if s.strip()]

    csv_df = load_csv()

    if st.button("🔄 Load Sleeper Data", type="primary", key="btn_load"):
        with st.spinner("Fetching players and ADP..."):
            client = Sleeper(season=season, scoring_type=scoring_type)
            players = client.players()
            # Decide ADP source
            adp_df = pd.DataFrame()
            diag = {}
            if use_my_drafts:
                adp_df, diag = client.adp_from_my_drafts(sleeper_user or None, league_ids or None)
            if adp_df.empty:
                adp_df = client.adp_aggregate()
            st.session_state.players = players
            st.session_state.adp_df = adp_df
            st.session_state.csv_df = csv_df
            st.session_state.diag = diag
        st.success("Loaded.")

# =====================================================
# Main Body
# =====================================================
if "players" not in st.session_state:
    st.info("Load data from the sidebar to begin.")
    st.stop()

client = Sleeper(season=season, scoring_type=scoring_type)
players = st.session_state.get("players", {})
adp_df = st.session_state.get("adp_df", pd.DataFrame())
csv_df = st.session_state.get("csv_df", None)

st.subheader("Build Board")
include_positions = st.multiselect("Filter positions", ["QB","RB","WR","TE","K","DEF"], default=["QB","RB","WR","TE"], key="pos_filter")

if st.button("⚙️ Build/Refresh Board", key="btn_build"):
    with st.spinner("Computing value board..."):
        table = build_value_board(players, adp_df, csv_df, include_positions if include_positions else None)
        st.session_state.board = table
    st.success("Board ready.")

board = st.session_state.get("board")
if board is None or board.empty:
    st.info("No board yet. Click Build/Refresh Board.")
    st.stop()

# KPIs
c1,c2,c3,c4 = st.columns(4)
with c1: st.metric("Players", f"{len(players):,}")
with c2: st.metric("ADP rows", f"{len(adp_df):,}")
with c3: st.metric("Missing ADP", f"{int(board['adp'].isna().sum()) if 'adp' in board.columns else 0:,}")
with c4: st.metric("CSV matches", f"{int(board['ecr'].notna().sum()) if 'ecr' in board.columns else 0:,}")

st.subheader("Top Values (higher = better)")
st.dataframe(board.head(50))

st.subheader("Most Overvalued (lower = worse)")
if "value_diff" in board.columns:
    st.dataframe(board.sort_values("value_diff", ascending=True).head(50))

with st.expander("Diagnostics"):
    st.write(st.session_state.get("diag", {}))
    st.caption("If ADP rows = 0, try a different 'Scoring Type (dataset)' or disable 'Prefer ADP from my drafts'. Some seasons/types have sparse data.")

# Download
@st.cache_data(show_spinner=False)
def _to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

st.download_button(
    "⬇️ Download Board (CSV)",
    data=_to_csv_bytes(board),
    file_name=f"value_board_{season}.csv",
    mime="text/csv",
    key="btn_download"
)
