import re
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# =========================
# Page / App configuration
# =========================
st.set_page_config(
    page_title="Fantasy Draft Guide + Live Board",
    page_icon="🏈",
    layout="wide",
)

# -------------------------
# Helpers: robust utilities
# -------------------------
def _lower_alnum(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s)
    # remove common suffixes and punctuation/spaces for matching
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", s, flags=re.I)
    s = re.sub(r"[^a-z0-9]", "", s.lower())
    return s


def autodetect(colnames: List[str], candidates: List[str]) -> Optional[str]:
    """Return the first column from colnames that fuzzy-matches any candidate (case-insensitive)."""
    lower = {c.lower(): c for c in colnames}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    # fuzzy contains
    for c in colnames:
        for cand in candidates:
            if cand.lower() in c.lower():
                return c
    return None


def safe_int(x) -> Optional[int]:
    try:
        if pd.isna(x):
            return None
        return int(float(x))
    except Exception:
        return None


def safe_float(x) -> Optional[float]:
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def to_round_pick(adp: float, league_size: int) -> Tuple[int, int]:
    """Given ADP and league size, return (round, pick_in_round) 1-indexed."""
    if adp is None or np.isnan(adp) or league_size <= 0:
        return (None, None)
    pick_num = int(round(adp))
    rnd = (pick_num - 1) // league_size + 1
    pick_in = (pick_num - 1) % league_size + 1
    return (rnd, pick_in)


# -----------------------------------
# Sleeper ADP CSV normalization layer
# -----------------------------------
def normalize_adp_df(raw: pd.DataFrame, source_name: str) -> pd.DataFrame:
    df = raw.copy()

    # Try to find usable columns
    name_col = autodetect(df.columns.tolist(), ["player_name", "name", "player", "full_name"])
    adp_col = autodetect(df.columns.tolist(), ["adp", "avg_pick", "average_pick", "adp_value", "overall_adp"])
    team_col = autodetect(df.columns.tolist(), ["team", "team_code", "nfl_team"])
    pos_col = autodetect(df.columns.tolist(), ["position", "pos"])
    rank_col = autodetect(df.columns.tolist(), ["rank", "overall_rank", "overall"])

    # Basic rename to standard schema
    rename_map = {}
    if name_col: rename_map[name_col] = "name"
    if adp_col: rename_map[adp_col] = "adp"
    if team_col: rename_map[team_col] = "team"
    if pos_col: rename_map[pos_col] = "pos"
    if rank_col: rename_map[rank_col] = "adp_rank_raw"

    df = df.rename(columns=rename_map)

    # Drop rows without names
    if "name" not in df.columns:
        # Create an empty standard table if unusable
        return pd.DataFrame(columns=["name", "adp", "team", "pos", "adp_rank", "adp_source", "match_key"])

    # Coerce numeric
    if "adp" in df.columns:
        df["adp"] = df["adp"].apply(safe_float)
    else:
        df["adp"] = np.nan

    # If rank missing, derive from adp
    if "adp_rank_raw" not in df.columns or df["adp_rank_raw"].isna().all():
        # If ADP available, derive rank by sorting; else leave NaN
        if df["adp"].notna().any():
            df = df.sort_values(by=["adp"], ascending=True, na_position="last")
            df["adp_rank_raw"] = range(1, len(df) + 1)
        else:
            df["adp_rank_raw"] = np.nan

    df["team"] = df.get("team", "").fillna("").astype(str)
    df["pos"] = df.get("pos", "").fillna("").astype(str)

    # Standardized fields
    df["adp_rank"] = df["adp_rank_raw"].apply(safe_int)
    df["adp_source"] = source_name
    df["match_key"] = df["name"].map(_lower_alnum)

    keep = ["name", "team", "pos", "adp", "adp_rank", "adp_source", "match_key"]
    return df[keep]


def load_adp_uploads(files: List) -> pd.DataFrame:
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except UnicodeDecodeError:
            f.seek(0)
            df = pd.read_csv(f, encoding="latin-1")
        except Exception:
            f.seek(0)
            df = pd.read_csv(f, sep=";")
        frames.append(normalize_adp_df(df, source_name=getattr(f, "name", "uploaded_adp")))
    if not frames:
        return pd.DataFrame(columns=["name", "team", "pos", "adp", "adp_rank", "adp_source", "match_key"])
    out = pd.concat(frames, ignore_index=True)
    # If multiple sources, keep best info per player by lowest ADP rank (market strongest signal)
    out = out.sort_values(by=["match_key", "adp_rank"], na_position="last").drop_duplicates("match_key", keep="first")
    return out.reset_index(drop=True)


# -------------------------------------
# Expert Rankings CSV normalization layer
# -------------------------------------
def normalize_rankings_df(
    raw: pd.DataFrame,
    player_col: Optional[str] = None,
    rank_col: Optional[str] = None,
    pos_col: Optional[str] = None,
    team_col: Optional[str] = None,
) -> pd.DataFrame:
    df = raw.copy()

    # Auto-detect if not provided
    detected_player = player_col or autodetect(df.columns.tolist(), ["player", "name", "player_name", "full_name"])
    detected_rank = rank_col or autodetect(df.columns.tolist(), ["rank", "overall_rank", "expert_rank"])
    detected_pos = pos_col or autodetect(df.columns.tolist(), ["position", "pos"])
    detected_team = team_col or autodetect(df.columns.tolist(), ["team", "team_code", "nfl_team"])

    if detected_player is None or detected_rank is None:
        # Create an empty frame flagged for mapping later
        return pd.DataFrame(columns=["name", "expert_rank", "pos", "team", "match_key"])

    rename_map = {detected_player: "name", detected_rank: "expert_rank"}
    if detected_pos: rename_map[detected_pos] = "pos"
    if detected_team: rename_map[detected_team] = "team"

    df = df.rename(columns=rename_map)

    # Coerce rank
    df["expert_rank"] = df["expert_rank"].apply(safe_int)
    df["pos"] = df.get("pos", "").fillna("").astype(str)
    df["team"] = df.get("team", "").fillna("").astype(str)

    # Match key
    df["match_key"] = df["name"].map(_lower_alnum)
    keep = ["name", "expert_rank", "pos", "team", "match_key"]
    df = df[keep].dropna(subset=["name", "expert_rank"])
    return df.reset_index(drop=True)


# -------------------------
# Value calculation & views
# -------------------------
def compute_value_table(adp_df: pd.DataFrame, ranks_df: pd.DataFrame, league_size: int) -> pd.DataFrame:
    if adp_df.empty or ranks_df.empty:
        return pd.DataFrame(columns=[
            "name", "team", "pos", "adp", "adp_rank", "expert_rank",
            "value_vs_adp", "value_color", "adp_rnd", "adp_pick_in_rnd"
        ])

    merged = pd.merge(
        adp_df,
        ranks_df[["match_key", "expert_rank"]],
        on="match_key",
        how="inner",
        validate="one_to_one"
    )

    # value (negative is good: expert says earlier than market)
    merged["value_vs_adp"] = merged["expert_rank"] - merged["adp_rank"]
    # round/pick from adp
    rp = merged["adp"].apply(lambda x: to_round_pick(x, league_size))
    merged["adp_rnd"] = [a if isinstance(a, int) else None for a, _ in rp]
    merged["adp_pick_in_rnd"] = [b if isinstance(b, int) else None for _, b in rp]

    # quick color hint (string tag – you can convert to styling in st.dataframe if desired)
    def tag(v):
        if pd.isna(v):
            return ""
        if v <= -15:
            return "💚💚 great"
        if v <= -7:
            return "💚 good"
        if v >= 15:
            return "💔 avoid"
        if v >= 7:
            return "🟧 pricey"
        return "⚪ neutral"

    merged["value_color"] = merged["value_vs_adp"].map(tag)

    cols = ["name", "team", "pos", "adp", "adp_rank", "expert_rank", "value_vs_adp", "value_color", "adp_rnd", "adp_pick_in_rnd"]
    merged = merged[cols].sort_values(by=["value_vs_adp", "expert_rank"], ascending=[True, True], na_position="last").reset_index(drop=True)
    return merged


# -------------------------
# Live draft board utilities
# -------------------------
def snake_team_for_pick(pick_number: int, num_teams: int) -> int:
    """Return team index (1..num_teams) for a given 1-indexed pick number."""
    if pick_number <= 0 or num_teams <= 0:
        return 1
    round_idx = (pick_number - 1) // num_teams  # 0-based round
    pos_in_round = (pick_number - 1) % num_teams  # 0..num_teams-1
    if round_idx % 2 == 0:
        # normal order
        return pos_in_round + 1
    else:
        # reversed order
        return num_teams - pos_in_round


def linear_team_for_pick(pick_number: int, num_teams: int) -> int:
    pos_in_round = (pick_number - 1) % num_teams
    return pos_in_round + 1


def build_board_df(picks: List[Dict], num_teams: int, num_rounds: int) -> pd.DataFrame:
    board = pd.DataFrame(index=[f"R{r}" for r in range(1, num_rounds + 1)],
                         columns=[f"T{t}" for t in range(1, num_teams + 1)])
    for p in picks:
        rnd = p["round"]
        team = p["team"]
        label = p["label"]
        if 1 <= rnd <= num_rounds and 1 <= team <= num_teams:
            board.loc[f"R{rnd}", f"T{team}"] = label
    return board


def next_pick_metadata(picks_count: int, num_teams: int, draft_type: str) -> Tuple[int, int]:
    pick_no = picks_count + 1
    rnd = (pick_no - 1) // num_teams + 1
    if draft_type == "Snake":
        team = snake_team_for_pick(pick_no, num_teams)
    else:
        team = linear_team_for_pick(pick_no, num_teams)
    return rnd, team


# ==============
# Session memory
# ==============
def ensure_state():
    ss = st.session_state
    ss.setdefault("adp_df", pd.DataFrame())
    ss.setdefault("ranks_df", pd.DataFrame())
    ss.setdefault("values_df", pd.DataFrame())
    ss.setdefault("draft_picks", [])  # list of dicts: {round, team, name, label, pos, team_code}
    ss.setdefault("draft_type", "Snake")
    ss.setdefault("num_teams", 12)
    ss.setdefault("num_rounds", 16)
    ss.setdefault("my_slot", 1)
    ss.setdefault("filters_pos", set(["QB", "RB", "WR", "TE", "DST", "K"]))
    ss.setdefault("sort_choice", "Best value (expert vs ADP)")
    ss.setdefault("only_available", True)
    ss.setdefault("show_position_adp", False)
    ss.setdefault("player_search", "")


ensure_state()

# ===========
# Side Bar UI
# ===========
with st.sidebar:
    st.header("⚙️ Settings")

    st.selectbox(
        "Draft Type",
        ["Snake", "Linear"],
        index=0 if st.session_state.draft_type == "Snake" else 1,
        key="draft_type_select",
    )
    st.session_state.draft_type = st.session_state.draft_type_select

    st.number_input("Number of Teams", 2, 20, value=st.session_state.num_teams, key="num_teams_input")
    st.session_state.num_teams = st.session_state.num_teams_input

    st.number_input("Rounds", 1, 30, value=st.session_state.num_rounds, key="num_rounds_input")
    st.session_state.num_rounds = st.session_state.num_rounds_input

    st.number_input("Your Draft Slot", 1, st.session_state.num_teams, value=st.session_state.my_slot, key="my_slot_input")
    st.session_state.my_slot = st.session_state.my_slot_input

    st.markdown("---")
    st.caption("Upload one or more Sleeper ADP CSV files:")
    adp_files = st.file_uploader(
        "Sleeper ADP CSVs",
        type=["csv"],
        accept_multiple_files=True,
        key="adp_uploader",
    )

    st.caption("Upload your expert rankings CSV file(s):")
    ranks_files = st.file_uploader(
        "Expert Rankings CSV(s)",
        type=["csv"],
        accept_multiple_files=True,
        key="ranks_uploader",
    )

    st.markdown("---")
    if st.button("🔄 Load / Refresh Data", type="primary", key="btn_refresh_data"):
        with st.spinner("Loading & normalizing ADP..."):
            adp_df = load_adp_uploads(adp_files or [])
        with st.spinner("Loading & normalizing Expert Rankings..."):
            ranks_frames = []
            for f in (ranks_files or []):
                try:
                    raw = pd.read_csv(f)
                except UnicodeDecodeError:
                    f.seek(0)
                    raw = pd.read_csv(f, encoding="latin-1")
                except Exception:
                    f.seek(0)
                    raw = pd.read_csv(f, sep=";")

                ranks_frames.append(normalize_rankings_df(raw))
            ranks_df = pd.concat(ranks_frames, ignore_index=True) if ranks_frames else pd.DataFrame(
                columns=["name", "expert_rank", "pos", "team", "match_key"]
            )

        st.session_state.adp_df = adp_df
        st.session_state.ranks_df = ranks_df

        with st.spinner("Computing values..."):
            st.session_state.values_df = compute_value_table(
                st.session_state.adp_df, st.session_state.ranks_df, st.session_state.num_teams
            )
        st.success("Data loaded ✅")

    if st.button("🧹 Reset Draft Board", key="btn_reset_draft"):
        st.session_state.draft_picks = []
        st.success("Draft board reset")

# ==================
# Main Page Sections
# ==================
st.title("🏈 Fantasy Draft Guide + Live Draft Board")

# Info about what was loaded
colA, colB, colC = st.columns([1, 1, 2])
with colA:
    st.metric("Players with ADP", f"{len(st.session_state.adp_df):,}")
with colB:
    st.metric("Players with Expert Rank", f"{len(st.session_state.ranks_df):,}")
with colC:
    st.metric("Matched for Value Calc", f"{len(st.session_state.values_df):,}")

st.markdown("---")

# ======================
# Value Finder / Filters
# ======================
st.subheader("📊 Value Finder (Expert Rank vs Market ADP)")
vf1, vf2, vf3, vf4 = st.columns([1.2, 1, 1, 1.2])
with vf1:
    st.selectbox(
        "Sort players by",
        ["Best value (expert vs ADP)", "Expert Rank", "Market ADP"],
        key="sort_choice",
    )
with vf2:
    st.checkbox("Only available (not drafted)", value=st.session_state.only_available, key="only_available")
with vf3:
    st.checkbox("Show Position ADP (by group)", value=st.session_state.show_position_adp, key="show_position_adp")
with vf4:
    st.text_input("Search player name", value=st.session_state.player_search, key="player_search")

pos_filters = st.multiselect(
    "Filter positions",
    ["QB", "RB", "WR", "TE", "DST", "K"],
    default=list(st.session_state.filters_pos),
    key="pos_filters",
)

st.session_state.filters_pos = set(pos_filters)

# Build "available" set
drafted_names = set(p["name"] for p in st.session_state.draft_picks)

df_values = st.session_state.values_df.copy()

# Position filter (map some DST aliases)
def _normalize_pos(p):
    p = (p or "").upper()
    if p in ("D/ST", "DST", "DEF", "D"):
        return "DST"
    return p

df_values["pos"] = df_values["pos"].map(_normalize_pos)
df_values = df_values[df_values["pos"].isin(st.session_state.filters_pos)]

if st.session_state.player_search.strip():
    key = _lower_alnum(st.session_state.player_search.strip())
    df_values = df_values[df_values["name"].map(_lower_alnum).str.contains(key)]

if st.session_state.only_available:
    df_values = df_values[~df_values["name"].isin(drafted_names)]

# Optional: compute position ADP (rank inside position) for current view
if st.session_state.show_position_adp:
    df_values["pos_adp_rank"] = (
        df_values.sort_values(["pos", "adp_rank"], ascending=[True, True])
        .groupby("pos")
        .cumcount() + 1
    )

# Sorting
if st.session_state.sort_choice == "Expert Rank":
    df_values = df_values.sort_values(by=["expert_rank", "adp_rank"], ascending=[True, True])
elif st.session_state.sort_choice == "Market ADP":
    df_values = df_values.sort_values(by=["adp_rank", "expert_rank"], ascending=[True, True])
else:
    df_values = df_values.sort_values(by=["value_vs_adp", "expert_rank"], ascending=[True, True])

# Show the table
st.dataframe(
    df_values.head(300),
    use_container_width=None,  # intentionally not used; replaced by width below
    width="stretch",
    height=450,
)

# ===============
# Live Draft Board
# ===============
st.subheader("🧩 Live Draft Board")

# Next pick info
picks_count = len(st.session_state.draft_picks)
np_round, np_team = next_pick_metadata(picks_count, st.session_state.num_teams, st.session_state.draft_type)
st.caption(f"Next pick ➜ Round {np_round}, Team {np_team}")

left, right = st.columns([1.6, 1.4])

with left:
    st.markdown("**Add a Pick**")

    # pick a player from available list
    available_names = df_values["name"].tolist()
    prefill = available_names[0] if available_names else ""
    picked_player = st.selectbox("Choose player", available_names, index=0 if available_names else None, key="sel_player")

    # allow override: which team is picking (default: computed)
    team_choice = st.number_input(
        "Team (1..N)",
        min_value=1, max_value=st.session_state.num_teams, value=np_team or 1,
        key="num_pick_team"
    )
    round_choice = st.number_input(
        "Round",
        min_value=1, max_value=st.session_state.num_rounds, value=np_round or 1,
        key="num_pick_round"
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("➕ Add Pick", type="primary", key="btn_add_pick") and picked_player:
            # Look up pos/team for label
            row = st.session_state.values_df.loc[st.session_state.values_df["name"] == picked_player]
            pos = row["pos"].iloc[0] if not row.empty else ""
            tcode = row["team"].iloc[0] if not row.empty else ""

            st.session_state.draft_picks.append({
                "round": int(round_choice),
                "team": int(team_choice),
                "name": picked_player,
                "label": f"{picked_player} ({pos}-{tcode})",
                "pos": pos,
                "team_code": tcode
            })
            st.success(f"Added: {picked_player} ➜ R{round_choice} T{team_choice}")

    with c2:
        if st.button("↩️ Undo Last", key="btn_undo_last"):
            if st.session_state.draft_picks:
                last = st.session_state.draft_picks.pop()
                st.info(f"Removed: {last['label']}")
            else:
                st.warning("No picks to undo")

    with c3:
        if st.button("💾 Export Board CSV", key="btn_export_board"):
            board = build_board_df(st.session_state.draft_picks, st.session_state.num_teams, st.session_state.num_rounds)
            csv = board.to_csv().encode("utf-8")
            st.download_button("Download Board.csv", data=csv, file_name="draft_board.csv", mime="text/csv",
                               key="btn_dl_board")

with right:
    board = build_board_df(st.session_state.draft_picks, st.session_state.num_teams, st.session_state.num_rounds)
    st.markdown("**Draft Board**")
    st.dataframe(board, width="stretch", height=550)

st.markdown("---")

# ====================
# Value by Position(s)
# ====================
st.subheader("🔍 Positional Views")
pos_group = st.radio("Choose a position", ["All", "QB", "RB", "WR", "TE", "DST", "K"], horizontal=True, key="radio_pos")
pos_df = st.session_state.values_df.copy()
pos_df["pos"] = pos_df["pos"].map(_normalize_pos)

if pos_group != "All":
    pos_df = pos_df[pos_df["pos"] == pos_group]

if st.session_state.only_available:
    drafted_names = set(p["name"] for p in st.session_state.draft_picks)
    pos_df = pos_df[~pos_df["name"].isin(drafted_names)]

st.dataframe(
    pos_df.sort_values(by=["value_vs_adp", "expert_rank"], ascending=[True, True]).head(200),
    width="stretch",
    height=400,
)

# =====
# Notes
# =====
st.caption("Tip: Value = Expert Rank − ADP Rank (negative is good value; positive is overpriced).")
