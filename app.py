# streamlit run app.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import re

import pandas as pd
import numpy as np
import streamlit as st

# ===============================
# Page config
# ===============================
st.set_page_config(page_title="Draft Guide (CSV Only)", page_icon="🏈", layout="wide")
st.title("🏈 Draft Guide — CSV-Only (Sleeper ADP + Expert Ranks)")

# ===============================
# Utilities
# ===============================
def norm_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    s = name.strip()
    # "Last, First" -> "First Last"
    if "," in s:
        parts = [p.strip() for p in s.split(",", 1)]
        if len(parts) == 2:
            s = f"{parts[1]} {parts[0]}"
    s = s.replace(".", " ")
    s = re.sub(r"\s+", " ", s)
    return s.lower().strip()

def coalesce(*vals):
    for v in vals:
        if v is not None and v != "" and not (isinstance(v, float) and np.isnan(v)):
            return v
    return None

def find_col(cols: List[str], candidates: List[str]) -> Optional[str]:
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    # contains fallback
    for c in cols:
        for cand in candidates:
            if cand.lower() in c.lower():
                return c
    return None

def infer_dataset_key(filename: str) -> str:
    f = filename.lower()
    key = []
    if "dynasty" in f:
        key.append("dynasty")
    if "2qb" in f or "superflex" in f or "sf" in f:
        key.append("2qb")
    if "half" in f or "half_ppr" in f:
        key.append("half_ppr")
    elif "ppr" in f:
        key.append("ppr")
    elif "std" in f or "standard" in f:
        key.append("std")
    if "all" in f and "league" in f:
        key.append("all_leagues")
    if "complete" in f:
        key.append("complete")
    if not key:
        key.append("unknown")
    return "-".join(key)

# ===============================
# CSV Loaders (ADP and Experts)
# ===============================
def parse_adp_csv(file, filename: str) -> pd.DataFrame:
    """Return normalized ADP table: player_name, position, team, adp, adp_rank, source"""
    try:
        file.seek(0)
        try:
            df = pd.read_csv(file)
        except Exception:
            file.seek(0)
            df = pd.read_csv(file, engine="python", sep=None, on_bad_lines="skip")
    except Exception as e:
        st.error(f"Failed to read ADP CSV {filename}: {e}")
        return pd.DataFrame(columns=["player_name","position","team","adp","adp_rank","source"])

    if df.empty:
        return pd.DataFrame(columns=["player_name","position","team","adp","adp_rank","source"])

    # Normalize headers
    df.columns = df.columns.str.strip().str.lower().str.replace(r"\s+","_", regex=True)

    name_col = find_col(df.columns.tolist(), ["player_name","name","player","full_name"])
    pos_col  = find_col(df.columns.tolist(), ["position","pos"])
    team_col = find_col(df.columns.tolist(), ["team","nfl_team"])
    adp_col  = find_col(df.columns.tolist(), ["adp","average_draft_position","adp_overall"])
    rank_col = find_col(df.columns.tolist(), ["rank","adp_rank","overall_rank"])
    count_col = find_col(df.columns.tolist(), ["count","samples"])

    if not name_col:
        # Some Sleeper CSVs store "player_id" and "name" separately; try to reconstruct name
        name_col = "name" if "name" in df.columns else None

    # Build normalized frame
    out = pd.DataFrame()
    if name_col:
        out["player_name"] = df[name_col].astype(str)
    else:
        # Can't proceed without names
        st.warning(f"{filename}: could not find a name column; skipping.")
        return pd.DataFrame(columns=["player_name","position","team","adp","adp_rank","source"])

    if pos_col in df.columns:
        out["position"] = df[pos_col]
    else:
        out["position"] = None

    if team_col in df.columns:
        out["team"] = df[team_col]
    else:
        out["team"] = None

    if adp_col in df.columns:
        out["adp"] = pd.to_numeric(df[adp_col], errors="coerce")
    else:
        out["adp"] = np.nan

    if rank_col in df.columns:
        out["adp_rank"] = pd.to_numeric(df[rank_col], errors="coerce")
    else:
        # derive rank from adp
        out["adp_rank"] = out["adp"].rank(method="dense")

    if count_col in df.columns:
        out["samples"] = pd.to_numeric(df[count_col], errors="coerce")
    # Add source key from filename
    out["source"] = infer_dataset_key(filename)
    # Normalize names and positions
    out["name_key"] = out["player_name"].map(norm_name)
    if "position" in out.columns:
        out["position"] = out["position"].astype(str).str.upper()
    # Drop duplicate name/position combos; keep best (lowest adp)
    out = out.sort_values(["name_key","adp"], na_position="last").drop_duplicates(subset=["name_key","position"], keep="first")
    return out

def parse_expert_csv(file) -> pd.DataFrame:
    """Return normalized expert ranks: player_name, ecr, position (optional), team (optional)"""
    try:
        file.seek(0)
        try:
            df = pd.read_csv(file)
        except Exception:
            file.seek(0)
            df = pd.read_csv(file, engine="python", sep=None, on_bad_lines="skip")
    except Exception as e:
        st.error(f"Failed to read Expert CSV: {e}")
        return pd.DataFrame(columns=["player_name","ecr","position","team","name_key"])

    if df.empty:
        return pd.DataFrame(columns=["player_name","ecr","position","team","name_key"])

    df.columns = df.columns.str.strip().str.lower().str.replace(r"\s+","_", regex=True)

    name_col = find_col(df.columns.tolist(), ["player_name","name","player","full_name"])
    rank_col = find_col(df.columns.tolist(), ["ecr","rank","overall","overall_rank","consensus_rank"])
    pos_col  = find_col(df.columns.tolist(), ["position","pos"])
    team_col = find_col(df.columns.tolist(), ["team","nfl_team"])

    if not name_col or not rank_col:
        st.warning("Expert CSV must include a player name and rank column (e.g., Name + Rank/ECR).")
        return pd.DataFrame(columns=["player_name","ecr","position","team","name_key"])

    out = pd.DataFrame()
    out["player_name"] = df[name_col].astype(str)
    out["ecr"] = pd.to_numeric(df[rank_col], errors="coerce")
    out["position"] = df[pos_col] if pos_col else None
    out["team"] = df[team_col] if team_col else None
    out["name_key"] = out["player_name"].map(norm_name)
    # Drop rows without rank
    out = out.dropna(subset=["ecr"])
    return out

# ===============================
# Sidebar: Uploads and Settings
# ===============================
st.sidebar.header("📥 Upload Sleeper ADP CSVs")
adp_files = st.sidebar.file_uploader(
    "Upload one or multiple ADP CSVs (PPR, Half-PPR, Standard, Dynasty, 2QB, etc.)",
    type=["csv"],
    accept_multiple_files=True,
    key="adp_uploader"
)
st.sidebar.caption("Tip: The app will infer the dataset type from filenames (e.g., 'ppr', 'half_ppr', '2qb', 'dynasty').")

st.sidebar.header("📥 Upload Expert Rankings CSV")
expert_file = st.sidebar.file_uploader(
    "Upload a CSV with Name and Rank/ECR columns",
    type=["csv"],
    accept_multiple_files=False,
    key="expert_uploader"
)

st.sidebar.header("⚙️ Dataset Choice")
st.sidebar.caption("If you upload multiple ADP files, pick which dataset to use for your guide.")
selected_dataset = st.sidebar.text_input(
    "Dataset key to use (leave blank to auto-pick first)",
    value="",
    key="dataset_key_input"
)

# Load ADP tables
adp_tables: Dict[str, pd.DataFrame] = {}
if adp_files:
    for f in adp_files:
        key = infer_dataset_key(f.name)
        adp_tables[key] = parse_adp_csv(f, f.name)
    # pick default
    default_key = selected_dataset or (list(adp_tables.keys())[0] if adp_tables else "")
else:
    default_key = ""

# Load expert ranks
expert_df = parse_expert_csv(expert_file) if expert_file else pd.DataFrame()

# Choose dataset
if not adp_tables:
    st.info("Upload at least one ADP CSV to proceed.")
    st.stop()

use_key = st.selectbox(
    "Choose ADP dataset",
    options=list(adp_tables.keys()),
    index=max(0, list(adp_tables.keys()).index(default_key) if default_key in adp_tables else 0),
    key="adp_dataset_select"
)
adp_df = adp_tables.get(use_key, pd.DataFrame())

if adp_df.empty:
    st.warning("The chosen ADP dataset is empty after parsing. Please check the file.")
    st.stop()

# ===============================
# Merge ADP with Expert Rankings
# ===============================
merged = adp_df.copy()
if not expert_df.empty:
    merged = merged.merge(
        expert_df[["name_key","ecr"]],
        on="name_key",
        how="left"
    )

# Compute value scores
if "ecr" in merged.columns:
    # If adp_rank missing, compute
    if "adp_rank" not in merged.columns or merged["adp_rank"].isna().all():
        merged["adp_rank"] = merged["adp"].rank(method="dense")
    merged["value_delta"] = (merged["adp_rank"] - merged["ecr"]).astype("Int64")
    merged["value_picks"] = (merged["adp"] - merged["ecr"]).astype("Int64")
else:
    merged["value_delta"] = pd.Series(dtype="Int64")
    merged["value_picks"] = pd.Series(dtype="Int64")

# ===============================
# Draft Guide (Available Players)
# ===============================
st.header("📋 Draft Guide — Available Players")
colA, colB, colC, colD = st.columns([2,1,1,1])
with colA:
    search = st.text_input("Search player", value="", key="search_input")
with colB:
    pos_filter = st.multiselect("Positions", options=["QB","RB","WR","TE","K","DEF"], default=["QB","RB","WR","TE"], key="pos_filter")
with colC:
    sort_by = st.selectbox("Sort by", options=["value_picks","value_delta","adp","adp_rank","ecr","player_name"], index=0, key="sort_by_select")
with colD:
    ascending = st.checkbox("Ascending", value=False, key="sort_ascending")

avail = merged.copy()
# Remove already drafted players (state-managed below)
drafted_keys = st.session_state.get("drafted_set", set())
if drafted_keys:
    avail = avail[~avail["name_key"].isin(drafted_keys)]
# Filters
if pos_filter:
    avail = avail[avail["position"].isin(pos_filter)]
if search.strip():
    s = search.strip().lower()
    avail = avail[avail["player_name"].str.lower().str.contains(s)]
# Sort
if sort_by in avail.columns:
    avail = avail.sort_values(sort_by, ascending=ascending, na_position="last")
st.dataframe(avail[["player_name","position","team","adp","adp_rank","ecr","value_picks","value_delta"]].head(200), width="stretch", hide_index=True)

# ===============================
# Live Draft Board
# ===============================
st.header("🧑‍💻 Live Draft Board (Manual Entry)")
with st.expander("Board Settings", expanded=True):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        teams = st.number_input("Number of teams", min_value=4, max_value=16, value=12, step=1, key="teams_num")
    with col2:
        rounds = st.number_input("Rounds", min_value=1, max_value=30, value=15, step=1, key="rounds_num")
    with col3:
        draft_type = st.selectbox("Draft type", options=["snake","linear"], index=0, key="draft_type_select")
    with col4:
        my_slot = st.number_input("My draft slot", min_value=1, max_value=int(teams), value=1, step=1, key="my_slot_num")

    if st.button("♻️ Reset Draft", key="reset_board_btn"):
        st.session_state.drafted_set = set()
        st.session_state.picks = []

# Initialize state
if "picks" not in st.session_state:
    st.session_state.picks = []
if "drafted_set" not in st.session_state:
    st.session_state.drafted_set = set()

# Compute pick order for current settings
def team_for_pick(pick_no: int) -> int:
    """Return team slot (1..teams) on the clock for overall pick_no (1-indexed)."""
    rnd = (pick_no - 1) // teams + 1
    idx_in_round = (pick_no - 1) % teams + 1
    if draft_type == "snake" and rnd % 2 == 0:
        # reverse
        slot = teams - idx_in_round + 1
    else:
        slot = idx_in_round
    return slot

def pick_index(round_num: int, slot_in_round: int) -> int:
    """Overall pick number (1-indexed) given round and slot (1..teams)."""
    if draft_type == "snake" and round_num % 2 == 0:
        slot = teams - slot_in_round + 1
    else:
        slot = slot_in_round
    return (round_num - 1) * teams + slot

# Current on-the-clock info
next_pick_no = len(st.session_state.picks) + 1
current_round = (next_pick_no - 1) // teams + 1
current_slot = team_for_pick(next_pick_no)
is_me = (current_slot == my_slot)

info_cols = st.columns(4)
with info_cols[0]:
    st.metric("Next Pick #", next_pick_no)
with info_cols[1]:
    st.metric("Round", current_round)
with info_cols[2]:
    st.metric("Team on Clock", current_slot)
with info_cols[3]:
    st.metric("Is this me?", "✅" if is_me else "—")

# Draft input
st.subheader("Enter Pick")
colp1, colp2, colp3, colp4 = st.columns([2,1,1,1])
with colp1:
    # Select from remaining players
    # Give a shorter list focusing on value by default
    pick_pool = avail.copy().sort_values(["value_picks","adp"], ascending=[False, True], na_position="last")
    pick_names = pick_pool["player_name"].tolist()
    chosen_player = st.selectbox("Select player", options=[""] + pick_names[:500], index=0, key="player_pick_select")
with colp2:
    slot_override = st.number_input("Team slot (optional)", min_value=1, max_value=int(teams), value=int(current_slot), step=1, key="slot_override_num")
with colp3:
    round_override = st.number_input("Round (optional)", min_value=1, max_value=int(rounds), value=int(current_round), step=1, key="round_override_num")
with colp4:
    add_now = st.button("➕ Add Pick", type="primary", key="add_pick_btn")

if add_now and chosen_player:
    nk = norm_name(chosen_player)
    if nk in st.session_state.drafted_set:
        st.warning("That player is already drafted.")
    else:
        overall_pick = pick_index(round_override, slot_override)
        st.session_state.picks.append({
            "overall": overall_pick,
            "round": int(round_override),
            "slot": int(slot_override),
            "team": int(slot_override),
            "player_name": chosen_player
        })
        st.session_state.drafted_set.add(nk)
        st.success(f"Added pick {overall_pick}: {chosen_player} (Team {slot_override}, Rd {round_override})")

# Undo last pick
if st.button("↩️ Undo Last Pick", key="undo_pick_btn"):
    if st.session_state.picks:
        last = st.session_state.picks.pop()
        st.session_state.drafted_set.discard(norm_name(last["player_name"]))
        st.info(f"Removed last pick: {last['player_name']}")
    else:
        st.info("No picks to undo.")

# Render draft board grid
st.subheader("Draft Board")
# Build empty grid
grid = pd.DataFrame(index=[f"Rd {r}" for r in range(1, rounds+1)], columns=[f"Team {t}" for t in range(1, teams+1)])
for p in st.session_state.picks:
    label = f"{p['player_name']}"
    grid.loc[f"Rd {p['round']}", f"Team {p['team']}"] = label

st.dataframe(grid, width="stretch", height=min(800, 40 + 35 * int(rounds)))

# My team
st.subheader("My Team (Team %d)" % my_slot)
my_picks = [p for p in st.session_state.picks if p["team"] == int(my_slot)]
if my_picks:
    my_df = pd.DataFrame(my_picks).sort_values("overall")
    st.dataframe(my_df[["overall","round","player_name"]], width="stretch", hide_index=True)
else:
    st.caption("No picks yet for your team.")

# Export / Save
@st.cache_data(show_spinner=False)
def _to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

if st.session_state.picks:
    export = pd.DataFrame(st.session_state.picks).sort_values("overall")
    st.download_button(
        "⬇️ Download Draft Picks CSV",
        data=_to_csv_bytes(export),
        file_name=f"draft_board_{use_key}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
        key="dl_picks_btn"
    )
