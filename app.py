"""
D3 NPI Explorer (soccer + women's volleyball)

Run with:
    streamlit run app.py

2025 season: reads d3_{sport}_2025.csv (produced by collector.py, where
sport is a key like "mens_soccer" or "womens_volleyball" -- see
collector.py's SPORT_CONFIG) as a shared base, and layers each viewer's
own edits on top
(score corrections, added games, removed games -- see edits.py) to show
NPI as of a chosen date.

2026 season: no base data exists yet (season hasn't happened). Same
mechanism, just starting from an empty base -- so "editing" is really
"entering games by hand." Standings always reflect every game entered so
far; there's no date picker since there's no collected season-long
calendar to filter against yet.

EDITS ARE PER-BROWSER-SESSION, NOT SHARED.
--------------------------------------------
Everything a user adds, changes, or deletes is stored in
st.session_state, which Streamlit keeps private and separate for every
person connected to the app. Nobody's edits are ever visible to anyone
else, and nothing gets written back to the underlying CSVs -- the only
shared, on-disk data is collector.py's output, which this app only ever
reads, never writes. Closing the tab / starting a new session clears
edits entirely; there's also an explicit "Reset to official NCAA data"
button for doing that on demand without leaving the page.
"""

import json
import os

import pandas as pd
import streamlit as st

import edits as E
from npi import calculate_npi

st.set_page_config(page_title="D3 NPI", layout="wide")

SEASONS = ["2025", "2026"]

# Sport keys match collector.py's SPORT_CONFIG exactly -- they double as
# the CSV file label (d3_{sport}_{season}.csv), so app.py, edits.py, and
# collector.py all agree on file names without any separate translation.
#
# NOTE: collector.py, edits.py, and npi.py all already have full support
# for "womens_field_hockey" (including home/away NPI multipliers) -- it's
# deliberately left out of the lists below because it's still a
# work-in-progress (neutral-site detection isn't automatable -- see
# collector.py's _parse_game docstring) and shouldn't be user-facing yet.
# To turn it back on later: add "womens_field_hockey" back to SPORTS,
# SPORT_LABELS, CACHE_LABELS, LOCATION_MULT_SPORTS, and DEFAULT_PARAMS.
SPORTS = ["mens_soccer", "womens_soccer", "womens_volleyball"]
SPORT_LABELS = {
    "mens_soccer": "Men's Soccer",
    "womens_soccer": "Women's Soccer",
    "womens_volleyball": "Women's Volleyball",
}

# cache_label mirrors collector.py's SPORT_CONFIG[...]["cache_label"] --
# kept as the original "mens"/"womens" for the two soccer sports so
# existing division caches / known-teams files aren't invalidated, and
# given its own label for each new sport added after the fact.
CACHE_LABELS = {
    "mens_soccer": "mens",
    "womens_soccer": "womens",
    "womens_volleyball": "womens_volleyball",
}

# Sports that have a location (home/away) multiplier at all. Sliders for
# home_mult/away_mult only render for sports listed here -- see
# render_rankings_tab. Everything else implicitly uses the npi.py default
# of 1.0 for both (a no-op).
LOCATION_MULT_SPORTS = set()

# Default NPI parameters per sport. Sliders below start here.
DEFAULT_PARAMS = {
    "mens_soccer": dict(win_dial=0.15, qwb_mult=0.75, qwb_threshold=54.0, min_wins=10),
    "womens_soccer": dict(win_dial=0.20, qwb_mult=0.50, qwb_threshold=54.0, min_wins=8),
    "womens_volleyball": dict(win_dial=0.20, qwb_mult=0.60, qwb_threshold=55.5, min_wins=10),
}

OUT_DIR = "."


def _mtime(path):
    return os.path.getmtime(path) if path and os.path.exists(path) else 0.0


def _team_division_cache_path(sport: str, out_dir: str) -> str:
    return os.path.join(out_dir, f"d3_team_divisions_{CACHE_LABELS[sport]}.json")


@st.cache_data
def load_known_teams(sport: str, out_dir: str, cache_mtime: float, base_mtime: float) -> list:
    """
    The set of team names to offer as autocomplete options when adding a
    2026 game. Prefers collector.py's division cache (authoritative, D3
    teams only, from d3_team_divisions_{men,women}.json); falls back to
    whatever team names appear in the 2025 base CSV if that cache doesn't
    exist yet; returns an empty list if neither is available (e.g. a
    completely fresh install with no 2025 data collected at all).
    """
    cache_path = _team_division_cache_path(sport, out_dir)
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            divisions = json.load(f)
        teams = sorted(t for t, d in divisions.items() if d == "d3")
        if teams:
            return teams

    base_2025 = E.load_base("2025", sport, out_dir=out_dir)
    if not base_2025.empty:
        teams = sorted(set(base_2025["home_team"]) | set(base_2025["away_team"]))
        return teams

    return []


@st.cache_data
def _load_base_cached(season: str, sport: str, out_dir: str, base_mtime: float) -> pd.DataFrame:
    """Caches the read of collector.py's shared CSV output. This is safe
    to cache globally (not per-session) because it's read-only, identical
    for everyone, and the mtime in the cache key means a fresh
    collector.py run is picked up automatically."""
    return E.load_base(season, sport, out_dir=out_dir)


def _session_edits_key(season: str, sport: str) -> str:
    return f"edits__{season}__{sport}"


def get_session_edits(season: str, sport: str) -> dict:
    """Edits for this exact browser session only -- st.session_state is
    private per Streamlit session, so this can never leak between users."""
    key = _session_edits_key(season, sport)
    if key not in st.session_state:
        st.session_state[key] = dict(E.EMPTY_EDITS)
    return st.session_state[key]


def set_session_edits(season: str, sport: str, edits: dict):
    st.session_state[_session_edits_key(season, sport)] = edits


def get_effective_games(season: str, sport: str) -> pd.DataFrame:
    base_path = E.base_csv_path(season, sport, out_dir=OUT_DIR)
    base_df = _load_base_cached(season, sport, OUT_DIR, _mtime(base_path))
    edit_data = get_session_edits(season, sport)
    return E.apply_edits(base_df, edit_data)


def _to_bool_col(series: pd.Series) -> pd.Series:
    """Same string/NaN robustness as npi._is_neutral, for the UI's
    checkbox column -- CSV round-trips through collector.py turn True/
    False into the literal strings "True"/"False", and bool("False") is
    True in plain Python, so a naive .astype(bool) would silently invert
    every unset/false row."""
    def _conv(v):
        if isinstance(v, str):
            return v.strip().lower() in ("true", "1", "yes", "y")
        if pd.isna(v):
            return False
        return bool(v)
    return series.map(_conv)


def _prep_for_editor(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce dtypes so st.data_editor's column_config renders correctly
    (real date objects for the date picker column, real ints for scores)."""
    out = df.copy()
    if out.empty:
        return pd.DataFrame(columns=E.FIELDNAMES)
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date
    out["home_score"] = pd.to_numeric(out["home_score"], errors="coerce").astype("Int64")
    out["away_score"] = pd.to_numeric(out["away_score"], errors="coerce").astype("Int64")
    out["neutral_site"] = _to_bool_col(out["neutral_site"])
    return out[E.FIELDNAMES]


def _base_column_config(sport: str = None) -> dict:
    # Underlying field names stay home_score/away_score for every sport
    # (npi.py and edits.py are sport-agnostic and just need two comparable
    # numbers) -- only the on-screen label changes for volleyball, where
    # that number is sets won rather than goals.
    score_label = "sets" if sport == "womens_volleyball" else "score"
    config = {
        "date": st.column_config.DateColumn("Date", required=True),
        "home_team": st.column_config.TextColumn("Home team", required=True),
        "away_team": st.column_config.TextColumn("Away team", required=True),
        "home_score": st.column_config.NumberColumn(f"Home {score_label}", min_value=0, step=1, required=True),
        "away_score": st.column_config.NumberColumn(f"Away {score_label}", min_value=0, step=1, required=True),
        "status": st.column_config.TextColumn("Status", disabled=True),
        "game_id": st.column_config.TextColumn("id", disabled=True, width="small"),
    }
    if sport in LOCATION_MULT_SPORTS:
        config["neutral_site"] = st.column_config.CheckboxColumn(
            "Neutral site", default=False,
            help="Check if this game was played at a neutral site (disables the home/away NPI multiplier for it).",
        )
    return config


def _editor_column_config(season: str, sport: str) -> dict:
    """2025 keeps free-text team entry (the collector already gets
    spelling right; edits here are usually score corrections). 2026 gets
    a searchable dropdown of known D3 teams instead of free text, so
    entering a game is 'type to filter, then pick' rather than typing a
    name from scratch and risking a typo that would silently create a
    duplicate 'team'."""
    config = _base_column_config(sport)
    if season != "2026":
        return config

    known_teams = load_known_teams(
        sport, OUT_DIR,
        _mtime(_team_division_cache_path(sport, OUT_DIR)),
        _mtime(E.base_csv_path("2025", sport, out_dir=OUT_DIR)),
    )
    if not known_teams:
        # No 2025 data collected yet anywhere -- nothing to offer as
        # options, so fall back to free text rather than an empty,
        # unusable dropdown.
        return config

    config["home_team"] = st.column_config.SelectboxColumn(
        "Home team", options=known_teams, required=True,
    )
    config["away_team"] = st.column_config.SelectboxColumn(
        "Away team", options=known_teams, required=True,
    )
    return config




def _reset_button(season: str, sport: str, label: str, key_suffix: str):
    if st.button(label, key=f"reset_{key_suffix}_{season}_{sport}"):
        set_session_edits(season, sport, dict(E.EMPTY_EDITS))
        st.toast(
            "Reset to official NCAA data." if season == "2025" else "All entered games cleared.",
            icon="✅",
        )
        st.rerun()


VISIBLE_COLUMNS = ["date", "home_team", "away_team", "home_score", "away_score"]


def _visible_columns(sport: str) -> list:
    if sport in LOCATION_MULT_SPORTS:
        return VISIBLE_COLUMNS + ["neutral_site"]
    return VISIBLE_COLUMNS


def render_games_tab(season: str, sport: str, effective_df: pd.DataFrame):
    is_2025 = season == "2025"
    current_edits = get_session_edits(season, sport)

    if is_2025:
        st.caption(
            "Edit any game's score, delete a game (remove its row), or add a new one "
            "(use the blank row at the bottom, or the + button). **These changes are "
            "visible only to you** -- they're kept in your browser session, never saved "
            "to shared data, and never affect what anyone else sees."
        )
    else:
        st.caption(
            "No 2026 results exist yet -- enter games by hand as they're played. "
            "Add a row for each game (date, teams, final score); standings on the "
            "Rankings tab always reflect everything entered here. **This is visible only "
            "to you**, kept in your browser session."
        )

    if E.has_edits(current_edits):
        n = len(current_edits["removed"]) + len(current_edits["overrides"]) + len(current_edits["added"])
        st.warning(f"You have {n} local change(s) applied, visible only to you.")
        _reset_button(
            season, sport,
            "↺ Reset to official NCAA data" if is_2025 else "↺ Clear all entered games",
            key_suffix="games_tab",
        )

    display_df = _prep_for_editor(effective_df)

    search = st.text_input(
        "Search by team (home or away)", key=f"search_{season}_{sport}",
        placeholder="e.g. Amherst",
    )
    if search:
        mask = (
            display_df["home_team"].str.contains(search, case=False, na=False)
            | display_df["away_team"].str.contains(search, case=False, na=False)
        )
        display_df = display_df[mask].reset_index(drop=True)
        st.caption(
            f"Showing {len(display_df)} game(s) matching \u201c{search}\u201d. "
            "Save before changing the search box, or unsaved edits to rows outside "
            "the current search will be lost when the view refreshes."
        )

    editor_key = f"editor_{season}_{sport}_{search}"

    edited_df = st.data_editor(
        display_df,
        num_rows="dynamic",
        width="stretch",
        column_config=_editor_column_config(season, sport),
        column_order=_visible_columns(sport),
        key=editor_key,
    )

    if st.button("Save changes", type="primary", key=f"save_{season}_{sport}"):
        updated_edits = E.reconcile_edit(display_df, edited_df, current_edits)
        set_session_edits(season, sport, updated_edits)
        st.toast("Saved (visible only to you).", icon="✅")
        st.rerun()


def render_rankings_tab(season: str, sport: str, effective_df: pd.DataFrame):
    current_edits = get_session_edits(season, sport)
    if E.has_edits(current_edits):
        st.warning(
            "⚠️ These rankings include your local edits and are visible only to you "
            "-- not the official NCAA results."
        )
        _reset_button(
            season, sport,
            "↺ Reset to official NCAA data" if season == "2025" else "↺ Clear all entered games",
            key_suffix="rankings_tab",
        )

    if effective_df.empty:
        st.info(
            "No games yet."
            + (" Run collector.py, or add games in the Edit Games tab." if season == "2025"
               else " Enter games in the Add Games tab.")
        )
        return

    defaults = DEFAULT_PARAMS[sport]
    with st.sidebar.expander("NPI parameters", expanded=False):
        win_dial = st.slider("win_dial", 0.0, 1.0, defaults["win_dial"], 0.01,
                              key=f"win_dial_{season}_{sport}")
        sos_dial = 1.0 - win_dial
        st.slider("sos_dial", 0.0, 1.0, sos_dial, 0.01, disabled=True,
                  help="Locked to 1 - win_dial", key=f"sos_dial_{season}_{sport}")
        qwb_mult = st.slider("qwb_mult", 0.0, 1.0, defaults["qwb_mult"], 0.01,
                              key=f"qwb_mult_{season}_{sport}")
        qwb_threshold = st.slider("qwb_threshold", 0.0, 100.0, defaults["qwb_threshold"], 0.5,
                                   key=f"qwb_threshold_{season}_{sport}")
        min_wins = st.slider("min_wins", 1.0, 20.0, float(defaults["min_wins"]), 0.5,
                              key=f"min_wins_{season}_{sport}")
        init = st.number_input("initial NPI", value=50.0, step=1.0, key=f"init_{season}_{sport}")

        if sport in LOCATION_MULT_SPORTS:
            st.markdown("---")
            st.caption("Home/away location multipliers (neutral-site games always use 1.0):")
            home_mult = st.slider("home_mult", 0.5, 1.5, defaults["home_mult"], 0.01,
                                   key=f"home_mult_{season}_{sport}")
            away_mult = st.slider("away_mult", 0.5, 1.5, defaults["away_mult"], 0.01,
                                   key=f"away_mult_{season}_{sport}")
        else:
            home_mult = away_mult = 1.0

    df = effective_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if season == "2025":
        min_date = df["date"].dt.date.min()
        max_date = df["date"].dt.date.max()
        st.sidebar.markdown("---")
        cutoff = st.sidebar.date_input(
            "Show NPI for games through:", value=max_date,
            min_value=min_date, max_value=max_date,
        )
        df_scope = df[df["date"].dt.date <= cutoff]
        st.caption(
            f"{len(df_scope)} games played through {cutoff.isoformat()} "
            f"(season data spans {min_date} to {max_date})"
        )
    else:
        df_scope = df
        st.caption(f"Current standings from {len(df_scope)} entered game(s).")

    if df_scope.empty:
        st.info("No games played by the selected date yet.")
        return

    result = calculate_npi(
        df_scope, win_dial=win_dial, sos_dial=sos_dial, qwb_mult=qwb_mult,
        qwb_threshold=qwb_threshold, min_wins=min_wins, init=init,
        home_mult=home_mult, away_mult=away_mult,
    )

    result_display = result.copy()
    result_display.insert(0, "rank", range(1, len(result_display) + 1))
    result_display["npi"] = result_display["npi"].round(2)

    team_filter = st.text_input("Filter by team name (optional)", key=f"filter_{season}_{sport}")
    if team_filter:
        result_display = result_display[
            result_display["team"].str.contains(team_filter, case=False, na=False)
        ]

    st.dataframe(result_display, width="stretch", hide_index=True)


def render_introduction():
    blah1 = '''# Introduction
    
This app provides a place to view/manipulate current and past D3 NPI rankings. Currently, D3 men's soccer, women's soccer, and women's volleyball are supported. 

## Getting Started
    
To begin, select the NPI Explorer page on the sidebar. From there you can choose the season and the sport. 
        
For prior seasons, the NPI ranking up to selection day is automatically loaded. There are no results loaded past selection day. On the sidebar under 'Show NPI for games through:' you can modify the dates included. The **Edit Games** tab allows you to add fake games, delete games, or change the score to any existing game. Any changes can be reset later. 
    
For the current season, the most up to date NPI rankings are loaded (I will try to update at least weekly). The **Add Games** tab allows you to add games in (but you cannot delete or edit games for the current season), this resource is meant to help you look at how your next result or another conference result may affect your standings. 

For those curious, the NPI Parameters sidebar also allows you to modify the current NPI parameters to those of your choosing. 

## NPI Accuracy

**These are not official NCAA results!** The NCAA provides limited and non-rigorous documentation regarding their NPI algorithm. My version of the algorithm was developed over the 2025-26 winter using the 2025 fall seasons. For the 2025 season the algorithm is accurate up to what I consider rounding error. As you inspect the current season results, please understand that there may be differences between these results and the official NCAA results. 

### More Details 

Ignoring the quality win bonus and the minimum wins threshold, the NPI algorithm is a straight forward linear model which we can compute the fixed point of using iteration. For the full algorithm, we still use iteration to compute the results however general convergence of the model is not explicitly clear (as it is no longer linear). Through testing we found multiple cases where convergence fails. 

Of subtle importance to this is the initial condition set for the iteration. For our version of the algorithm, we set all team initial NPIs to 50 (originally, we tried 0 and found that we could not match the 2025 results). For those interested in inquiring more about the algorithm implementation and validity, see the contact information below. 

## Front Matter

All code for this app, including the NPI algorithm, can be found on GitHub (mshea-08). This code is licensed under an MIT license, meaning you are free to use, copy, or modify this work as long as the copyright remains. 

The NPI algorithm was written by me, Meredith Shea, the supporting code for the applet was developed using Claude. 

## Contact

If you have questions, suggestions, or find any errors, you can email me at mshea@plu.edu. 

'''
    
    st.markdown(blah1)



def main():
    page = st.sidebar.radio("Page", ["Introduction", "NPI Explorer"])
    st.sidebar.markdown("---")

    if page == "Introduction":
        render_introduction()
        return

    st.title("NPI Explorer")

    season = st.sidebar.radio("Season", SEASONS)
    sport = st.sidebar.radio("Sport", SPORTS, format_func=lambda s: SPORT_LABELS[s])

    effective_df = get_effective_games(season, sport)

    games_tab_label = "Edit Games" if season == "2025" else "Add Games"
    tab_rankings, tab_games = st.tabs(["Rankings", games_tab_label])

    with tab_rankings:
        render_rankings_tab(season, sport, effective_df)
    with tab_games:
        render_games_tab(season, sport, effective_df)


if __name__ == "__main__":
    main()
