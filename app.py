"""
D3 NPI Explorer (soccer, women's volleyball, field hockey)

Data is collected using collector.py to a file names d3_{sport}_{season}.csv. 

Up to date NPI is given for the sport. Users are also allowed to make local 
edits which include:
    - adding games
    - deleting games
    - editing results
Local edits may be reverted at any time. 
"""

import json
import os

import pandas as pd
import streamlit as st

import edits as E
from npi import calculate_npi

st.set_page_config(page_title="D3 NPI", layout="wide")

SEASONS = ["2025", "2026"]

# Sport keys 
SPORTS = ["mens_soccer", "womens_soccer", "womens_volleyball", "womens_field_hockey"]
SPORT_LABELS = {
    "mens_soccer": "Men's Soccer",
    "womens_soccer": "Women's Soccer",
    "womens_volleyball": "Women's Volleyball",
    "womens_field_hockey": "Women's Field Hockey",
}

# cache_label 
CACHE_LABELS = {
    "mens_soccer": "mens",
    "womens_soccer": "womens",
    "womens_volleyball": "womens_volleyball",
    "womens_field_hockey": "womens_field_hockey",
}

# Default NPI parameters per sport. boost_mult/discount_mult default to
# 1.0 (a no-op) for soccer/volleyball since the NCAA doesn't apply a
# location weight there, but sliders are exposed for every sport so users
# can experiment with a location multiplier regardless of sport.
DEFAULT_PARAMS = {
    "mens_soccer": dict(win_dial=0.15, qwb_mult=0.75, qwb_threshold=54.0, min_wins=10,
                         boost_mult=1.0, discount_mult=1.0),
    "womens_soccer": dict(win_dial=0.20, qwb_mult=0.50, qwb_threshold=54.0, min_wins=8,
                           boost_mult=1.0, discount_mult=1.0),
    "womens_volleyball": dict(win_dial=0.20, qwb_mult=0.60, qwb_threshold=55.5, min_wins=10,
                               boost_mult=1.0, discount_mult=1.0),
    "womens_field_hockey": dict(win_dial=0.20, qwb_mult=0.50, qwb_threshold=52.5, min_wins=6.5,
                                 boost_mult=1.1, discount_mult=0.9),
}

OUT_DIR = "."


def _mtime(path):
    return os.path.getmtime(path) if path and os.path.exists(path) else 0.0


def _team_division_cache_path(sport: str, out_dir: str) -> str:
    return os.path.join(out_dir, f"d3_team_divisions_{CACHE_LABELS[sport]}.json")


@st.cache_data
def load_known_teams(sport: str, out_dir: str, cache_mtime: float, base_mtimes: tuple) -> list:
    """
    The set of team names to offer as autocomplete options. 
    """
    cache_path = _team_division_cache_path(sport, out_dir)
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            divisions = json.load(f)
        teams = sorted(t for t, d in divisions.items() if d == "d3")
        if teams:
            return teams

    for season in SEASONS:
        base = E.load_base(season, sport, out_dir=out_dir)
        if not base.empty:
            return sorted(set(base["home_team"]) | set(base["away_team"]))

    return []


@st.cache_data
def _load_base_cached(season: str, sport: str, out_dir: str, base_mtime: float) -> pd.DataFrame:
    """Caches the read of collector.py's shared CSV output."""
    return E.load_base(season, sport, out_dir=out_dir)


def _session_edits_key(season: str, sport: str) -> str:
    return f"edits__{season}__{sport}"


def get_session_edits(season: str, sport: str) -> dict:
    """Edits for this exact browser session only, st.session_state is
    private per Streamlit session."""
    key = _session_edits_key(season, sport)
    if key not in st.session_state:
        st.session_state[key] = dict(E.EMPTY_EDITS)
    return st.session_state[key]


def set_session_edits(season: str, sport: str, edits: dict):
    st.session_state[_session_edits_key(season, sport)] = edits


def load_official_base(season: str, sport: str) -> pd.DataFrame:
    """The official, collected-by-collector.py data only."""
    base_path = E.base_csv_path(season, sport, out_dir=OUT_DIR)
    return _load_base_cached(season, sport, OUT_DIR, _mtime(base_path))


def get_effective_games(season: str, sport: str, base_df: pd.DataFrame = None) -> pd.DataFrame:
    if base_df is None:
        base_df = load_official_base(season, sport)
    edit_data = get_session_edits(season, sport)
    return E.apply_edits(base_df, edit_data)


def _to_bool_col(series: pd.Series) -> pd.Series:
    """Cleans up booleans. For the neutral site checkbox."""
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
    """Creates NPI table format."""
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
    config["neutral_site"] = st.column_config.CheckboxColumn(
        "Neutral site", default=False,
        help="Check if this game was played at a neutral site (disables the home/away NPI multiplier for it).",
    )
    return config


def _editor_column_config(sport: str, effective_df: pd.DataFrame) -> dict:
    """Searchable dropbox for team columns."""
    config = _base_column_config(sport)

    known_teams = set(load_known_teams(
        sport, OUT_DIR,
        _mtime(_team_division_cache_path(sport, OUT_DIR)),
        tuple(_mtime(E.base_csv_path(s, sport, out_dir=OUT_DIR)) for s in SEASONS),
    ))
    if not effective_df.empty:
        known_teams |= set(effective_df["home_team"].dropna()) | set(effective_df["away_team"].dropna())

    if not known_teams:
        # Nothing collected anywhere yet for this sport 
        return config

    options = sorted(known_teams)
    config["home_team"] = st.column_config.SelectboxColumn(
        "Home team", options=options, required=True,
    )
    config["away_team"] = st.column_config.SelectboxColumn(
        "Away team", options=options, required=True,
    )
    return config




def _reset_button(season: str, sport: str, label: str, key_suffix: str):
    if st.button(label, key=f"reset_{key_suffix}_{season}_{sport}"):
        set_session_edits(season, sport, dict(E.EMPTY_EDITS))
        st.toast(
            "Reset to true schedule data.",
            icon="✅",
        )
        st.rerun()


VISIBLE_COLUMNS = ["date", "home_team", "away_team", "home_score", "away_score", "neutral_site"]


def _visible_columns(sport: str) -> list:
    return VISIBLE_COLUMNS


def render_games_tab(season: str, sport: str, effective_df: pd.DataFrame, has_data: bool):
    current_edits = get_session_edits(season, sport)

    if has_data:
        st.caption(
            "Edit any game's score, delete a game (remove its row), or add a new one "
            "(use the blank row at the bottom, or the + button). **These changes are "
            "visible only to you and can be removed at any time.**"
        )
    else:
        st.caption(
            f"No {season} results exist yet. Enter games by hand as they're played. "
            "**These changes are visible only to you and can be removed at any time.**"
        )

    if E.has_edits(current_edits):
        n = len(current_edits["removed"]) + len(current_edits["overrides"]) + len(current_edits["added"])
        st.warning(f"You have {n} local change(s) applied.")
        _reset_button(
            season, sport,
            "↺ Reset to official NCAA data" if has_data else "↺ Clear all entered games",
            key_suffix="games_tab",
        )

    display_df = _prep_for_editor(effective_df)

    search = st.text_input(
        "Search by team (home or away)", key=f"search_{season}_{sport}",
        placeholder="e.g. CWRU",
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
        column_config=_editor_column_config(sport, effective_df),
        column_order=_visible_columns(sport),
        key=editor_key,
    )

    if st.button("Save changes", type="primary", key=f"save_{season}_{sport}"):
        updated_edits = E.reconcile_edit(display_df, edited_df, current_edits)
        set_session_edits(season, sport, updated_edits)
        st.toast("Saved (visible only to you).", icon="✅")
        st.rerun()


def compute_npi_context(season: str, sport: str, effective_df: pd.DataFrame, has_data: bool):
    """Renders the NPI-parameter sidebar controls (and, when has_data, the
    date-cutoff picker), then computes NPI over the resulting scope."""
    if effective_df.empty:
        return None

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

        st.markdown("---")
        st.caption(
            "Location multiplier (neutral-site games always use 1.0): "
            "boost_mult applies to an away win or a home loss; "
            "discount_mult applies to a home win "
            "or an away loss."
        )
        boost_mult = st.slider("boost_mult", 0.5, 1.5, defaults["boost_mult"], 0.01,
                                key=f"boost_mult_{season}_{sport}")
        discount_mult = st.slider("discount_mult", 0.5, 1.5, defaults["discount_mult"], 0.01,
                                   key=f"discount_mult_{season}_{sport}")

    df = effective_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if has_data:
        min_date = df["date"].dt.date.min()
        max_date = df["date"].dt.date.max()
        st.sidebar.markdown("---")
        cutoff = st.sidebar.date_input(
            "Show NPI for games through:", value=max_date,
            min_value=min_date, max_value=max_date,
        )
        df_scope = df[df["date"].dt.date <= cutoff]
        caption = (
            f"{len(df_scope)} games played through {cutoff.isoformat()} "
            f"(season data spans {min_date} to {max_date})"
        )
    else:
        df_scope = df
        caption = f"Current standings from {len(df_scope)} entered game(s)."

    if df_scope.empty:
        return None

    result = calculate_npi(
        df_scope, win_dial=win_dial, sos_dial=sos_dial, qwb_mult=qwb_mult,
        qwb_threshold=qwb_threshold, min_wins=min_wins, init=init,
        boost_mult=boost_mult, discount_mult=discount_mult,
    )

    if not result.attrs.get("converged", True):
        st.warning(
            "⚠️ **NPI did not converge for this set of games.** This typically happens early in a "
            "season, when most teams have only played 1-2 games. Please come back later or add results."
        )

    return {
        "result": result,
        "df_scope": df_scope,
        "caption": caption,
        "converged": result.attrs.get("converged", True),
    }


def render_rankings_tab(season: str, sport: str, effective_df: pd.DataFrame, has_data: bool, ctx: dict):
    current_edits = get_session_edits(season, sport)
    if E.has_edits(current_edits):
        st.warning(
            "⚠️ These rankings include your local edits "
            "-- not the official NCAA results."
        )
        _reset_button(
            season, sport,
            "↺ Reset to official NCAA data" if has_data else "↺ Clear all entered games",
            key_suffix="rankings_tab",
        )

    if effective_df.empty:
        st.info(
            "No games yet."
            + (" Run collector.py, or add games in the Edit Games tab." if has_data
               else " Enter games in the Add Games tab.")
        )
        return

    if ctx is None:
        st.info("No games played by the selected date yet.")
        return

    if not ctx["converged"]:
        st.info(
            "Rankings are hidden, NPI did not converge."
        )
        return

    st.caption(ctx["caption"])

    result_display = ctx["result"].copy()
    result_display.insert(0, "rank", range(1, len(result_display) + 1))
    result_display["npi"] = result_display["npi"].round(2)

    team_filter = st.text_input("Filter by team name (optional)", key=f"filter_{season}_{sport}")
    if team_filter:
        result_display = result_display[
            result_display["team"].str.contains(team_filter, case=False, na=False)
        ]

    st.dataframe(result_display, width="stretch", hide_index=True)


def render_team_lookup_tab(season: str, sport: str, effective_df: pd.DataFrame, has_data: bool, ctx: dict):
    if effective_df.empty:
        st.info(
            "No games yet."
            + (" Run collector.py, or add games in the Edit Games tab." if has_data
               else " Enter games in the Add Games tab.")
        )
        return

    if ctx is None:
        st.info("No games played by the selected date yet.")
        return

    if not ctx["converged"]:
        st.info(
            "Team lookup is hidden until NPI converges for this data -- see the "
            "warning above. Check back once more games have been collected."
        )
        return

    result = ctx["result"]
    df_scope = ctx["df_scope"]
    has_neutral_col = "neutral_site" in df_scope.columns

    all_teams = sorted(set(df_scope["home_team"]) | set(df_scope["away_team"]))
    team = st.selectbox(
        "Search for a team", options=all_teams, index=None,
        placeholder="Start typing a team name...",
        key=f"lookup_team_{season}_{sport}",
    )
    if not team:
        st.caption("Pick a team above to see its full schedule, results, and each opponent's NPI.")
        return

    npi_lookup = result.set_index("team")["npi"]
    team_row = result[result["team"] == team]
    if not team_row.empty:
        rank = int(result.index[result["team"] == team][0]) + 1
        st.metric(f"{team} — NPI", f"{team_row.iloc[0]['npi']:.2f}", help=f"Rank #{rank} of {len(result)}")

    team_games = df_scope[(df_scope["home_team"] == team) | (df_scope["away_team"] == team)].copy()
    if has_neutral_col:
        team_games["neutral_site"] = _to_bool_col(team_games["neutral_site"])

    rows = []
    for _, g in team_games.sort_values("date").iterrows():
        is_home = g["home_team"] == team
        opponent = g["away_team"] if is_home else g["home_team"]
        team_score = g["home_score"] if is_home else g["away_score"]
        opp_score = g["away_score"] if is_home else g["home_score"]

        scores_known = pd.notna(team_score) and pd.notna(opp_score)
        if not scores_known:
            outcome, score_str = "?", ""
        elif team_score > opp_score:
            outcome, score_str = "W", f"{int(team_score)}-{int(opp_score)}"
        elif team_score < opp_score:
            outcome, score_str = "L", f"{int(team_score)}-{int(opp_score)}"
        else:
            outcome, score_str = "T", f"{int(team_score)}-{int(opp_score)}"

        if has_neutral_col and g.get("neutral_site"):
            location = "Neutral"
        else:
            location = "Home" if is_home else "Away"

        rows.append({
            "date": g["date"].date() if pd.notna(g["date"]) else None,
            "location": location,
            "opponent": opponent,
            "result": outcome,
            "score": score_str,
            "opponent_npi": round(npi_lookup[opponent], 2) if opponent in npi_lookup.index else None,
        })

    if not rows:
        st.info(f"No games found for {team} in the selected date range.")
        return

    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def render_introduction():
    st.markdown(
        """
<style>
.npi-hl { color: #5EEAD4; text-decoration: none; }
a.npi-hl:hover { text-decoration: underline; }
.npi-hl-pink { color: #F472B6; font-weight: 600; }
</style>

<h1>Welcome!</h1>

<p>Hi there! This website was created to help NCAA D3 coaches and fans get
consistent and manipulatable NPI information throughout the season. For the
fall season I will be supporting men's soccer, women's soccer, women's
volleyball, and field hockey. Before you begin, please read through the
<a class="npi-hl" href="#getting-started">Getting Started</a>,
<a class="npi-hl" href="#accuracy-disclaimer">Accuracy Disclaimer</a>, and
<a class="npi-hl" href="#contact">Contact</a> sections.</p>

<p>For the 24/25 and 25/26 seasons, I was able to work with the Vassar
Women's Soccer coaching staff in an analytics capacity. With NPI being
newly implemented in 2024, one common conversation was trying to
understand how games (both our own games and games of conference
opponents) and scheduling would impact our NPI. We had <em>a lot</em> of
questions. So in winter 25/26 I recreated the NCAA's NPI algorithm for
women's soccer. We (s/o Prof Deford, Department of Mathematics and
Statistics, Vassar College) spent some time generating fake schedules and
had students test out their impact on NPI rankings. We also spent
sometimes asking questions about the general convergence of the
algorithm.</p>

<p><strong>Now the goal is transparency for teams and coaches.</strong>
No more guessing how it is going until October. Want to know how much
your next game could improve your score? Want to know if playing team X
instead of team Y would have helped? Test it out!</p>

<h2 id="getting-started">Getting Started</h2>

<p>On the Navigation bar, navigate over to <span class="npi-hl">NPI
Explorer</span> to begin. From there you can select your sport to see
the NPI rankings. Please give the website a few minutes to load. You can
go to the <span class="npi-hl">Edit Games</span> tab to add games,
delete games, or change any result. If you are adding a game, please
make sure you fill in all fields (including the date). To reload the NPI
rankings with your changes, toggle back to the Rankings tab. From there
you can also clear any changes made.</p>

<p>You can also search for a single team using the
<span class="npi-hl">Team Lookup</span>. From there you can see all the
games they have played as well as the NPIs for their opponents. Please
note that the games do not automatically update, so this is a great
place to check if a game has been included or not. I will <em>try</em>
to update the game files every couple days.</p>

<p>For the real sickos, you can use the sidebar to modify the NPI
parameters. Bonus points if you break convergence. Early on in the
season the NPI algorithm might not converge (and it will tell you so)!
A low number of games played can easily cause oscillations which should
go away after each team has a few games under their belts.</p>

<h2 id="accuracy-disclaimer">Accuracy Disclaimer</h2>

<p><strong>These are not official NCAA results!!</strong> While I've
done a ton of testing on the 25/26 seasons and feel confident about the
algorithm, there are a lot of little reasons why the numbers presented
here can vary from the official results. Please think of this as
<strong>approximate NPI</strong>.</p>

<p>For field hockey I <span class="npi-hl-pink">know</span> the numbers
are not exact. This is because the database where I source my games
from does not accurately denote neutral site games and I have not found
an easy fix for this. Based on testing on the 25/26 season, I expect
neutral games will only cause an NPI error of at most approximately 0.2,
although this margin of error could be larger at the beginning of the
season (it appears that most neutral site games happen during the
conference tournaments, though).</p>

<h2 id="contact">Contact</h2>

<p>You can contact me, Meredith Shea, at
<a class="npi-hl" href="mailto:mshea@plu.edu">mshea@plu.edu</a> with
suggestions, questions, or errors. In particular, if you notice errors
in the games databases please let me know! I simply do not have the
capacity to check all the games. If you are a field hockey person and
would like to notify me of neutral site games I am also happy to put
those in the database&mdash;please send me the teams and date of the
game.</p>
""",
        unsafe_allow_html=True,
    )





def main():
    page = st.sidebar.radio("Page", ["Introduction", "NPI Explorer"])
    st.sidebar.markdown("---")

    if page == "Introduction":
        render_introduction()
        return

    st.title("NPI Explorer")
    st.warning(
        "⚠️ **These are not official NCAA rankings.** This is an independent, "
        "unofficial recreation of the NCAA's NPI algorithm and may differ from "
        "the NCAA's own published results."
    )

    season = st.sidebar.radio("Season", SEASONS)
    sport = st.sidebar.radio("Sport", SPORTS, format_func=lambda s: SPORT_LABELS[s])

    if sport == "womens_field_hockey":
        st.warning(
            "⚠️ **Neutral-site games are not addressed for field hockey.** "
            "collector.py cannot detect whether a game was actually played at a "
            "neutral site, so every game is scored with a real home/away split "
            "unless manually corrected via the \"Neutral site\" checkbox in the "
            "Edit/Add Games tab. Since field hockey's boost/discount defaults "
            "away from 1.0, this will cause a small margin of error in scores "
            "for any team with an unflagged neutral-site game."
        )
    else:
        st.info(
            "ℹ️ Neutral-site games can't be auto-detected for any sport (see the "
            "\"Neutral site\" checkbox in the Edit/Add Games tab) -- this only "
            "affects your numbers here if you adjust boost_mult/discount_mult "
            "away from their 1.0 (no-op) defaults in the NPI parameters sidebar."
        )

    base_df = load_official_base(season, sport)
    has_data = not base_df.empty
    effective_df = get_effective_games(season, sport, base_df=base_df)

    games_tab_label = "Edit Games" if has_data else "Add Games"
    tab_rankings, tab_games, tab_lookup = st.tabs(["Rankings", games_tab_label, "Team Lookup"])

    # Computed once here (not inside either tab) so the NPI-parameter
    # sidebar widgets render exactly once per rerun, and both tabs below
    # see the identical result/date-scope.
    ctx = compute_npi_context(season, sport, effective_df, has_data)

    with tab_rankings:
        render_rankings_tab(season, sport, effective_df, has_data, ctx)
    with tab_games:
        render_games_tab(season, sport, effective_df, has_data)
    with tab_lookup:
        render_team_lookup_tab(season, sport, effective_df, has_data, ctx)


if __name__ == "__main__":
    main()
