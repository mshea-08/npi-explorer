"""
NPI (Net Performance Index) calculator.

Core rating engine. calculate_npi takes `init` as an explicit parameter
(instead of hardcoding the initial condition to 50.0) so callers -- like
the Streamlit app -- can control it directly.

HOME/AWAY LOCATION MULTIPLIERS (home_mult, away_mult)
-------------------------------------------------------
Added for sports where a win or loss is worth more/less depending on
where it was played (e.g. football, field hockey: home_mult=1.1,
away_mult=0.9). Sports without this concept (soccer, volleyball) just
leave these at the default of 1.0, which makes every game's location
multiplier a no-op -- so this is purely additive and doesn't change
behavior for existing sports.

The multiplier applies in two places, both driven by the same number, so
the two stay in lockstep by construction rather than by convention:
  1. It scales the game's contribution to NPI directly -- a home win
     produces a higher game-NPI value than the identical result would on
     the road (rather than only changing how that value gets averaged).
  2. It's reused as that game's WEIGHT in the min_wins capacity
     accounting and the final weighted average (replacing what was
     always a flat 1.0 for wins/losses before this feature existed) --
     so a home win also both fills the min_wins cap faster and pulls
     the final average toward it harder than an away win would.
Neutral-site games (a "neutral_site" column, if present, truthy for that
row) always get a location multiplier of 1.0 for both teams, regardless
of home_mult/away_mult.

Ties (home_result == away_result, e.g. old soccer data) are unaffected
by any of this -- their weight stays a flat 0.5 either way, since
sports with a location multiplier (football, field hockey) don't have
ties in the first place.
"""

import pandas as pd
import numpy as np


def _is_neutral(val) -> bool:
    """Robustly interprets a 'neutral_site' cell as a bool. Needed
    because this column round-trips through collector.py's plain-CSV
    writer (booleans become the literal strings "True"/"False") and
    through pandas' NaN-for-missing-value convention (bool(nan) is True
    in plain Python, which would silently make every untagged game
    'neutral' -- this function exists specifically to avoid that trap)."""
    if val is None:
        return False
    if isinstance(val, float) and pd.isna(val):
        return False
    if isinstance(val, str):
        return val.strip().lower() in ("true", "1", "yes", "y")
    return bool(val)


def calculate_npi(
    df,
    win_dial=0.2,
    sos_dial=0.8,
    qwb_mult=0.5,
    qwb_threshold=54.0,
    min_wins=8.0,
    init=50.0,
    home_mult=1.0,
    away_mult=1.0,
    max_iterations=1000,
    tolerance=1e-6,
):
    """
    Calculate Net Performance Index (NPI) for teams in a league.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: home_team, away_team, home_score, away_score.
        An optional "neutral_site" column (truthy = neutral site) disables
        home_mult/away_mult for that row -- see module docstring.
    win_dial : float, default=0.2
        Weight for win/loss/tie result
    sos_dial : float, default=0.8
        Weight for strength of schedule (opponent NPI)
    qwb_mult : float, default=0.5
        Multiplier for quality win bonus
    qwb_threshold : float, default=54.0
        NPI threshold for quality win bonus
    min_wins : float, default=8.0
        Wins-count cap used in the NPI averaging (ties count as 0.5 wins).
        Named min_wins for consistency with the NPI documentation.
    init : float or dict, default=50.0
        Initial NPI value for every team, or a dict of {team: value} to
        warm-start from a previous computation.
    home_mult : float, default=1.0
        Location multiplier applied to a team's own game-NPI value (and
        reused as that game's win/loss weight) when they were the home
        team. 1.0 is a no-op -- see module docstring. Ignored for
        neutral-site games.
    away_mult : float, default=1.0
        Same as home_mult, for the away team. Ignored for neutral-site
        games.
    max_iterations : int, default=1000
        Maximum iterations for convergence
    tolerance : float, default=1e-6
        Convergence tolerance

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: team, npi, games_played
    """
    if df.empty:
        return pd.DataFrame(columns=['team', 'npi', 'games_played'])

    teams = pd.concat([df['home_team'], df['away_team']]).unique()

    if isinstance(init, dict):
        npi_scores = {team: float(init.get(team, 50.0)) for team in teams}
    else:
        npi_scores = {team: float(init) for team in teams}

    team_game_results = {team: [] for team in teams}
    for _, game in df.iterrows():
        home = game['home_team']
        away = game['away_team']
        home_score = game['home_score']
        away_score = game['away_score']
        team_game_results[home].append(home_score - away_score)
        team_game_results[away].append(away_score - home_score)

    has_neutral_col = 'neutral_site' in df.columns

    for iteration in range(max_iterations):
        old_npi = npi_scores.copy()
        game_npis = {team: [] for team in teams}

        for _, game in df.iterrows():
            home = game['home_team']
            away = game['away_team']
            home_score = game['home_score']
            away_score = game['away_score']

            is_neutral = _is_neutral(game['neutral_site']) if has_neutral_col else False
            home_loc = 1.0 if is_neutral else home_mult
            away_loc = 1.0 if is_neutral else away_mult

            if home_score > away_score:
                home_result, away_result = 100, 0
                home_won, away_won = True, False
                home_tied = away_tied = False
            elif home_score < away_score:
                home_result, away_result = 0, 100
                home_won, away_won = False, True
                home_tied = away_tied = False
            else:
                home_result = away_result = 50
                home_won = away_won = False
                home_tied = away_tied = True

            away_npi = old_npi[away]
            home_npi = old_npi[home]

            home_qwb = 0
            away_qwb = 0
            if home_won and away_npi > qwb_threshold:
                home_qwb = (away_npi - qwb_threshold) * qwb_mult
            if away_won and home_npi > qwb_threshold:
                away_qwb = (home_npi - qwb_threshold) * qwb_mult

            if home_tied:
                home_game_npi_win = home_loc * (win_dial * 100 + sos_dial * away_npi)
                if away_npi > qwb_threshold:
                    home_game_npi_win += home_loc * (away_npi - qwb_threshold) * qwb_mult
                home_game_npi_loss = home_loc * (win_dial * 0 + sos_dial * away_npi)
                game_npis[home].append(('tie', home_game_npi_win, home_game_npi_loss, home_loc))
            else:
                home_game_npi = home_loc * (win_dial * home_result + sos_dial * away_npi + home_qwb)
                game_npis[home].append(('win' if home_won else 'loss', home_game_npi, None, home_loc))

            if away_tied:
                away_game_npi_win = away_loc * (win_dial * 100 + sos_dial * home_npi)
                if home_npi > qwb_threshold:
                    away_game_npi_win += away_loc * (home_npi - qwb_threshold) * qwb_mult
                away_game_npi_loss = away_loc * (win_dial * 0 + sos_dial * home_npi)
                game_npis[away].append(('tie', away_game_npi_win, away_game_npi_loss, away_loc))
            else:
                away_game_npi = away_loc * (win_dial * away_result + sos_dial * home_npi + away_qwb)
                game_npis[away].append(('win' if away_won else 'loss', away_game_npi, None, away_loc))

        for team in teams:
            if not game_npis[team]:
                continue

            wins_ties = []
            losses = []
            for game_data in game_npis[team]:
                result_type, npi_val, npi_loss_val, loc_mult = game_data
                if result_type == 'win':
                    wins_ties.append((npi_val, loc_mult, npi_val, None))
                elif result_type == 'tie':
                    # Ties are unaffected by location multipliers -- see
                    # module docstring (sports with a location multiplier
                    # don't have ties in the first place).
                    wins_ties.append((npi_val, 0.5, npi_val, npi_loss_val))
                else:
                    losses.append((npi_val, loc_mult))

            wins_ties.sort(key=lambda x: x[0], reverse=True)

            weighted_games = []
            win_count = 0.0
            for sort_npi, win_value, npi_win, npi_loss in wins_ties:
                remaining_capacity = min_wins - win_count
                if remaining_capacity <= 0:
                    win_weight = 1.0 if npi_win > old_npi[team] else 0.0
                elif win_value <= remaining_capacity:
                    win_weight = 1.0
                else:
                    if npi_win > old_npi[team]:
                        win_weight = 1.0
                    else:
                        win_weight = remaining_capacity / win_value

                weighted_games.append((npi_win, win_weight * win_value))
                if npi_loss is not None:
                    weighted_games.append((npi_loss, 0.5))
                win_count += win_value

            for loss_npi, loss_weight in losses:
                if loss_npi <= old_npi[team]:
                    weighted_games.append((loss_npi, loss_weight))

            if losses:
                min_loss_val = min(l[0] for l in losses)

            if weighted_games:
                total_weighted = sum(npi * weight for npi, weight in weighted_games)
                total_weight = sum(weight for _, weight in weighted_games)
                npi_scores[team] = total_weighted / total_weight if total_weight > 0 else 50.0
            else:
                npi_scores[team] = min_loss_val

        max_change = max(abs(npi_scores[team] - old_npi[team]) for team in teams)
        if max_change < tolerance:
            break

    results = []
    for team in teams:
        results.append({
            'team': team,
            'npi': npi_scores[team],
            'games_played': len(game_npis[team]),
        })

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values('npi', ascending=False).reset_index(drop=True)
    return result_df
