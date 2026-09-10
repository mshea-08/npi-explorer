import pandas as pd
import numpy as np


def _is_neutral(val) -> bool:
    """Neutral site indicator."""
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
    boost_mult=1.0,
    discount_mult=1.0,
    max_iterations=1000,
    tolerance=1e-6,
    warn_threshold=1e-3,
):
    """
    Calculate Net Performance Index (NPI) for teams in a league.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: home_team, away_team, home_score, away_score.
        An optional "neutral_site" column (truthy = neutral site) disables
        boost_mult/discount_mult for that row -- see module docstring.
    win_dial : float, default=0.2
        Weight for win/loss/tie result
    sos_dial : float, default=0.8
        Weight for strength of schedule (opponent NPI)
    qwb_mult : float, default=0.5
        Multiplier for quality win bonus
    qwb_threshold : float, default=54.0
        NPI threshold for quality win bonus
    min_wins : float, default=8.0
        Wins-count cap used in the NPI averaging.
    init : float or dict, default=50.0
        Initial NPI value for every team, or a dict of {team: value} to
        warm-start from a previous computation.
    boost_mult : float, default=1.0
        Multiplier applied to an AWAY WIN or a HOME LOSS.
    discount_mult : float, default=1.0
        Multiplier applied to a HOME WIN or an AWAY LOSS.
    max_iterations : int, default=1000
        Maximum iterations for convergence
    tolerance : float, default=1e-6
        Convergence tolerance. Unchanged/authoritative -- this is what
        actually governs the iteration loop (see `converged` below).
        Left as-is regardless of `warn_threshold`.
    warn_threshold : float, default=1e-3
        Reporting-only cutoff, does not affect the math or the iteration
        loop at all. A team can fail the strict `tolerance` check (e.g.
        because it's caught in a tiny non-decaying oscillation) without
        that being practically meaningful. `flag_for_review` uses this
        looser threshold so trivial sub-tolerance wobble isn't reported
        as something to investigate.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: team, npi, games_played, converged,
        final_diff, flag_for_review.
        "converged" is per-team: True if that team's own
        iteration-to-iteration change was below `tolerance` at the point
        the loop stopped, False if it was still moving by more than that.
        "final_diff" is that same final iteration-to-iteration change,
        as a raw number (not thresholded), so the actual size of any
        residual movement can be inspected directly.
        "flag_for_review" is True only if a team is not converged AND its
        final_diff exceeds `warn_threshold` -- i.e. movement large enough
        to plausibly matter, as opposed to microscopic oscillation that
        happens to sit above the strict `tolerance` cutoff.
        Also carries two summary values in result_df.attrs:
          "converged"  : True only if EVERY team's own column above is
                          True (i.e. the whole computation reached a
                          stable fixed point before max_iterations).
          "iterations" : how many iterations actually ran.
    """
    if df.empty:
        empty = pd.DataFrame(columns=['team', 'npi', 'games_played', 'converged', 'final_diff', 'flag_for_review'])
        empty.attrs['converged'] = True
        empty.attrs['iterations'] = 0
        return empty

    teams = pd.concat([df['home_team'], df['away_team']]).unique()

    if isinstance(init, dict):
        npi_scores = {team: float(init.get(team, 50.0)) for team in teams}
    else:
        npi_scores = {team: float(init) for team in teams}

    has_neutral_col = 'neutral_site' in df.columns

    converged = False
    iteration = 0
    team_diffs = {team: 0.0 for team in teams}
    for iteration in range(max_iterations):
        old_npi = npi_scores.copy()
        game_npis = {team: [] for team in teams}

        for _, game in df.iterrows():
            home = game['home_team']
            away = game['away_team']
            home_score = game['home_score']
            away_score = game['away_score']

            is_neutral = _is_neutral(game['neutral_site']) if has_neutral_col else False

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

            # result-dependent location multiplier
            if is_neutral or home_tied:
                home_loc = away_loc = 1.0
            else:
                home_loc = discount_mult if home_won else boost_mult
                away_loc = boost_mult if away_won else discount_mult

            away_npi = old_npi[away]
            home_npi = old_npi[home]

            home_qwb = 0
            away_qwb = 0
            if home_won and away_npi > qwb_threshold:
                home_qwb = (away_npi - qwb_threshold) * qwb_mult
            if away_won and home_npi > qwb_threshold:
                away_qwb = (home_npi - qwb_threshold) * qwb_mult

            if home_tied:
                # Ties: modeled as half a win (at the win-side value) and
                # half a loss (at the loss-side value) simultaneously.
                home_win_value = win_dial * 100 + sos_dial * away_npi
                if away_npi > qwb_threshold:
                    home_win_value += (away_npi - qwb_threshold) * qwb_mult
                home_loss_value = win_dial * 0 + sos_dial * away_npi
                game_npis[home].append(('tie', home_win_value, home_win_value, home_loss_value, home_loc))
            else:
                home_win_value = win_dial * home_result + sos_dial * away_npi + home_qwb
                game_npis[home].append(('win' if home_won else 'loss', home_win_value, home_win_value, None, home_loc))

            if away_tied:
                away_win_value = win_dial * 100 + sos_dial * home_npi
                if home_npi > qwb_threshold:
                    away_win_value += (home_npi - qwb_threshold) * qwb_mult
                away_loss_value = win_dial * 0 + sos_dial * home_npi
                game_npis[away].append(('tie', away_win_value, away_win_value, away_loss_value, away_loc))
            else:
                away_win_value = win_dial * away_result + sos_dial * home_npi + away_qwb
                game_npis[away].append(('win' if away_won else 'loss', away_win_value, away_win_value, None, away_loc))

        for team in teams:
            if not game_npis[team]:
                continue

            wins_ties = []
            losses = []
            for game_data in game_npis[team]:
                result_type, win_value, raw_value, loss_value, loc_mult = game_data
                if result_type == 'win':
                    wins_ties.append((win_value, loc_mult, raw_value, None))
                elif result_type == 'tie':
                    wins_ties.append((win_value, 0.5, raw_value, loss_value))
                else:
                    losses.append((raw_value, loc_mult))

            # Sort by win_value descending 
            wins_ties.sort(key=lambda x: x[0], reverse=True)

            weighted_games = []
            win_count = 0.0
            for win_value, loc_mult, raw_value, loss_value in wins_ties:
                remaining_capacity = min_wins - win_count
                if remaining_capacity <= 0:
                    # Overflow check uses the win_value
                    weight = 1.0 if win_value > old_npi[team] else 0.0
                elif loc_mult <= remaining_capacity:
                    weight = 1.0
                else:
                    if win_value > old_npi[team]:
                        weight = 1.0
                    else:
                        weight = remaining_capacity / loc_mult

                weighted_games.append((raw_value, weight * loc_mult))
                if loss_value is not None:
                    weighted_games.append((loss_value, 0.5))
                win_count += loc_mult

            for loss_raw_value, loss_weight in losses:
                if loss_raw_value <= old_npi[team]:
                    weighted_games.append((loss_raw_value, loss_weight))

            if losses:
                min_loss_val = min(l[0] for l in losses)

            if weighted_games:
                total_weighted = sum(val * weight for val, weight in weighted_games)
                total_weight = sum(weight for _, weight in weighted_games)
                npi_scores[team] = total_weighted / total_weight if total_weight > 0 else 50.0
            else:
                npi_scores[team] = min_loss_val

        team_diffs = {team: abs(npi_scores[team] - old_npi[team]) for team in teams}
        max_change = max(team_diffs.values())
        if max_change < tolerance:
            converged = True
            break

    results = []
    for team in teams:
        is_converged = team_diffs[team] < tolerance
        results.append({
            'team': team,
            'npi': npi_scores[team],
            'games_played': len(game_npis[team]),
            'converged': is_converged,
            'final_diff': team_diffs[team],
            'flag_for_review': (not is_converged) and (team_diffs[team] > warn_threshold),
        })

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values('npi', ascending=False).reset_index(drop=True)
    result_df.attrs['converged'] = converged
    result_df.attrs['iterations'] = iteration + 1
    return result_df
