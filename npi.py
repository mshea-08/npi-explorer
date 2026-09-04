"""
NPI (Net Performance Index) calculator.

Core rating engine. calculate_npi takes `init` as an explicit parameter
(instead of hardcoding the initial condition to 50.0) so callers -- like
the Streamlit app -- can control it directly.

TERMINOLOGY -- matches the NCAA's own published team-detail pages:
  - win_value / loss_value : the RAW value of a game, BEFORE the location
    multiplier is applied. win_value = win_dial*100 + sos_dial*opponent_npi
    + QWB. loss_value = sos_dial*opponent_npi (no QWB on a loss).
  - loc_mult : the location multiplier for this specific game (1.0 for a
    neutral-site game, else boost_mult or discount_mult -- see below).
  - net_npi (displayed on NCAA's site, NOT used internally here) :
    loc_mult * win_value (or loc_mult * loss_value). This is a DISPLAY
    quantity only -- see the critical note below.

*** CRITICAL: loc_mult is applied EXACTLY ONCE, as the WEIGHT in the
team's weighted average -- NOT baked into the averaged value itself. ***
The value that gets averaged is the RAW win_value/loss_value. NCAA's own
site displays "Net npi" = loc_mult * win_value for each game (confirmed
correct as a DISPLAY figure, many times over, against real opponent-detail
pages), which made it easy to wrongly conclude that Net npi is also what
should be averaged with loc_mult as the weight -- i.e. applying loc_mult
TWICE (once into the value, once again as the weight). That double
application is exactly the bug that caused a large systemic downward
(or, in an earlier broken variant, runaway upward) bias across an entire
league computation, and separately caused the iteration to oscillate
indefinitely instead of converging -- both symptoms of the same
over-amplified feedback loop, since a team's own (over- or under-scaled)
NPI feeds forward every iteration as its opponents' strength-of-schedule
input. Confirmed by solving for the exact formula that reproduces real
opponent-detail data exactly (e.g. Ohio Wesleyan: correct formula gives
48.0889 against a true value of 48.089; Chris. Newport: 91.1636 against
91.163) where the double-multiplied version was off by several points
and never converged at full-season scale.

HOME/AWAY LOCATION MULTIPLIERS (boost_mult, discount_mult)
-------------------------------------------------------------
Per the NCAA's own NPI Weights Guide (D3CC_NPIWeights.pdf), a sport's
location weight isn't literally "a home number and an away number" --
it's a BOOST applied to the "notable" outcomes (an away win, or a home
loss) and a DISCOUNT applied to the "expected" outcomes (a home win, or
an away loss). Quoting the guide directly: "Away wins/home losses weigh
as 1.2 wins/losses. Home wins/away losses weigh as 0.8 wins/losses" (for
a 1.2/0.8 setting).

For field hockey (1.1/0.9): boost_mult=1.1, discount_mult=0.9.

Sports without this concept (soccer, volleyball) leave both at the
default of 1.0, a no-op -- and since raw_value == net_npi whenever
loc_mult is 1.0, the double-application bug above was invisible for
these sports, which is why it went undetected for so long.

Neutral-site games (a "neutral_site" column, if present, truthy for that
row) always get loc_mult=1.0 for both teams.

KNOWN LIMITATION -- collector.py cannot detect real-world neutral-site
games; every game defaults to neutral_site=False unless manually
corrected. Confirmed cases (via NCAA's own opponent-detail pages) show
preseason kickoff/showcase and some conference-tournament games get
scored as a genuine home/away split in our data even when they were
actually played at a neutral site. Several have been manually corrected
in the current dataset; residual per-team error on the order of a few
tenths of a point is consistent with remaining undetected cases (or
simple rounding in the published, 3-decimal NPI figures) rather than a
remaining formula bug -- full-season validation against 163 real teams
now shows mean absolute error of 0.03 and max error of 0.16.

THE WIN-CAPACITY WALK (min_wins)
-------------------------------------------------------------
Wins (and ties) are sorted by win_value (RAW, pre-multiplier -- equiv.
to sorting by the opponent's own NPI) in descending order -- confirmed
against the NCAA's own opponent-detail page row ordering. Each win then
consumes loc_mult worth of the min_wins capacity:
  - If there's enough capacity left, the win counts in full.
  - If a win doesn't fit in the remaining capacity, but its own
    win_value (RAW) exceeds the team's own current NPI, it still counts
    in full ("overflow"). This is the mathematically meaningful check:
    since a team's own NPI is itself a weighted average, a candidate
    game raises that average if and only if the value being added
    (win_value) exceeds the current average.
  - Otherwise, if there's still some capacity left, the win is diluted
    down to exactly the capacity remaining.
  - Once capacity is fully exhausted, only the overflow check applies;
    anything that doesn't clear it is excluded entirely (weight 0).
Losses are included in the final average (at their own loc_mult weight)
only if their own RAW loss_value doesn't exceed the team's current NPI
-- i.e. only losses that would actually pull the average down get
counted. (This loss-side check must also use the RAW value, not a
loc-multiplied one -- confirmed against Ohio Wesleyan's Mary Washington
game, which sits right at this exact boundary.)
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
    boost_mult=1.0,
    discount_mult=1.0,
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
        Wins-count cap used in the NPI averaging (ties count as 0.5 wins).
        Named min_wins for consistency with the NPI documentation.
    init : float or dict, default=50.0
        Initial NPI value for every team, or a dict of {team: value} to
        warm-start from a previous computation.
    boost_mult : float, default=1.0
        Multiplier applied to an AWAY WIN or a HOME LOSS -- the
        "notable" outcomes. 1.0 is a no-op. Ignored for neutral-site
        games. See module docstring.
    discount_mult : float, default=1.0
        Multiplier applied to a HOME WIN or an AWAY LOSS -- the
        "expected" outcomes. 1.0 is a no-op. Ignored for neutral-site
        games. See module docstring.
    max_iterations : int, default=1000
        Maximum iterations for convergence
    tolerance : float, default=1e-6
        Convergence tolerance

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: team, npi, games_played. Also carries two
        diagnostic values in result_df.attrs (doesn't change the columns
        or break any existing caller that only looks at the columns):
          "converged"  : True if max_change dropped below `tolerance`
                          before max_iterations was reached, else False.
          "iterations" : how many iterations actually ran.
        CONVERGENCE ISN'T GUARANTEED -- see module docstring. With very
        few games (e.g. early in a season, most teams with 1-2 games),
        the hard win/loss-inclusion thresholds can create a feedback loop
        with no fixed point, so the iteration oscillates indefinitely
        instead of settling down. When that happens this function still
        returns a DataFrame (whatever npi_scores holds when the loop
        exhausts max_iterations), but that value is essentially arbitrary
        -- it depends on which phase of the oscillation the loop happened
        to stop on. attrs["converged"] = False is the caller's signal
        that the returned npi values shouldn't be trusted as-is.
    """
    if df.empty:
        empty = pd.DataFrame(columns=['team', 'npi', 'games_played'])
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

            # RESULT-dependent location multiplier: home win / away loss =
            # discount ("expected"); away win / home loss = boost
            # ("notable"). Neutral-site and tied games always use 1.0.
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
                    # raw_value == win_value here (win case). Averaged value
                    # is the RAW value; loc_mult is applied ONLY as the
                    # weight, not baked into the value itself.
                    wins_ties.append((win_value, loc_mult, raw_value, None))
                elif result_type == 'tie':
                    wins_ties.append((win_value, 0.5, raw_value, loss_value))
                else:
                    # For a loss, raw_value here is actually loss_value.
                    losses.append((raw_value, loc_mult))

            # Sort by win_value (RAW, pre-multiplier) descending -- equivalent
            # to sorting by the opponent's own NPI, since win_value is a
            # strictly increasing function of opponent_npi alone.
            wins_ties.sort(key=lambda x: x[0], reverse=True)

            weighted_games = []
            win_count = 0.0
            for win_value, loc_mult, raw_value, loss_value in wins_ties:
                remaining_capacity = min_wins - win_count
                if remaining_capacity <= 0:
                    # Overflow check uses the RAW win_value -- see module
                    # docstring for why this is the mathematically
                    # meaningful comparison.
                    weight = 1.0 if win_value > old_npi[team] else 0.0
                elif loc_mult <= remaining_capacity:
                    weight = 1.0
                else:
                    if win_value > old_npi[team]:
                        weight = 1.0
                    else:
                        weight = remaining_capacity / loc_mult

                # VALUE averaged is raw_value (NOT loc_mult * raw_value).
                # loc_mult is applied exactly once, as the weight.
                weighted_games.append((raw_value, weight * loc_mult))
                if loss_value is not None:
                    weighted_games.append((loss_value, 0.5))
                win_count += loc_mult

            for loss_raw_value, loss_weight in losses:
                # Loss-inclusion check also uses the RAW loss_value, not a
                # loc-multiplied value -- confirmed against Ohio Wesleyan's
                # Mary Washington game (see module docstring).
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

        max_change = max(abs(npi_scores[team] - old_npi[team]) for team in teams)
        if max_change < tolerance:
            converged = True
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
    result_df.attrs['converged'] = converged
    result_df.attrs['iterations'] = iteration + 1
    return result_df
