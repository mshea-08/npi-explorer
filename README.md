# D3 Soccer NPI Explorer

## What this is

1. `npi.py` — your NPI calculator (`calculate_npi`), unchanged except `init`
   is now a real parameter instead of hardcoded.
2. `collector.py` — pulls final scores for D3 men's/women's soccer from
   ncaa.com (via the free [ncaa-api](https://github.com/henrygd/ncaa-api)
   wrapper) and writes them to a CSV.
3. `edits.py` — merge logic for user edits (score corrections, added/
   removed games) layered on top of collected data. Season-agnostic: also
   powers 2026's from-scratch manual entry, since a season with no base
   data is just the same merge with an empty starting point.
4. `app.py` — a Streamlit app: pick a season and sport, see NPI rankings,
   and edit/add games directly in the browser.

## Setup

```bash
pip install -r requirements.txt
```

## Step 1 — collect the data

```bash
python collector.py --sport men   --season 2025
python collector.py --sport women --season 2025
```

(`--season` defaults to 2025, so it can be omitted for this year; explicit
here for clarity, and required once 2026 rolls around.)

This writes two files per sport: `d3_mens_soccer_2025_raw.csv` (everything
collected, manual exclusions already stripped) and `d3_mens_soccer_2025.csv`
(the one `app.py` actually reads — the raw file filtered down to confirmed
D3-vs-D3 games only). Fetching is resumable — re-running only pulls dates
it hasn't already gotten.

**Cross-division filtering — how it actually works now:** the per-game
detail endpoint (`/game/{id}`) returns each team's actual division
directly —

```json
{"contests": [{"teams": [
    {"isHome": true,  "divisionName": "d1"},
    {"isHome": false, "divisionName": "d3"}
]}]}
```

Division belongs to the **team**, not the game — a school doesn't change
divisions between games — so instead of caching one lookup per game
(~3,600 requests), the collector caches one lookup per unique **team**
(typically a few hundred). Every `/game/{id}` call reveals both
participants' divisions at once, so a game only gets fetched if at least
one of its two teams isn't already known; once a team's division is
known, every other game it plays is free. That alone cuts total requests
by roughly 5–10x. On top of that, lookups run concurrently (`--division-workers`,
default 4) instead of one at a time, with a shared rate limiter
(`--division-rate-limit`, default 4/sec) that caps the combined request
rate across all workers so the API's 5 req/sec limit is respected
regardless of concurrency. Net effect: what used to take 15–20 minutes
now typically finishes in a couple of minutes.

Cached to `d3_team_divisions_mens.json` / `_womens.json`. **If you have an
old `d3_game_divisions_mens.json` from an earlier version of this
script** (the old, slower per-game cache) — including one from a run
that's still in progress — it's automatically migrated into the new
cache format on your next run, so any partial progress isn't lost; you
don't need to wait for an old run to finish or delete anything first.

**If you only need to re-filter** (edited `manual_exclusions.csv` or a
`provisional_teams_{season}.csv`, or want to retry teams whose division
lookup failed) without re-fetching the scoreboard itself:

```bash
python collector.py --sport men --filter-only
python collector.py --sport men --filter-only --retry-failed-divisions
python collector.py --sport men --filter-only --division-workers 8 --division-rate-limit 4.5
```

**Seasons.** Everything above is per-season: `--season 2025` (the
default) or `--season 2026` produces entirely separate files
(`d3_mens_soccer_2025*.csv` vs `d3_mens_soccer_2026*.csv`), and
`--start`/`--end` default to `<season>-08-29` / `<season>-11-09` if not
given explicitly. Run a specific season with:

```bash
python collector.py --sport men --season 2026
```

**Provisional teams — season-specific, not automatic.** NCAA D3
provisional members shouldn't count toward NPI, but nothing in the API
flags this — a provisional team looks like a completely normal D3
program (correct division, full slate of games). This has to be a
manually maintained list: `provisional_teams_{season}.csv`, one team name
per line using ncaa.com's naming (e.g. `Penn St. Brandywine`). It's keyed
by season on purpose, since provisional status is a multi-year
transition — a team on the 2025 list may be a full member by 2026, and a
different school may be newly provisional that year. `provisional_teams_2025.csv`
ships with the 4 teams confirmed so far (Regent, Carlow, Penn St.
Brandywine, JWU Charlotte); `provisional_teams_2026.csv` ships empty as a
template — populate it once 2026's provisional list is known. The
division cache (`d3_team_divisions_mens.json`), by contrast, **is**
shared across seasons on purpose — a team's division essentially never
changes year to year, so there's no reason to re-resolve several hundred
teams from scratch every season.

Penalty-kick shootouts are handled: NCAA soccer records a shootout-decided
game as a **tie** (the shootout only decides who advances in a bracket,
not a win/loss), so the collector detects shootout status markers and
forces `home_score == away_score` for those games regardless of what the
raw score field shows.

**On exhibitions/scrimmages:** we looked for a way to detect these
directly — checked every field in both the scoreboard summary and the
per-game detail endpoint — and found no exhibition/scrimmage flag
anywhere in either. One candidate exhibition game we checked closely
(a cross-division matchup) actually had incrementing official win-loss
records for both teams, suggesting NCAA counted it rather than excluding
it. Given that, this is no longer handled by automatic detection — if you
know specific games are exhibitions, add their `game_id` to
`manual_exclusions.csv` (one per line, or a `game_id` column) and re-run
with `--filter-only`.

**First-run checks:** this was built against the documented API shape but
couldn't be fully tested against live data from this environment. The
scoreboard and per-game division parsing were both verified against real
example payloads, so they should be solid — but worth a spot check:

```bash
# Scoreboard shape -- prints raw JSON for one day
python collector.py --sport men --start 2025-09-13 --end 2025-09-13 --debug
```

**If you already ran an earlier version of this script**, the first run
of this version will automatically migrate your existing
`d3_mens_soccer_2025.csv` into the new raw file (no re-fetching needed)
and build a proper division cache from it — you don't need to delete
anything or start over. **You will need to delete the old
`d3_teams_mens.json` / `d3_teams_mens_counts.json` files** if they exist —
those were the retired whitelist approach and are no longer used.

The default date range (`2025-08-29` to `2025-11-09`) starts right around
when the regular season begins (after the mid-to-late-August preseason
exhibition window) and runs through conference tournaments, stopping just
before the NCAA tournament selection show. Starting at 8/29 instead of
earlier in August is a deliberate side-benefit: it skips over the window
where most exhibitions/scrimmages cluster, which matters since — as
covered further down — there's no reliable way to detect exhibition
status from the API itself. Adjust `--start`/`--end` if you want to
include the NCAA tournament itself, narrow the regular-season window
further, or you know exhibitions extend past 8/29 for a given season.

## Step 2 — run the app

```bash
streamlit run app.py
```

Opens in your browser. Pick a **Season** (2025 or 2026) and **Sport** in
the sidebar. Each combination has two tabs:

**Rankings** — pick a date (2025 only), see NPI computed from every game
up to that date. 2026 has no date picker — since there's no full season
calendar yet, it always shows current standings from whatever's been
entered. NPI parameters (win_dial, sos_dial, qwb_mult, qwb_threshold,
min_wins, initial NPI) are adjustable from a sidebar expander, same as we
were testing earlier — each sport keeps its own slider positions
independently.

**Edit Games** (2025) / **Add Games** (2026) — an editable table.
- **2025:** correct a score, delete a game (remove its row), or add a
  game the collector missed.
- **2026:** there's no collected data (the season hasn't happened), so
  this tab is just a way to enter games by hand as they're played — add a
  row with the date, teams, and final score. The Rankings tab picks up
  new entries immediately after saving.
- Either way: click **Save changes** to apply, or use the **Reset to
  official NCAA data** button (shown whenever any edit is active, on both
  tabs) to instantly discard everything and go back to the real, unedited
  numbers.

**Edits are private to your browser session — never shared, never
written to disk.** They live in Streamlit's `st.session_state`, which is
isolated per person connected to the app; nothing you add or change is
ever visible to anyone else using it, and `collector.py`'s output files
are only ever read, never modified. Closing the tab or starting a fresh
session clears everything back to official data automatically — the
Reset button just does that on demand without needing to reload the page.
One consequence worth knowing: since nothing persists to disk, your edits
also won't survive *your own* page reload — if you want to keep exploring
a specific hypothetical across a longer session, keep the tab open rather
than refreshing.

Each sport has its own NPI-parameter defaults, set in `DEFAULT_PARAMS` at
the top of `app.py`:

- **Men:** win_dial=0.15, qwb_mult=0.75, qwb_threshold=54, min_wins=10
- **Women:** win_dial=0.20, qwb_mult=0.50, qwb_threshold=54, min_wins=8
  (unchanged from the original)

## Notes

- Games are only included once they're finished — in-progress and
  not-yet-played games are skipped.
- Only D3-vs-D3 games are kept, confirmed via each game's authoritative
  division data (not a guess or a name whitelist).
- Shootout-decided games are recorded as ties, not wins/losses.
- Exhibition/scrimmage status isn't detectable from this API (checked
  both endpoints, no such field exists) — use `manual_exclusions.csv` for
  any specific games you know should be excluded, or just delete/edit
  them directly in the app's Edit Games tab.
- Team names come straight from ncaa.com's "short" name field. If the same
  school shows up under two slightly different names across games (rare,
  but possible with things like abbreviation inconsistencies), that would
  silently split into two "teams" in the ratings — worth spot-checking the
  team list after the first full collection run.
- The Streamlit app reads local files directly; there's no live scraping
  happening when you interact with the date picker or editor, so it's
  instant. Re-run `collector.py` to pull in new 2025 results as the season
  goes on -- this is always safe to do, since edits live in each user's
  session, never in the CSVs themselves.
- `edits.py` holds the merge logic (base data + session edits -> effective
  games). It's plain Python with no Streamlit dependency and no disk
  writes of its own -- it only ever reads the shared base CSV -- so it's
  directly unit-testable outside the app, and it's easy to audit that
  there's no path by which one user's edits could end up persisted
  somewhere another user might see them.
  `collector.py` to pull in new results as the season goes on.
