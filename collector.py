"""
Collects final game/match results for NCAA D3 sports from the ncaa.com
scoreboard, via the free ncaa-api wrapper (https://github.com/henrygd/ncaa-api).
Currently supported: men's soccer, women's soccer, women's volleyball
(see SPORT_CONFIG below to add more).

USAGE
-----
    python collector.py --sport mens_soccer       --season 2025
    python collector.py --sport womens_soccer     --season 2026
    python collector.py --sport womens_volleyball --season 2025

Note for volleyball: "home_score"/"away_score" hold SETS WON (e.g. 3-1),
not points -- ncaa.com's scoreboard API returns whatever the sport's
official final tally is in the same "score" field regardless of sport, so
no parsing changes were needed, but sanity-check the first collected day
with --debug if you want to confirm the shape looks right. Volleyball has
no ties and no shootouts, so those code paths simply never trigger for it.

--season controls which season's files get read/written and which
provisional-teams list applies (see below). Defaults to 2025 if omitted.
--start/--end default to <season>-08-29 and <season>-11-09 if not given
explicitly.

Writes two files per sport per season (see SPORT_CONFIG for each sport's
csv_label):
  d3_{csv_label}_2025_raw.csv   -- everything collected (manually-excluded
                                    games already removed, but NOT filtered
                                    for cross-division)
  d3_{csv_label}_2025.csv       -- the raw file filtered down to confirmed
                                    D3-vs-D3 games only. This is what
                                    app.py reads.
e.g. d3_mens_soccer_2025.csv, d3_womens_volleyball_2025.csv. (2026 files
are named d3_{csv_label}_2026*.csv, etc.)

PROVISIONAL TEAMS -- SEASON-SPECIFIC
---------------------------------------
NCAA D3 provisional members shouldn't count toward NPI, but they look
completely normal in every automated check this script does (correct
division, a full slate of games) -- there's no field anywhere in the API
that flags provisional status. So this is a manually maintained list:
provisional_teams_{season}.csv, one team name per line (ncaa.com's
naming, e.g. "Penn St. Brandywine"). This is deliberately keyed by season
rather than being one shared list, because provisional status is a
multi-year transition -- a team on the 2025 list may be a full member by
2026, and a different school may be newly provisional. Games involving
any listed team are dropped in filter_to_final, alongside cross-division
games.

CROSS-DIVISION FILTERING -- HOW IT WORKS
-------------------------------------------
The per-game detail endpoint (/game/{id}) returns each team's real
division:

    {"contests": [{"teams": [
        {"isHome": true,  "divisionName": "d1", ...},
        {"isHome": false, "divisionName": "d3", ...}
    ]}]}

Division is a property of the TEAM (an institution's whole athletic
program), not of an individual game -- so instead of caching one lookup
per game_id (~3,600 requests), this caches one lookup per unique TEAM
NAME (typically a few hundred). Every /game/{id} call reveals BOTH
teams' divisions at once, so once a team has appeared in one resolved
lookup, every other game involving that team is free -- no network call
needed. In practice this cuts total requests by roughly 5-10x compared to
a naive one-request-per-game approach. Unlike the provisional list, this
cache is shared across seasons (see _team_division_cache_path) since
division essentially never changes year to year.

On top of that, lookups run concurrently (a small thread pool) instead of
one at a time, with a shared rate limiter that caps the AGGREGATE request
rate across all threads (not per-thread) so the API's stated 5 req/sec
limit is respected regardless of how many workers are running.

Together these two changes are what actually matter for speed -- fewer
requests, and requests overlapped instead of serialized -- rather than
just tuning a fixed per-request delay.

If you already have a d3_game_divisions_{men,women}.json from an earlier
version of this script (the old per-game cache), it's automatically
migrated into the new per-team cache on first run so that progress isn't
wasted, then ignored from then on.

PENALTY-KICK SHOOTOUTS
------------------------
NCAA soccer records a shootout-decided game as a TIE -- the shootout only
determines who advances in a bracket, it is not a win/loss result. This
script detects shootout status markers (SO/PK/Shootout in the final
status string) and forces home_score == away_score for those games,
regardless of what the raw score field shows.

RESUMABILITY
-------------
Fetching skips dates already pulled (tracked in a .done file next to the
raw CSV). Division lookups skip teams already resolved. Both mean
re-running as the season progresses only does new work.
"""

import argparse
import csv
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta

import requests

BASE_URL = "https://ncaa-api.henrygd.me"
DIVISION = "d3"

# Sport configuration. Each key is the value passed to --sport and used
# throughout app.py/edits.py as the "sport" identifier. Each entry has:
#   "path"        -- URL segment on ncaa.com's scoreboard, i.e.
#                     ncaa.com/scoreboard/{path}/d3/YYYY/MM/DD/all-conf
#   "csv_label"   -- file label for the raw/final game CSVs:
#                     d3_{csv_label}_{season}(_raw).csv
#   "cache_label" -- file label for the team-division cache and the
#                     known-teams list: d3_team_divisions_{cache_label}.json,
#                     d3_known_teams_{cache_label}_{season}.csv
#
# cache_label is kept separate from csv_label, and deliberately equal to
# the ORIGINAL "mens"/"womens" labels for the two soccer sports, so adding
# new sports here never invalidates an already-resolved division cache or
# renames a known-teams file out from under anyone who already has one.
# New sports get their own cache_label (not shared with soccer's), since
# division/team-name lookups are per-sport network calls and there's no
# guarantee ncaa.com's team-name strings match exactly across sports.
SPORT_CONFIG = {
    "mens_soccer": {
        "path": "soccer-men", "csv_label": "mens_soccer", "cache_label": "mens",
    },
    "womens_soccer": {
        "path": "soccer-women", "csv_label": "womens_soccer", "cache_label": "womens",
    },
    "womens_volleyball": {
        "path": "volleyball-women", "csv_label": "womens_volleyball",
        "cache_label": "womens_volleyball",
    },
    "womens_field_hockey": {
        "path": "fieldhockey", "csv_label": "womens_field_hockey",
        "cache_label": "womens_field_hockey",
    },
}
SPORT_PATHS = {k: v["path"] for k, v in SPORT_CONFIG.items()}  # kept for readability at call sites
REQUEST_DELAY_SECONDS = 0.3  # scoreboard fetch stays serial/day-by-day; this paces those calls
MAX_RETRIES = 3
DEFAULT_DIVISION_WORKERS = 4
DEFAULT_DIVISION_RATE_LIMIT = 4.0  # requests/sec, aggregate across all workers; API allows 5
DEFAULT_SEASON = "2025"

SHOOTOUT_MARKERS = ("SO", "PK", "PKS", "SHOOTOUT", "PENALTIES")

FIELDNAMES = ["date", "home_team", "away_team", "home_score", "away_score", "status", "game_id", "neutral_site"]


class RateLimiter:
    """Caps aggregate calls/sec across however many threads share this
    instance -- NOT a per-thread delay, which would let concurrency blow
    past the intended rate. Thread-safe."""

    def __init__(self, max_per_second: float):
        self.min_interval = 1.0 / max_per_second
        self._lock = threading.Lock()
        self._last_call = 0.0

    def wait(self):
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self._last_call = time.monotonic()


# ---------------------------------------------------------------------------
# Per-game division lookup (authoritative cross-division filtering)
# ---------------------------------------------------------------------------

def fetch_game_division(game_id: str, debug: bool = False):
    """
    Hits /game/{id} and returns {'home_team': str, 'away_team': str,
    'home_division': 'd3', 'away_division': 'd1'} (lowercased), or None if
    the lookup failed or the shape didn't match. Returns team names too
    (not just divisions) since the caller needs to know which teams this
    lookup just resolved.
    """
    url = f"{BASE_URL}/game/{game_id}"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            contests = data.get("contests", [])
            if not contests:
                return None
            teams = contests[0].get("teams", [])
            home = next((t for t in teams if t.get("isHome")), None)
            away = next((t for t in teams if not t.get("isHome")), None)
            if not home or not away:
                return None
            result = {
                "home_team": home.get("nameShort") or "",
                "away_team": away.get("nameShort") or "",
                "home_division": (home.get("divisionName") or "").strip().lower(),
                "away_division": (away.get("divisionName") or "").strip().lower(),
            }
            if debug:
                print(f"  [debug-div] game {game_id}: {result}")
            return result
        except (requests.RequestException, json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
            if attempt == MAX_RETRIES:
                if debug:
                    print(f"  [debug-div] game {game_id} lookup failed: {e}")
                return None
            time.sleep(1.0 * attempt)
    return None


def inspect_game(game_id: str):
    """Read-only: prints the full raw JSON from /game/{id}. No writes."""
    url = f"{BASE_URL}/game/{game_id}"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, json.JSONDecodeError) as e:
        print(f"=== game {game_id}: fetch failed: {e} ===")
        return
    print(f"=== game {game_id} (full raw JSON from /game/{game_id}) ===")
    print(json.dumps(data, indent=2))


def _team_division_cache_path(sport: str, out_dir: str) -> str:
    """
    Deliberately NOT season-specific, unlike provisional_teams_{season}.csv.
    A team's division (D1/D2/D3) is a much slower-changing property than
    provisional membership -- schools essentially never reclassify
    divisions year to year -- so this cache is shared across seasons to
    avoid re-resolving several hundred teams' divisions every year for no
    real benefit. If a team's division genuinely does change, delete its
    entry from this file (or the whole file, to force a full re-resolve)
    and re-run -- a missing entry gets looked up again automatically, no
    special flag needed.
    """
    label = SPORT_CONFIG[sport]["cache_label"]
    return os.path.join(out_dir, f"d3_team_divisions_{label}.json")


def _legacy_game_cache_path(sport: str, out_dir: str) -> str:
    label = SPORT_CONFIG[sport]["cache_label"]
    return os.path.join(out_dir, f"d3_game_divisions_{label}.json")


def _migrate_legacy_game_cache(sport: str, out_dir: str, team_cache: dict):
    """One-time: pull whatever team divisions can be salvaged out of an
    old per-game cache into the new per-team cache, so a prior run's
    progress isn't wasted. The old cache didn't store team names inside
    each entry, so this can only recover games whose game_id we can still
    match against the raw CSV to get team names."""
    legacy_path = _legacy_game_cache_path(sport, out_dir)
    if not os.path.exists(legacy_path):
        return
    raw_path = _raw_csv_path(sport, out_dir)
    if not os.path.exists(raw_path):
        return

    with open(legacy_path) as f:
        legacy = json.load(f)

    game_to_teams = {}
    with open(raw_path, newline="") as f:
        for row in csv.DictReader(f):
            game_to_teams[row["game_id"]] = (row["home_team"], row["away_team"])

    recovered = 0
    for gid, entry in legacy.items():
        teams = game_to_teams.get(gid)
        if not teams:
            continue
        home_team, away_team = teams
        hd = entry.get("home_division")
        ad = entry.get("away_division")
        if hd and home_team not in team_cache:
            team_cache[home_team] = hd
            recovered += 1
        if ad and away_team not in team_cache:
            team_cache[away_team] = ad
            recovered += 1

    if recovered:
        print(f"Migrated {recovered} team division(s) from the old per-game cache ({legacy_path}).")
    try:
        os.rename(legacy_path, legacy_path + ".migrated")
    except OSError:
        pass


def resolve_team_divisions(sport: str, raw_path: str, out_dir: str = ".", debug: bool = False,
                            retry_failed: bool = False, workers: int = DEFAULT_DIVISION_WORKERS,
                            rate_limit: float = DEFAULT_DIVISION_RATE_LIMIT) -> dict:
    """
    Returns {team_name: 'd3'/'d1'/... } covering every team in the raw
    CSV, resolved with as few network calls as possible: each /game/{id}
    call reveals both participants' divisions, so a game is only fetched
    if at least one of its two teams isn't already known. Concurrent
    (up to `workers` in flight) but rate-limited in aggregate.
    """
    cache_path = _team_division_cache_path(sport, out_dir)
    team_cache = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            team_cache = json.load(f)

    _migrate_legacy_game_cache(sport, out_dir, team_cache)

    if retry_failed:
        team_cache = {t: d for t, d in team_cache.items() if d is not None}

    games = []
    seen_gids = set()
    all_teams = set()
    with open(raw_path, newline="") as f:
        for row in csv.DictReader(f):
            all_teams.add(row["home_team"])
            all_teams.add(row["away_team"])
            gid = row["game_id"]
            if gid and gid not in seen_gids:
                seen_gids.add(gid)
                games.append((gid, row["home_team"], row["away_team"]))

    already_known = sum(1 for t in all_teams if t in team_cache)
    print(
        f"Team division cache: {len(all_teams)} unique teams in raw data, "
        f"{already_known} already known, resolving the rest from {len(games)} games "
        f"({workers} workers, ~{rate_limit}/sec)..."
    )

    lock = threading.Lock()
    limiter = RateLimiter(rate_limit)
    resolved_count = [0]
    fetched_count = [0]

    def worker(game):
        gid, home_team, away_team = game
        with lock:
            if home_team in team_cache and away_team in team_cache:
                return  # nothing new to learn from this game -- skip the network call entirely
        limiter.wait()
        result = fetch_game_division(gid, debug=debug)
        with lock:
            fetched_count[0] += 1
            if result is None:
                if home_team not in team_cache:
                    team_cache[home_team] = None
                if away_team not in team_cache:
                    team_cache[away_team] = None
                return
            # Prefer the API's own team names as keys when they match what
            # we expect; fall back to our CSV's names either way so both
            # sides of the join stay consistent.
            if home_team not in team_cache or team_cache[home_team] is None:
                team_cache[home_team] = result["home_division"]
                resolved_count[0] += 1
            if away_team not in team_cache or team_cache[away_team] is None:
                team_cache[away_team] = result["away_division"]
                resolved_count[0] += 1

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(worker, g) for g in games]
        done = 0
        for _ in as_completed(futures):
            done += 1
            if done % 100 == 0 or done == len(futures):
                with lock:
                    print(f"  {done}/{len(futures)} games checked, {fetched_count[0]} actually fetched, "
                          f"{resolved_count[0]} teams newly resolved...")
                    with open(cache_path, "w") as f:
                        json.dump(team_cache, f, indent=2)

    with open(cache_path, "w") as f:
        json.dump(team_cache, f, indent=2)

    unresolved = [t for t in all_teams if team_cache.get(t) is None]
    print(f"Done. {fetched_count[0]} network requests made (vs {len(games)} games / "
          f"{'a game-per-lookup approach would have needed up to that many'}).")
    if unresolved:
        print(f"  {len(unresolved)} team(s) still unresolved after failed lookups "
              f"(re-run with --retry-failed-divisions to retry): {unresolved[:10]}"
              f"{' ...' if len(unresolved) > 10 else ''}")

    return team_cache


def _load_manual_exclusions(out_dir: str) -> set:
    path = os.path.join(out_dir, "manual_exclusions.csv")
    if not os.path.exists(path):
        return set()
    ids = set()
    with open(path) as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            val = row[0].strip()
            if val and val.lower() != "game_id":
                ids.add(val)
    return ids


def _load_provisional_teams(out_dir: str, season: str = DEFAULT_SEASON) -> set:
    """
    provisional_teams_{season}.csv (one team name per line, or a 'team'
    column, using ncaa.com's naming -- e.g. 'Penn St. Brandywine') lists
    teams that are NCAA D3 provisional members for that specific season.
    This is season-specific on purpose: provisional status is a multi-year
    transition, so a team on the 2025 list may graduate to full membership
    by 2026, and a different set of teams may be newly provisional --
    provisional_teams_2025.csv and provisional_teams_2026.csv are
    independent lists, not one shared file.

    Provisional teams show up as normal, legitimate D3 opponents in every
    other check this script does (correct division, plays a full slate of
    games), so there's no way to detect them automatically -- this list is
    manually maintained. Games involving any of these teams are dropped in
    filter_to_final, the same way cross-division games are.
    """
    path = os.path.join(out_dir, f"provisional_teams_{season}.csv")
    if not os.path.exists(path):
        return set()
    teams = set()
    with open(path) as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            val = row[0].strip()
            if val and val.lower() != "team":
                teams.add(val)
    return teams


def _known_teams_path(sport: str, out_dir: str, season: str = DEFAULT_SEASON) -> str:
    label = SPORT_CONFIG[sport]["cache_label"]
    return os.path.join(out_dir, f"d3_known_teams_{label}_{season}.csv")


def _load_known_teams(sport: str, out_dir: str, season: str = DEFAULT_SEASON):
    """
    d3_known_teams_{sport}_{season}.csv (one team name per line, or a
    'team' column, using ncaa.com's naming) is an alternative to
    resolve_team_divisions' network-based approach: if this file exists,
    it's treated as the complete, authoritative list of legitimate D3
    teams for cross-division filtering, and NO network calls are made at
    all for division checking -- a team either matches a name on this
    list (kept) or it doesn't (dropped), pure local computation. This is
    dramatically faster than the network-based approach when you already
    have a trustworthy source for the team list (e.g. an official NPI
    rankings export), at the cost of needing the names to match
    ncaa.com's own naming exactly.

    Returns None (not an empty set) if the file doesn't exist, so callers
    can distinguish "use this list" from "no list supplied, fall back to
    network-based division resolution" -- an empty file would otherwise
    be indistinguishable from "not supplied" if this returned set().
    """
    path = _known_teams_path(sport, out_dir, season)
    if not os.path.exists(path):
        return None
    teams = set()
    with open(path) as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            val = row[0].strip()
            if val and val.lower() != "team":
                teams.add(val)
    return teams


# ---------------------------------------------------------------------------
# Scoreboard parsing
# ---------------------------------------------------------------------------

def _scoreboard_url(sport_path: str, d: date) -> str:
    return f"{BASE_URL}/scoreboard/{sport_path}/{DIVISION}/{d.year}/{d.month:02d}/{d.day:02d}/all-conf"


def _is_shootout(status: str) -> bool:
    s = status.upper()
    return any(marker in s for marker in SHOOTOUT_MARKERS)


def _parse_game(game_wrapper: dict, game_date: date):
    """
    Returns (game_dict_or_None, reason). reason is one of:
    'ok', 'not_finished', 'parse_error'.

    NEUTRAL SITE -- always False here, by design
    ------------------------------------------------
    Checked three separate ncaa-api endpoints (scoreboard, /game/{id},
    /game/{id}/boxscore) against a known neutral-site field hockey game
    and confirmed none of them expose venue or neutral-site information
    anywhere -- ncaa.com's public feed just doesn't track this. With no
    reliable source and no practical way to hand-curate it across ~160+
    D3 field hockey teams, this is left as a permanent False here. If a
    specific game is known to have been neutral-site, correct it by hand
    via the "Neutral site" checkbox in the app's game editor (Edit Games
    tab) -- that path is fully wired up in npi.py already.
    """
    game = game_wrapper.get("game", game_wrapper)

    status = (game.get("finalMessage") or game.get("gameState") or "").strip()
    if not status.upper().startswith("FINAL"):
        return None, "not_finished"

    try:
        away = game["away"]
        home = game["home"]
        # .strip() matters here: ncaa.com's own naming has been observed
        # to inconsistently include stray whitespace for the same team
        # across different games (e.g. "SUNY Cobleskill " with a trailing
        # space). Since npi.py uses team name strings as dictionary keys,
        # an unstripped inconsistency would silently split one team into
        # two entries in the standings -- normalize at the source so that
        # can't happen.
        away_name = away["names"]["short"].strip()
        home_name = home["names"]["short"].strip()
        away_score = int(away["score"])
        home_score = int(home["score"])
    except (KeyError, TypeError, ValueError):
        return None, "parse_error"

    if _is_shootout(status):
        tied_score = min(home_score, away_score)
        home_score = away_score = tied_score

    parsed = {
        "date": game_date.isoformat(),
        "home_team": home_name,
        "away_team": away_name,
        "home_score": home_score,
        "away_score": away_score,
        "status": status,
        "game_id": game.get("gameID", ""),
        "neutral_site": False,  # TODO -- see docstring above
    }
    return parsed, "ok"


def fetch_day(sport_path: str, d: date, debug: bool = False):
    """Returns kept_games for the day."""
    url = _scoreboard_url(sport_path, d)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 404:
                return []
            resp.raise_for_status()
            data = resp.json()
            if debug:
                print(f"--- RAW RESPONSE for {d.isoformat()} ---")
                print(json.dumps(data, indent=2)[:3000])
                print("--- END RAW ---")
            games_raw = data.get("games", [])
            results = [_parse_game(g, d) for g in games_raw]
            return [g for g, reason in results if reason == "ok"]
        except (requests.RequestException, json.JSONDecodeError) as e:
            if attempt == MAX_RETRIES:
                print(f"  [WARN] failed on {d.isoformat()} after {MAX_RETRIES} attempts: {e}", file=sys.stderr)
                return []
            time.sleep(1.0 * attempt)
    return []


def inspect_day(sport_path: str, d: date):
    """
    Read-only debugging helper: fetches the scoreboard for this exact day
    and prints the FULL, untruncated raw JSON for every game on it.
    Always hits the network (ignores the .done cache) and never writes to
    any file.
    """
    url = _scoreboard_url(sport_path, d)
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 404:
            print(f"=== {d.isoformat()}: no games (404) ===")
            return
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, json.JSONDecodeError) as e:
        print(f"=== {d.isoformat()}: fetch failed: {e} ===")
        return

    games_raw = data.get("games", [])
    print(f"=== {d.isoformat()}: {len(games_raw)} game(s) ===")
    for i, g in enumerate(games_raw):
        print(f"--- game {i} (full raw JSON, untruncated) ---")
        print(json.dumps(g, indent=2))
    print()


def _daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def _done_file(csv_path: str) -> str:
    return csv_path + ".done"


def _load_done_dates(csv_path: str) -> set:
    done_path = _done_file(csv_path)
    if not os.path.exists(done_path):
        return set()
    with open(done_path) as f:
        return set(line.strip() for line in f if line.strip())


def _append_done_date(csv_path: str, d: date):
    with open(_done_file(csv_path), "a") as f:
        f.write(d.isoformat() + "\n")


def _raw_csv_path(sport: str, out_dir: str, season: str = DEFAULT_SEASON) -> str:
    label = SPORT_CONFIG[sport]["csv_label"]
    return os.path.join(out_dir, f"d3_{label}_{season}_raw.csv")


def _final_csv_path(sport: str, out_dir: str, season: str = DEFAULT_SEASON) -> str:
    label = SPORT_CONFIG[sport]["csv_label"]
    return os.path.join(out_dir, f"d3_{label}_{season}.csv")


def _migrate_legacy_final_to_raw(sport: str, out_dir: str, season: str = DEFAULT_SEASON):
    """One-time migration: if a final CSV exists from an earlier version
    and no raw CSV exists yet, treat the final CSV as raw data so
    already-fetched games aren't lost or re-downloaded. Only relevant for
    2025, since that's the only season that ever used the pre-raw/final-split
    file layout."""
    final_path = _final_csv_path(sport, out_dir, season)
    raw_path = _raw_csv_path(sport, out_dir, season)
    if os.path.exists(raw_path) or not os.path.exists(final_path):
        return
    print(f"Migrating existing {final_path} -> {raw_path} (one-time, no re-fetching needed)...")
    with open(final_path, newline="") as src, open(raw_path, "w", newline="") as dst:
        dst.write(src.read())
    legacy_done = _done_file(final_path)
    if os.path.exists(legacy_done):
        with open(legacy_done) as src, open(_done_file(raw_path), "w") as dst:
            dst.write(src.read())


def fetch_raw_season(sport: str, start: date, end: date, out_dir: str = ".", debug: bool = False,
                      season: str = DEFAULT_SEASON) -> str:
    """Fetches (resumably) into the raw CSV, applying only manual-exclusion
    filtering. Cross-division filtering happens later, in filter_to_final."""
    sport_path = SPORT_PATHS[sport]
    raw_path = _raw_csv_path(sport, out_dir, season)
    manual_exclusions = _load_manual_exclusions(out_dir)
    if manual_exclusions:
        print(f"Loaded {len(manual_exclusions)} manually-excluded game_id(s) from manual_exclusions.csv")

    already_done = _load_done_dates(raw_path)
    file_exists = os.path.exists(raw_path)

    total_games = 0
    total_dropped_manual = 0
    with open(raw_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()

        for d in _daterange(start, end):
            if d.isoformat() in already_done:
                continue
            games = fetch_day(sport_path, d, debug=debug)

            kept = []
            dropped_manual = 0
            for g in games:
                if g["game_id"] in manual_exclusions:
                    dropped_manual += 1
                    continue
                kept.append(g)
            total_dropped_manual += dropped_manual

            for g in kept:
                writer.writerow(g)
            f.flush()
            total_games += len(kept)
            _append_done_date(raw_path, d)

            dropped_note = f", dropped {dropped_manual} manually excluded" if dropped_manual else ""
            print(f"  {d.isoformat()}: {len(kept)} games{dropped_note}")
            time.sleep(REQUEST_DELAY_SECONDS)

    print(f"\nRaw fetch done. {total_games} new games added to {raw_path}.")
    if total_dropped_manual:
        print(f"  Manually-excluded games dropped: {total_dropped_manual}")
    return raw_path


def filter_to_final(sport: str, out_dir: str = ".", debug: bool = False,
                     retry_failed_divisions: bool = False,
                     workers: int = DEFAULT_DIVISION_WORKERS,
                     rate_limit: float = DEFAULT_DIVISION_RATE_LIMIT,
                     season: str = DEFAULT_SEASON) -> str:
    """
    Rebuilds the final (cross-division-filtered) CSV from the raw CSV.

    Two ways to determine which teams are legitimate D3 opponents:
      1. d3_known_teams_{sport}_{season}.csv exists -> use it directly as
         the authoritative team list. Pure local set-membership checks,
         ZERO network calls. Fast, but only as correct as the supplied
         list's names matching ncaa.com's naming exactly -- see the
         unmatched-team diagnostic below.
      2. No such file -> fall back to resolve_team_divisions, which looks
         up each team's real division via /game/{id} (cached, concurrent,
         rate-limited -- still much faster than a naive approach, but
         real network time, roughly a couple of minutes for a full season).
    """
    raw_path = _raw_csv_path(sport, out_dir, season)
    final_path = _final_csv_path(sport, out_dir, season)

    if not os.path.exists(raw_path):
        print(f"  [WARN] No raw data at {raw_path} yet -- run a fetch first.", file=sys.stderr)
        return final_path

    known_teams = _load_known_teams(sport, out_dir, season)
    using_known_list = known_teams is not None

    if using_known_list:
        print(
            f"Using {_known_teams_path(sport, out_dir, season)} as the authoritative D3 team list "
            f"({len(known_teams)} teams) -- skipping network division lookups entirely."
        )
        team_divisions = None
    else:
        team_divisions = resolve_team_divisions(
            sport, raw_path, out_dir=out_dir, debug=debug, retry_failed=retry_failed_divisions,
            workers=workers, rate_limit=rate_limit,
        )

    provisional_teams = _load_provisional_teams(out_dir, season)
    provisional_path = os.path.join(out_dir, f"provisional_teams_{season}.csv")
    if provisional_teams:
        print(f"Loaded {len(provisional_teams)} provisional team(s) to exclude from {provisional_path}")
    else:
        print(f"No {provisional_path} found -- provisional-team filtering skipped for {season}.")

    total = 0
    kept = 0
    dropped_cross = 0
    dropped_unknown = 0
    dropped_provisional = 0
    unmatched_counts: dict = {}

    def is_d3(team: str) -> bool:
        team = team.strip()
        if using_known_list:
            return team in known_teams
        return team_divisions.get(team) == DIVISION

    with open(raw_path, newline="") as src, open(final_path, "w", newline="") as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in reader:
            total += 1
            home_team = row["home_team"].strip()
            away_team = row["away_team"].strip()

            if home_team in provisional_teams or away_team in provisional_teams:
                dropped_provisional += 1
                continue

            home_ok = is_d3(home_team)
            away_ok = is_d3(away_team)

            if using_known_list:
                if not home_ok:
                    unmatched_counts[home_team] = unmatched_counts.get(home_team, 0) + 1
                if not away_ok:
                    unmatched_counts[away_team] = unmatched_counts.get(away_team, 0) + 1
                if not home_ok or not away_ok:
                    dropped_unknown += 1
                    continue
                writer.writerow(row)
                kept += 1
            else:
                home_div = team_divisions.get(home_team)
                away_div = team_divisions.get(away_team)
                if home_div is None or away_div is None:
                    dropped_unknown += 1
                    continue
                if home_div == DIVISION and away_div == DIVISION:
                    writer.writerow(row)
                    kept += 1
                else:
                    dropped_cross += 1

    print(f"Filtered {total} raw games -> {kept} confirmed D3-vs-D3 games in {final_path}.")
    if dropped_provisional:
        print(f"  Provisional-team games dropped: {dropped_provisional}")

    if using_known_list:
        if dropped_unknown:
            print(f"  Games dropped (team not on the known-teams list): {dropped_unknown}")
            top = sorted(unmatched_counts.items(), key=lambda kv: -kv[1])[:15]
            print(
                "  Teams NOT found on the known-teams list, most-frequent first -- check these for "
                "naming mismatches (e.g. missing a period, different abbreviation) vs. genuine "
                "non-D3/cross-division opponents:"
            )
            for team, n in top:
                print(f"    {team!r}: appeared in {n} dropped game(s)")
            if len(unmatched_counts) > 15:
                print(f"    ... and {len(unmatched_counts) - 15} more distinct unmatched name(s)")
    else:
        print(f"  Cross-division games dropped: {dropped_cross}")
        if dropped_unknown:
            print(
                f"  Games dropped due to failed/unknown division lookups: {dropped_unknown} "
                f"(re-run with --retry-failed-divisions to retry just these)"
            )
    return final_path


def collect_season(sport: str, start: date, end: date, out_dir: str = ".", debug: bool = False,
                    retry_failed_divisions: bool = False,
                    workers: int = DEFAULT_DIVISION_WORKERS,
                    rate_limit: float = DEFAULT_DIVISION_RATE_LIMIT,
                    season: str = DEFAULT_SEASON):
    _migrate_legacy_final_to_raw(sport, out_dir, season)
    fetch_raw_season(sport, start, end, out_dir=out_dir, debug=debug, season=season)
    return filter_to_final(
        sport, out_dir=out_dir, debug=debug, retry_failed_divisions=retry_failed_divisions,
        workers=workers, rate_limit=rate_limit, season=season,
    )


def main():
    parser = argparse.ArgumentParser(description="Collect D3 soccer scores from ncaa.com")
    parser.add_argument("--sport", choices=list(SPORT_CONFIG.keys()), required=True)
    parser.add_argument("--season", default=DEFAULT_SEASON,
                         help=f"season year, e.g. 2025 or 2026 (default {DEFAULT_SEASON}). Determines "
                              f"which files are read/written (d3_{{csv_label}}_{{season}}*.csv) and "
                              f"which provisional-teams list is used (provisional_teams_{{season}}.csv), "
                              f"since provisional status changes year to year.")
    parser.add_argument("--start", default=None,
                         help="YYYY-MM-DD (default: <season>-08-29)")
    parser.add_argument("--end", default=None,
                         help="YYYY-MM-DD (default: <season>-11-09)")
    parser.add_argument("--out-dir", default=".")
    parser.add_argument("--debug", action="store_true",
                         help="print raw scoreboard/division JSON as it's fetched")
    parser.add_argument("--filter-only", action="store_true",
                         help="skip the scoreboard fetch; just (re)build the division cache and "
                              "final CSV from existing raw data (useful after editing "
                              "manual_exclusions.csv or provisional_teams_{season}.csv)")
    parser.add_argument("--retry-failed-divisions", action="store_true",
                         help="retry team division lookups that failed on a previous run, instead of "
                              "leaving them cached as failed/excluded")
    parser.add_argument("--division-workers", type=int, default=DEFAULT_DIVISION_WORKERS,
                         help=f"concurrent workers for division lookups (default {DEFAULT_DIVISION_WORKERS})")
    parser.add_argument("--division-rate-limit", type=float, default=DEFAULT_DIVISION_RATE_LIMIT,
                         help=f"max aggregate division-lookup requests/sec across all workers "
                              f"(default {DEFAULT_DIVISION_RATE_LIMIT}; API allows 5)")
    parser.add_argument("--inspect", action="store_true",
                         help="read-only debugging mode: prints full untruncated scoreboard JSON for "
                              "every game in --start/--end, always re-fetching, writes nothing.")
    parser.add_argument("--inspect-game", default=None,
                         help="read-only: prints full raw JSON from the per-game detail endpoint "
                              "for this game_id. Writes nothing.")
    args = parser.parse_args()

    if args.inspect_game:
        inspect_game(args.inspect_game)
        return

    start_str = args.start or f"{args.season}-08-29"
    end_str = args.end or f"{args.season}-11-09"
    start = datetime.strptime(start_str, "%Y-%m-%d").date()
    end = datetime.strptime(end_str, "%Y-%m-%d").date()

    if args.inspect:
        sport_path = SPORT_PATHS[args.sport]
        for d in _daterange(start, end):
            inspect_day(sport_path, d)
            time.sleep(REQUEST_DELAY_SECONDS)
    elif args.filter_only:
        filter_to_final(
            args.sport, out_dir=args.out_dir, debug=args.debug,
            retry_failed_divisions=args.retry_failed_divisions,
            workers=args.division_workers, rate_limit=args.division_rate_limit,
            season=args.season,
        )
    else:
        print(f"Collecting D3 {args.sport}, season {args.season}, {start} to {end} ...")
        collect_season(
            args.sport, start, end, out_dir=args.out_dir, debug=args.debug,
            retry_failed_divisions=args.retry_failed_divisions,
            workers=args.division_workers, rate_limit=args.division_rate_limit,
            season=args.season,
        )


if __name__ == "__main__":
    main()
