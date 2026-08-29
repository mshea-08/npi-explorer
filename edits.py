"""
Pure data-merge logic for user-editable game data, used by app.py.

`sport` here is the same sport key used throughout app.py and
collector.py (e.g. "mens_soccer", "womens_volleyball") -- it doubles
directly as the CSV file label collector.py writes to
(d3_{sport}_{season}.csv), so no separate name-mapping lives here.

Two data sources get merged into one "effective" set of games per
(season, sport):
  - a base CSV collected by collector.py (only exists for the 2025 season)
  - an "edits" dict layered on top, recording:
      "removed":   game_ids from the base CSV the user deleted
      "overrides": {game_id: {changed_field: new_value, ...}} for base
                   games the user edited (score corrections, etc.)
      "added":     brand-new games the user entered by hand, each with a
                   generated "manual_..." game_id

2026 has no base CSV at all -- every 2026 game is a manual addition, so
the exact same merge logic handles both seasons: for 2026, base_df is
just empty, "removed"/"overrides" are no-ops, and "added" is the entire
season.

DELIBERATELY NO PERSISTENCE HERE. Edits live in Streamlit's
st.session_state (see app.py), not on disk -- each browser session is a
separate Streamlit session with its own isolated session_state, so one
person's score edits or added games are only ever visible to them, never
to anyone else using the app. This module only knows how to merge and
diff in-memory data; it never reads or writes a file, on purpose. This
also means it's plain Python with no Streamlit dependency, so it's
directly unit-testable outside the app.
"""

import os
import uuid

import pandas as pd

FIELDNAMES = ["date", "home_team", "away_team", "home_score", "away_score", "status", "game_id", "neutral_site"]
EDITABLE_FIELDS = ("date", "home_team", "away_team", "home_score", "away_score", "neutral_site")

EMPTY_EDITS = {"removed": [], "overrides": {}, "added": []}


def base_csv_path(season: str, sport: str, out_dir: str = "."):
    """Only 2025 has a collected base file. Returns None for other seasons."""
    if season != "2025":
        return None
    return os.path.join(out_dir, f"d3_{sport}_{season}.csv")


def load_base(season: str, sport: str, out_dir: str = ".") -> pd.DataFrame:
    """The one piece of real disk I/O in this module -- reading the
    collector's output. This is shared, read-only, and the same for every
    user, which is exactly why it's fine for it to come from disk."""
    path = base_csv_path(season, sport, out_dir)
    if path is None or not os.path.exists(path):
        return pd.DataFrame(columns=FIELDNAMES)
    df = pd.read_csv(path, dtype={"game_id": str})
    for col in FIELDNAMES:
        if col not in df.columns:
            df[col] = None
    return df[FIELDNAMES]


def is_manual_id(game_id) -> bool:
    return isinstance(game_id, str) and game_id.startswith("manual_")


def new_manual_id() -> str:
    return "manual_" + uuid.uuid4().hex[:10]


def has_edits(edits: dict) -> bool:
    return bool(edits.get("removed") or edits.get("overrides") or edits.get("added"))


def apply_edits(base_df: pd.DataFrame, edits: dict) -> pd.DataFrame:
    """Returns the effective games dataframe: base minus removed games,
    with overrides applied, plus added rows. This is what NPI gets
    computed on."""
    df = base_df.copy()

    if not df.empty:
        removed = set(str(g) for g in edits.get("removed", []))
        if removed:
            df = df[~df["game_id"].astype(str).isin(removed)]

        overrides = edits.get("overrides", {})
        for gid, fields in overrides.items():
            mask = df["game_id"].astype(str) == str(gid)
            for col, val in fields.items():
                if col in df.columns:
                    df.loc[mask, col] = val

    added = edits.get("added", [])
    if added:
        added_df = pd.DataFrame(added)
        for col in FIELDNAMES:
            if col not in added_df.columns:
                added_df[col] = None
        df = pd.concat([df, added_df[FIELDNAMES]], ignore_index=True)

    if df.empty:
        return pd.DataFrame(columns=FIELDNAMES)

    df = df[FIELDNAMES].reset_index(drop=True)
    return df


def _is_blank(val) -> bool:
    if val is None:
        return True
    if isinstance(val, float) and pd.isna(val):
        return True
    if isinstance(val, str) and val.strip() == "":
        return True
    return False


def reconcile_edit(before_df: pd.DataFrame, after_df: pd.DataFrame, edits: dict) -> dict:
    """
    Diffs what the user submitted (after_df, from a data_editor) against
    what they were shown before editing (before_df, the previous effective
    dataframe -- real game_ids for base rows, "manual_..." ids for
    previously-added rows). New rows from the editor have a blank
    game_id. Returns an updated edits dict capturing:
      - base rows the user deleted -> added to "removed"
      - base rows the user changed -> recorded in "overrides"
      - manual rows the user deleted -> dropped from "added"
      - manual rows the user changed -> updated in "added"
      - brand-new rows -> appended to "added" with a fresh manual id
    """
    edits = {
        "removed": list(edits.get("removed", [])),
        "overrides": {k: dict(v) for k, v in edits.get("overrides", {}).items()},
        "added": [dict(row) for row in edits.get("added", [])],
    }
    removed_set = set(str(g) for g in edits["removed"])
    overrides = edits["overrides"]
    added_by_id = {str(row["game_id"]): row for row in edits["added"]}

    before_by_id = {str(row["game_id"]): row.to_dict() for _, row in before_df.iterrows()}

    seen_ids = set()
    for row in after_df.to_dict("records"):
        raw_gid = row.get("game_id")
        gid = None if _is_blank(raw_gid) else str(raw_gid)

        if gid is None:
            new_id = new_manual_id()
            new_row = {f: row.get(f) for f in EDITABLE_FIELDS}
            new_row["status"] = row.get("status") or "FINAL"
            new_row["game_id"] = new_id
            added_by_id[new_id] = new_row
            seen_ids.add(new_id)
            continue

        seen_ids.add(gid)

        if is_manual_id(gid):
            updated = {f: row.get(f) for f in EDITABLE_FIELDS}
            updated["status"] = row.get("status") or added_by_id.get(gid, {}).get("status", "FINAL")
            updated["game_id"] = gid
            added_by_id[gid] = updated
            continue

        before_row = before_by_id.get(gid)
        if before_row is None:
            continue
        changed_fields = {}
        for col in EDITABLE_FIELDS:
            if str(row.get(col)) != str(before_row.get(col)):
                changed_fields[col] = row.get(col)
        if changed_fields:
            existing = overrides.get(gid, {})
            existing.update(changed_fields)
            overrides[gid] = existing

    for gid in before_by_id:
        if gid in seen_ids:
            continue
        if is_manual_id(gid):
            added_by_id.pop(gid, None)
        else:
            removed_set.add(gid)

    edits["removed"] = sorted(removed_set)
    edits["overrides"] = overrides
    edits["added"] = list(added_by_id.values())
    return edits
