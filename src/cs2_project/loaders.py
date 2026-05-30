from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"


def project_root() -> Path:
    return PROJECT_ROOT


DATA_FILES = {
    "matches_full": "cs2_newestcombinedmatches.csv",
    "matches_modeling": "cs2_newestcombinedmatches_team1_reference_reduced2.csv",
    "rounds": "combined_round_by_round_with_map_names_cleaned.csv",
    "timeseries": "newest_ts_ds.csv",
}

CORE_TABLE_KEYS: tuple[str, ...] = ("matches_modeling", "rounds", "timeseries")


def extract_hltv_match_page_id(url: str | float | None) -> int | None:
    if url is None or (isinstance(url, float) and pd.isna(url)):
        return None
    m = re.search(r"/matches/(\d+)", str(url))
    return int(m.group(1)) if m else None


def normalize_team_pair_key(team1: str, team2: str) -> str:
    a, b = sorted([str(team1).strip().lower(), str(team2).strip().lower()])
    return f"{a}|{b}"


def load_tables(
    keys: Iterable[str] | None = None,
    raw_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Wczytuje wskazane tabele z ``data/raw``. Domyslnie wszystkie wpisy z DATA_FILES."""
    base = raw_dir or DEFAULT_RAW_DIR
    key_list = tuple(DATA_FILES.keys()) if keys is None else tuple(keys)
    out: dict[str, pd.DataFrame] = {}
    for key in key_list:
        if key not in DATA_FILES:
            raise KeyError(f"Nieznany klucz tabeli: {key!r}. Dostepne: {list(DATA_FILES)}")
        name = DATA_FILES[key]
        path = base / name
        if not path.exists():
            raise FileNotFoundError(
                f"Brak pliku {path}. Pobierz zbiory Kaggle do data/raw (patrz README)."
            )
        out[key] = pd.read_csv(path, low_memory=False)
    return out


def load_core_tables(raw_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Mecze (reduced2) + rundy + time-series — wystarcza do dashboardu i notatnika."""
    return load_tables(CORE_TABLE_KEYS, raw_dir=raw_dir)


def load_all_tables(raw_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Wszystkie cztery pliki, jesli potrzebujesz pelnego ``cs2_newestcombinedmatches.csv``."""
    return load_tables(None, raw_dir=raw_dir)


def merge_matches_with_rounds(
    matches: pd.DataFrame,
    rounds: pd.DataFrame,
    *,
    how: str = "left",
) -> pd.DataFrame:
    m = matches.copy()
    r = rounds.copy()
    m["match_page_id"] = m["hltv_url"].map(extract_hltv_match_page_id)
    r["match_page_id"] = r["match_url"].map(extract_hltv_match_page_id)
    return m.merge(r, on="match_page_id", how=how, suffixes=("_match", "_round"))


def merge_matches_with_timeseries(
    matches: pd.DataFrame,
    ts: pd.DataFrame,
    *,
    how: str = "left",
) -> pd.DataFrame:
    m = matches.copy()
    t = ts.copy()
    m["date_norm"] = pd.to_datetime(m["date"], utc=True, errors="coerce").dt.normalize()
    t["date_norm"] = pd.to_datetime(t["date"], utc=True, errors="coerce").dt.normalize()
    m["team_pair_key"] = [
        normalize_team_pair_key(a, b) for a, b in zip(m["team1_name"], m["team2_name"])
    ]
    t["team_pair_key"] = [
        normalize_team_pair_key(a, b) for a, b in zip(t["team1_name"], t["team2_name"])
    ]
    return m.merge(
        t,
        on=["date_norm", "team_pair_key"],
        how=how,
        suffixes=("_match", "_ts"),
    )
