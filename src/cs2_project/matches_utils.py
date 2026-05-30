from __future__ import annotations

import numpy as np
import pandas as pd


def prepare_matches(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date_parsed"] = pd.to_datetime(out["date"], utc=True, errors="coerce")
    for c in (
        "rating_diff",
        "adr_diff",
        "team1_avg_RATING",
        "team2_avg_RATING",
        "team1_win_flag",
        "score_team1",
        "score_team2",
        "season",
    ):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def add_real_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    w = out["team1_win_flag"].fillna(-1).astype(int)
    out["zwyciezca"] = np.where(w == 1, out["team1_name"], out["team2_name"])
    out["pokonany"] = np.where(w == 1, out["team2_name"], out["team1_name"])
    out["wynik_slownie"] = np.where(
        w == 1,
        out["team1_name"].astype(str) + " wygral(a)",
        out["team2_name"].astype(str) + " wygral(a)",
    )
    return out


def unique_teams(df: pd.DataFrame) -> list[str]:
    t1 = df["team1_name"].dropna().astype(str)
    t2 = df["team2_name"].dropna().astype(str)
    return sorted(set(t1) | set(t2), key=str.casefold)


def perspective_focus(df: pd.DataFrame, focus: str) -> pd.DataFrame:
    m = df[(df["team1_name"] == focus) | (df["team2_name"] == focus)].copy()
    if m.empty:
        return m
    is_t1 = m["team1_name"] == focus
    m["przeciwnik"] = np.where(is_t1, m["team2_name"], m["team1_name"])
    m["wygrala_wybrana"] = np.where(
        is_t1,
        m["team1_win_flag"] == 1,
        m["team1_win_flag"] == 0,
    )
    m["przewaga_rating_wybranej"] = np.where(
        is_t1,
        m["rating_diff"],
        -m["rating_diff"],
    )
    m["przewaga_adr_wybranej"] = np.where(
        is_t1,
        m["adr_diff"],
        -m["adr_diff"],
    )
    return m
