"""Agregacje pod slajdy prezentacji (dashboard)."""

from __future__ import annotations

import numpy as np
import pandas as pd

def top_teams_by_match_count(m: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    t1 = m["team1_name"].dropna().astype(str)
    t2 = m["team2_name"].dropna().astype(str)
    cnt = pd.concat([t1, t2], ignore_index=True).value_counts().head(n)
    return pd.DataFrame({"druzyna": cnt.index.astype(str), "mecze": cnt.values})


def team_winrates_long(m: pd.DataFrame) -> pd.DataFrame:
    t1 = m[["team1_name", "team1_win_flag"]].rename(columns={"team1_name": "druzyna"})
    t1["wygrana"] = t1["team1_win_flag"] == 1
    t2 = m[["team2_name", "team1_win_flag"]].rename(columns={"team2_name": "druzyna"})
    t2["wygrana"] = t2["team1_win_flag"] == 0
    long = pd.concat(
        [t1[["druzyna", "wygrana"]], t2[["druzyna", "wygrana"]]],
        ignore_index=True,
    )
    long = long.dropna(subset=["druzyna"])
    g = long.groupby("druzyna", as_index=False).agg(mecze=("wygrana", "count"), wygrane=("wygrana", "sum"))
    g["winrate"] = g["wygrane"] / g["mecze"]
    return g.sort_values("winrate", ascending=False)


def top_teams_by_winrate(m: pd.DataFrame, min_meczy: int, n: int = 10) -> pd.DataFrame:
    g = team_winrates_long(m)
    g = g[g["mecze"] >= min_meczy].head(n)
    return g


def top_maps_from_rounds(rounds: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    cols = [c for c in ("map1_name", "map2_name", "map3_name") if c in rounds.columns]
    if not cols:
        return pd.DataFrame(columns=["mapa", "liczba"])
    parts = []
    for c in cols:
        parts.append(rounds[c].dropna().astype(str))
    s = pd.concat(parts, ignore_index=True)
    cnt = s.value_counts().head(n).reset_index()
    cnt.columns = ["mapa", "liczba"]
    return cnt


def top_tournaments(m: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    if "tournament" not in m.columns:
        return pd.DataFrame(columns=["turniej", "mecze"])
    c = m["tournament"].dropna().astype(str).value_counts().head(n).reset_index()
    c.columns = ["turniej", "mecze"]
    return c


def player_rating_long(m: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for side in ("team1", "team2"):
        for i in range(1, 6):
            name_c = f"{side}_player_{i}_name"
            rat_c = f"{side}_player_{i}_RATING"
            if name_c not in m.columns or rat_c not in m.columns:
                continue
            sub = m[[name_c, rat_c]].copy()
            sub.columns = ["gracz", "rating"]
            sub["rating"] = pd.to_numeric(sub["rating"], errors="coerce")
            parts.append(sub)
    if not parts:
        return pd.DataFrame(columns=["gracz", "rating"])
    return pd.concat(parts, ignore_index=True).dropna(subset=["gracz"])


def top_players_by_rating(m: pd.DataFrame, min_wystapien: int, n: int = 10) -> pd.DataFrame:
    pl = player_rating_long(m)
    if pl.empty:
        return pd.DataFrame(columns=["gracz", "sredni_rating", "n", "sem"])
    g = pl.groupby("gracz", as_index=False).agg(
        sredni_rating=("rating", "mean"),
        n=("rating", "count"),
        sem=("rating", lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0),
    )
    g = g[g["n"] >= min_wystapien].sort_values("sredni_rating", ascending=False).head(n)
    return g


def upsets_table(m: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    req = ["team1_avg_RATING", "team2_avg_RATING", "rating_diff", "team1_win_flag"]
    if not all(c in m.columns for c in req):
        return pd.DataFrame()
    sub = m.dropna(subset=["team1_avg_RATING", "team2_avg_RATING", "team1_win_flag"]).copy()
    fav_t1 = sub["team1_avg_RATING"] > sub["team2_avg_RATING"]
    fav_t2 = sub["team2_avg_RATING"] > sub["team1_avg_RATING"]
    sub = sub.loc[fav_t1 | fav_t2].copy()
    fav_t1 = sub["team1_avg_RATING"] > sub["team2_avg_RATING"]
    fav_t2 = sub["team2_avg_RATING"] > sub["team1_avg_RATING"]
    upset = (fav_t1 & (sub["team1_win_flag"] == 0)) | (fav_t2 & (sub["team1_win_flag"] == 1))
    sub = sub.loc[upset].copy()
    sub["abs_rd"] = sub["rating_diff"].abs()
    sub = sub.sort_values("abs_rd", ascending=False)
    cols = [
        "date_parsed",
        "team1_name",
        "team2_name",
        "tournament",
        "team1_avg_RATING",
        "team2_avg_RATING",
        "rating_diff",
        "team1_win_flag",
    ]
    cols = [c for c in cols if c in sub.columns]
    return sub[cols].head(n)


def map1_round_margin_rows(rounds: pd.DataFrame) -> pd.DataFrame:
    win_cols = [c for c in rounds.columns if c.startswith("map1_round") and c.endswith("_winner")]
    if not win_cols or "map1_name" not in rounds.columns:
        return pd.DataFrame()
    r = rounds[["map1_name"] + win_cols].dropna(subset=["map1_name"]).copy()

    def margin(row: pd.Series) -> float:
        s = row[win_cols].astype(str).str.lower()
        return float((s == "team1").sum() - (s == "team2").sum())

    r["margines"] = r.apply(margin, axis=1)
    r["abs_m"] = r["margines"].abs()
    med = r.groupby("map1_name", as_index=False)["abs_m"].median().sort_values("abs_m", ascending=False)
    med.columns = ["mapa", "mediana_abs_marginesu"]
    return med


def score_series_heatmap_counts(m: pd.DataFrame) -> pd.DataFrame:
    if "score_team1" not in m.columns or "score_team2" not in m.columns:
        return pd.DataFrame()
    sub = m.dropna(subset=["score_team1", "score_team2"]).copy()
    sub["s1"] = sub["score_team1"].astype(int)
    sub["s2"] = sub["score_team2"].astype(int)
    return (
        sub.groupby(["s1", "s2"], as_index=False)
        .size()
        .rename(columns={"size": "liczba"})
    )


def season_match_counts(m: pd.DataFrame) -> pd.DataFrame:
    if "season" not in m.columns:
        return pd.DataFrame(columns=["season", "mecze"])
    g = m.dropna(subset=["season"]).groupby("season").size().reset_index(name="mecze")
    return g.sort_values("season")


def most_even_matches(m: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    if "rating_diff" not in m.columns:
        return pd.DataFrame()
    sub = m.dropna(subset=["rating_diff", "team1_name", "team2_name"]).copy()
    sub["abs_rd"] = sub["rating_diff"].abs()
    sub = sub.sort_values("abs_rd", ascending=True)
    cols = ["date_parsed", "team1_name", "team2_name", "tournament", "rating_diff", "team1_avg_RATING", "team2_avg_RATING"]
    cols = [c for c in cols if c in sub.columns]
    return sub[cols].head(n)


def winrate_monthly(mf: pd.DataFrame) -> pd.DataFrame:
    if mf.empty or "date_parsed" not in mf.columns:
        return pd.DataFrame()
    x = mf.dropna(subset=["date_parsed", "wygrala_wybrana"]).set_index("date_parsed").sort_index()
    g = x["wygrala_wybrana"].astype(float).resample("ME").mean().reset_index()
    g.columns = ["okres", "winrate"]
    return g


def winrate_by_event_type(mf: pd.DataFrame) -> pd.DataFrame:
    if mf.empty or "event_type" not in mf.columns:
        return pd.DataFrame()
    return (
        mf.dropna(subset=["wygrala_wybrana"])
        .groupby("event_type", as_index=False)
        .agg(winrate=("wygrala_wybrana", "mean"), mecze=("wygrala_wybrana", "count"))
    )


def favorite_winrate_by_rating_edge(
    m: pd.DataFrame,
    bins: list[float] | None = None,
) -> pd.DataFrame:
    """Skutecznosc faworyta wg przedzialu przewagi ratingu."""
    req = ["team1_avg_RATING", "team2_avg_RATING", "team1_win_flag", "rating_diff"]
    if not all(c in m.columns for c in req):
        return pd.DataFrame(columns=["bucket", "srodek", "faworyt_winrate", "mecze"])
    sub = m.dropna(subset=req).copy()
    sub = sub[sub["team1_avg_RATING"] != sub["team2_avg_RATING"]]
    if sub.empty:
        return pd.DataFrame(columns=["bucket", "srodek", "faworyt_winrate", "mecze"])
    fav_t1 = sub["team1_avg_RATING"] > sub["team2_avg_RATING"]
    sub["faworyt_wygral"] = np.where(
        fav_t1, sub["team1_win_flag"] == 1, sub["team1_win_flag"] == 0
    ).astype(float)
    sub["abs_rd"] = sub["rating_diff"].abs()
    if bins is None:
        bins = [0, 0.02, 0.05, 0.08, 0.12, 1.0]
    sub["bucket"] = pd.cut(sub["abs_rd"], bins=bins, include_lowest=True)
    g = (
        sub.groupby("bucket", observed=True)
        .agg(faworyt_winrate=("faworyt_wygral", "mean"), mecze=("faworyt_wygral", "size"))
        .reset_index()
    )
    g["srodek"] = g["bucket"].apply(lambda iv: (iv.left + iv.right) / 2)
    g["bucket"] = g["bucket"].astype(str)
    return g[["bucket", "srodek", "faworyt_winrate", "mecze"]]


def favorite_winrate_overall(m: pd.DataFrame) -> float | None:
    """Ogolny odsetek meczow wygranych przez druzyne z wyzszym ratingiem."""
    req = ["team1_avg_RATING", "team2_avg_RATING", "team1_win_flag"]
    if not all(c in m.columns for c in req):
        return None
    sub = m.dropna(subset=req)
    sub = sub[sub["team1_avg_RATING"] != sub["team2_avg_RATING"]]
    if sub.empty:
        return None
    fav_t1 = sub["team1_avg_RATING"] > sub["team2_avg_RATING"]
    fav_win = np.where(fav_t1, sub["team1_win_flag"] == 1, sub["team1_win_flag"] == 0)
    return float(np.mean(fav_win))


def team_winrate_by_event(
    m: pd.DataFrame,
    min_per_segment: int = 5,
    top_n: int = 10,
) -> pd.DataFrame:

    req = ["team1_name", "team2_name", "team1_win_flag", "event_type"]
    if not all(c in m.columns for c in req):
        return pd.DataFrame(columns=["druzyna", "event_type", "winrate", "mecze"])
    t1 = m[["team1_name", "team1_win_flag", "event_type"]].rename(
        columns={"team1_name": "druzyna"}
    )
    t1["wygrana"] = t1["team1_win_flag"] == 1
    t2 = m[["team2_name", "team1_win_flag", "event_type"]].rename(
        columns={"team2_name": "druzyna"}
    )
    t2["wygrana"] = t2["team1_win_flag"] == 0
    long = pd.concat(
        [t1[["druzyna", "event_type", "wygrana"]], t2[["druzyna", "event_type", "wygrana"]]],
        ignore_index=True,
    ).dropna(subset=["druzyna", "event_type"])
    long["event_type"] = long["event_type"].astype(str).str.upper()
    g = long.groupby(["druzyna", "event_type"], as_index=False).agg(
        winrate=("wygrana", "mean"), mecze=("wygrana", "count")
    )
    g = g[g["mecze"] >= min_per_segment]
    both = g.groupby("druzyna")["event_type"].nunique()
    keep = both[both >= 2].index
    g = g[g["druzyna"].isin(keep)]
    if g.empty:
        return g
    order = (
        g.groupby("druzyna")["mecze"].sum().sort_values(ascending=False).head(top_n).index
    )
    return g[g["druzyna"].isin(order)].sort_values(["druzyna", "event_type"])


def team_strength_vs_winrate(m: pd.DataFrame, min_meczy: int = 15) -> pd.DataFrame:
    """Sredni rating skladu i realny winrate per druzyna."""
    wr = team_winrates_long(m)
    if wr.empty:
        return pd.DataFrame(columns=["druzyna", "avg_rating", "winrate", "mecze"])
    req = ["team1_name", "team2_name", "team1_avg_RATING", "team2_avg_RATING"]
    if not all(c in m.columns for c in req):
        return pd.DataFrame(columns=["druzyna", "avg_rating", "winrate", "mecze"])
    s1 = m[["team1_name", "team1_avg_RATING"]].rename(
        columns={"team1_name": "druzyna", "team1_avg_RATING": "r"}
    )
    s2 = m[["team2_name", "team2_avg_RATING"]].rename(
        columns={"team2_name": "druzyna", "team2_avg_RATING": "r"}
    )
    strength = (
        pd.concat([s1, s2], ignore_index=True)
        .dropna(subset=["druzyna", "r"])
        .groupby("druzyna", as_index=False)["r"]
        .mean()
        .rename(columns={"r": "avg_rating"})
    )
    out = wr.merge(strength, on="druzyna", how="inner")
    out = out[out["mecze"] >= min_meczy]
    return out[["druzyna", "avg_rating", "winrate", "mecze"]].sort_values(
        "winrate", ascending=False
    )


def map_score_distribution(m: pd.DataFrame) -> pd.DataFrame:
    """Rozklad wynikow bo3 po odfiltrowaniu wynikow rund pojedynczych map."""
    if "score_team1" not in m.columns or "score_team2" not in m.columns:
        return pd.DataFrame(columns=["typ", "liczba", "udzial"])
    sub = m.dropna(subset=["score_team1", "score_team2"]).copy()
    sub["s1"] = pd.to_numeric(sub["score_team1"], errors="coerce")
    sub["s2"] = pd.to_numeric(sub["score_team2"], errors="coerce")
    sub = sub.dropna(subset=["s1", "s2"])
    sub = sub[(sub["s1"] <= 2) & (sub["s2"] <= 2)]
    sub["maxs"] = sub[["s1", "s2"]].max(axis=1)
    sub["mins"] = sub[["s1", "s2"]].min(axis=1)
    sub = sub[sub["maxs"] == 2]
    if sub.empty:
        return pd.DataFrame(columns=["typ", "liczba", "udzial"])
    sub["typ"] = np.where(sub["mins"] == 0, "2-0 (pewna)", "2-1 (decyder)")
    g = sub.groupby("typ", as_index=False).size().rename(columns={"size": "liczba"})
    g["udzial"] = g["liczba"] / g["liczba"].sum()
    return g.sort_values("typ")


def top_head_to_head_pairs(m: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Top N par drużyn (alfabetycznie A vs B), które grały ze sobą najczęściej."""
    if m.empty or "team1_name" not in m.columns:
        return pd.DataFrame(columns=["para", "mecze"])
    t1 = m["team1_name"].fillna("").astype(str)
    t2 = m["team2_name"].fillna("").astype(str)
    lo = t1.str.casefold() <= t2.str.casefold()
    a = np.where(lo, t1, t2)
    b = np.where(lo, t2, t1)
    pair = pd.Series(a, index=m.index, dtype=object).str.cat(
        pd.Series(b, index=m.index, dtype=object),
        sep=" vs ",
    )
    cnt = pair.value_counts().head(n).reset_index()
    cnt.columns = ["para", "mecze"]
    return cnt
