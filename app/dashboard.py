from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cs2_project.loaders import (  
    extract_hltv_match_page_id,
    load_core_tables,
)
from cs2_project import presentation as pr  
from cs2_project.matches_utils import (  
    add_real_names,
    perspective_focus,
    prepare_matches,
    unique_teams,
)
from cs2_project.prediction import ( 
    FEATURE_LABELS,
    LOWER_IS_BETTER,
    explain_prediction,
    load_or_train,
    predict_matchup,
)

ACCENT = "#c23b22"
BLUE = "#1f77b4"
GREEN = "#2ca02c"
EVENT_COLORS = {"LAN": ACCENT, "ONLINE": BLUE}


@st.cache_data(show_spinner=False)
def load_frames() -> dict[str, pd.DataFrame]:
    return load_core_tables()


@st.cache_resource(show_spinner="Trenuje model predykcji (tylko przy pierwszym uruchomieniu)...")
def get_predictor(_matches: pd.DataFrame) -> dict:
    """Wczytuje lub trenuje artefakt modelu predykcji."""
    return load_or_train(_matches)


RADAR_FEATURES = ["avg_RATING", "avg_ADR", "avg_KAST", "avg_KPR", "avg_DPR", "past3"]


def _head_to_head(matches: pd.DataFrame, team_a: str, team_b: str) -> tuple[int, int, int]:
    """Bilans bezposrednich spotkan A vs B z pelnej historii: (mecze, wygrane_A, wygrane_B)."""
    req = {"team1_name", "team2_name", "team1_win_flag"}
    if not req.issubset(matches.columns):
        return 0, 0, 0
    pair = matches[
        ((matches["team1_name"] == team_a) & (matches["team2_name"] == team_b))
        | ((matches["team1_name"] == team_b) & (matches["team2_name"] == team_a))
    ]
    if pair.empty:
        return 0, 0, 0
    flag = pd.to_numeric(pair["team1_win_flag"], errors="coerce")
    a_wins = int(
        (((pair["team1_name"] == team_a) & (flag == 1)) | ((pair["team2_name"] == team_a) & (flag == 0))).sum()
    )
    b_wins = int(
        (((pair["team1_name"] == team_b) & (flag == 1)) | ((pair["team2_name"] == team_b) & (flag == 0))).sum()
    )
    return a_wins + b_wins, a_wins, b_wins


def _probability_bar(team_a: str, team_b: str, proba_a: float, proba_b: float) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(
        y=["szansa"], x=[proba_a], orientation="h", name=team_a,
        marker_color=ACCENT, text=[f"{team_a} {proba_a:.0%}"], textposition="inside",
    )
    fig.add_bar(
        y=["szansa"], x=[proba_b], orientation="h", name=team_b,
        marker_color=BLUE, text=[f"{team_b} {proba_b:.0%}"], textposition="inside",
    )
    fig.update_layout(
        barmode="stack", height=130, showlegend=False,
        xaxis=dict(range=[0, 1], tickformat=".0%", title=""),
        yaxis=dict(showticklabels=False),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig


def render_match_prediction(matches: pd.DataFrame, focus: str | None) -> None:
    st.markdown("#### Kto wygra? Typ modelu dla pary druzyn A vs B")
    st.caption(
        "Model klasyfikacji uczony na ROZNICACH profili obu druzyn (rating skladu, ADR, KAST, "
        "KPR, DPR, spojnosc skladu, gwiazda/najslabszy gracz, winrate per mapa, forma). Cechy "
        "znajace wynik meczu (wynik mapy, kumulatywne winrate'y) sa celowo wykluczone, by uniknac "
        "przeciekow danych. To ocena „jak silne sa te druzyny ogolem”, nie typ na konkretny termin."
    )

    try:
        art = get_predictor(matches)
    except Exception as e:  # noqa: BLE001
        st.error(f"Nie udalo sie wczytac/wytrenowac modelu: {e}")
        return

    profiles: pd.DataFrame = art["profiles"]
    teams = list(profiles.index)
    if len(teams) < 2:
        st.warning("Za malo druzyn z profilem, aby wykonac predykcje.")
        return

    default_a = focus if (focus in teams) else teams[0]
    idx_a = teams.index(default_a)
    idx_b = 1 if idx_a == 0 else 0

    csa, csb = st.columns(2)
    team_a = csa.selectbox("Druzyna A", teams, index=idx_a, key="pred_team_a")
    team_b = csb.selectbox("Druzyna B", teams, index=idx_b, key="pred_team_b")

    if team_a == team_b:
        st.warning("Wybierz dwie ROZNE druzyny.")
        return

    res = predict_matchup(art, team_a, team_b)
    proba_a, proba_b = res["proba_a"], res["proba_b"]

    m1, m2, m3 = st.columns([2, 2, 3])
    m1.metric(f"{team_a}", f"{proba_a:.0%}")
    m2.metric(f"{team_b}", f"{proba_b:.0%}")
    conf = abs(proba_a - proba_b)
    pewnosc = "wyrazny faworyt" if conf >= 0.20 else ("lekki faworyt" if conf >= 0.08 else "mecz wyrownany")
    m3.metric("Typ modelu", res["winner"], help="Druzyna z wyzszym prawdopodobienstwem.")
    st.plotly_chart(_probability_bar(team_a, team_b, proba_a, proba_b), use_container_width=True)
    st.caption(f"Pewnosc typu: **{pewnosc}** (roznica szans {conf:.0%}).")

    st.markdown("---")

    st.markdown("##### Porownanie druzyn")
    cradar, ctab = st.columns([3, 2])

    radar_feats = [f for f in RADAR_FEATURES if f in profiles.columns]
    norm = profiles[radar_feats].astype(float).copy()
    for f in radar_feats:
        lo, hi = norm[f].min(), norm[f].max()
        rng = (hi - lo) or 1.0
        norm[f] = (norm[f] - lo) / rng
        if f in LOWER_IS_BETTER:
            norm[f] = 1.0 - norm[f]
    theta = [FEATURE_LABELS.get(f, f) for f in radar_feats]

    with cradar:
        fig_r = go.Figure()
        for team, color in ((team_a, ACCENT), (team_b, BLUE)):
            vals = norm.loc[team, radar_feats].tolist()
            fig_r.add_trace(
                go.Scatterpolar(
                    r=vals + vals[:1], theta=theta + theta[:1], fill="toself",
                    name=team, line_color=color,
                )
            )
        fig_r.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1], showticklabels=False)),
            title="Profil sily (znormalizowany; dalej od srodka = lepiej)",
            margin=dict(l=40, r=40, t=60, b=40),
        )
        st.plotly_chart(fig_r, use_container_width=True)

    with ctab:
        rows = []
        for f in radar_feats:
            rows.append(
                {
                    "Cecha": FEATURE_LABELS.get(f, f),
                    team_a: round(float(profiles.loc[team_a, f]), 2),
                    team_b: round(float(profiles.loc[team_b, f]), 2),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        n_h2h, a_w, b_w = _head_to_head(matches, team_a, team_b)
        if n_h2h > 0:
            st.caption(f"**Bilans bezposredni (cala historia):** {team_a} {a_w} – {b_w} {team_b} ({n_h2h} meczow).")
        else:
            st.caption("Brak bezposrednich spotkan tych druzyn w danych.")

    st.markdown("---")

    st.markdown("##### Dlaczego taki typ?")
    ex = explain_prediction(art, team_a, team_b, top_n=8)
    contrib = ex["contributions"]

    if contrib:
        cdf = pd.DataFrame(
            {
                "cecha": [FEATURE_LABELS.get(b, b) for b in contrib],
                "wklad": list(contrib.values()),
            }
        )
        cdf["na_korzysc"] = np.where(cdf["wklad"] >= 0, team_a, team_b)
        fig_c = px.bar(
            cdf, x="wklad", y="cecha", orientation="h", color="na_korzysc",
            color_discrete_map={team_a: ACCENT, team_b: BLUE},
            title="Wklad cech w typ (w prawo = na korzysc A, w lewo = na korzysc B)",
            labels={"wklad": "wplyw na predykcje", "cecha": "", "na_korzysc": "na korzysc"},
        )
        fig_c.update_layout(yaxis={"categoryorder": "total ascending"})
        fig_c.add_vline(x=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig_c, use_container_width=True)

        top3 = list(contrib.items())[:3]
        zdania = []
        for base, val in top3:
            kto = team_a if val >= 0 else team_b
            zdania.append(f"**{FEATURE_LABELS.get(base, base)}** przemawia za {kto}")
        st.markdown(
            f"Najwiekszy wplyw na typ ma: " + "; ".join(zdania) + ". "
            f"Pozostale cechy przewazaja w mniejszym stopniu."
        )
        st.caption(f"Metoda wyjasnienia: {ex['method']}.")

    imp = art.get("importance", {})
    if imp:
        idf = (
            pd.DataFrame({"cecha": list(imp.keys()), "waznosc": list(imp.values())})
            .sort_values("waznosc", ascending=False)
            .head(10)
        )
        idf["cecha"] = idf["cecha"].map(lambda b: FEATURE_LABELS.get(b, b))
        with st.expander("Globalna waznosc cech (caly model)"):
            fig_i = px.bar(
                idf, x="waznosc", y="cecha", orientation="h",
                color_discrete_sequence=[GREEN],
                title="Ktore cechy model wykorzystuje najczesciej (srednio dla wszystkich meczow)",
                labels={"waznosc": "waznosc", "cecha": ""},
            )
            fig_i.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_i, use_container_width=True)

    with st.expander("Jakosc modelu — na ile mu ufac"):
        all_metrics = art.get("all_models_metrics", {})
        if all_metrics:
            mtab = pd.DataFrame(all_metrics).T
            show_cols = [c for c in ["accuracy", "precision", "recall", "f1", "roc_auc", "cv_accuracy_mean"] if c in mtab.columns]
            mtab = mtab[show_cols].rename(
                columns={
                    "accuracy": "Accuracy", "precision": "Precision", "recall": "Recall",
                    "f1": "F1", "roc_auc": "ROC-AUC", "cv_accuracy_mean": "CV acc (5-fold)",
                }
            ).round(3)
            st.markdown(f"Wybrany model: **{art.get('model_name', '?')}** "
                        f"(trening {art.get('n_train', 0)} / test {art.get('n_test', 0)} meczow).")
            st.dataframe(mtab, use_container_width=True)

        base = art.get("baseline")
        if base:
            st.caption(
                f"Baseline „typuj wyzej oceniana druzyne”: accuracy {base['accuracy']:.1%}, "
                f"F1 {base['f1']:.2f}. Model powinien byc nie gorszy od tej prostej reguly."
            )

        cm = art.get("metrics", {}).get("confusion_matrix")
        if cm:
            fig_cm = px.imshow(
                cm, text_auto=True, color_continuous_scale="Reds",
                labels=dict(x="Predykcja", y="Prawda", color="liczba"),
                x=["B wygra (0)", "A wygra (1)"], y=["B wygra (0)", "A wygra (1)"],
                title="Macierz pomylek (zbior testowy)",
            )
            st.plotly_chart(fig_cm, use_container_width=True)
        st.caption(
            "Predykcja wynikow CS2 jest z natury trudna — nawet dobry model bije „rzut moneta” "
            "tylko o kilka-kilkanascie punktow. Traktuj typ jako wsparcie, nie pewnik."
        )


def main() -> None:
    st.set_page_config(
        page_title="CS2 HLTV — dashboard",
        page_icon="\U0001f3ae",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("CS2 — profesjonalne mecze (HLTV / Kaggle)")
    st.caption(
        "Zrodlo danych: Griffin DesRoches na Kaggle. Cel: wsparcie decyzji o inwestycji "
        "w druzyny oraz ocena przewidywalnosci wynikow. Wszystkie wykresy reaguja na filtry z panelu."
    )

    try:
        tables = load_frames()
    except FileNotFoundError as e:
        st.error(str(e))
        st.info(
            "Potrzebne 3 pliki CSV w `data/raw/`: zbiory Kaggle "
            "`cs2-hltv-professional-match-statistics-dataset`, "
            "`cs2-professional-round-by-round-statistics-dataset`, "
            "`cs2-professional-hltv-match-data-time-series` "
            "(polecenie: `kaggle datasets download -d ... --unzip` w `data/raw/`). "
            "Szczegoly w README."
        )
        return

    matches = prepare_matches(tables["matches_modeling"])
    rounds = tables["rounds"].copy()

    matches_f = matches.dropna(subset=["date_parsed"])
    if "event_type" in matches_f.columns:
        matches_f = matches_f.copy()
        matches_f["event_type"] = (
            matches_f["event_type"].fillna("").astype(str).str.strip().str.upper()
        )
        matches_f.loc[matches_f["event_type"] == "", "event_type"] = np.nan

    dmin = matches_f["date_parsed"].min().date()
    dmax = matches_f["date_parsed"].max().date()

    all_teams = unique_teams(matches_f)

    ev_options = sorted(
        matches_f["event_type"].dropna().astype(str).unique(),
        key=str.casefold,
    )

    with st.sidebar:
        st.header("Filtry")
        dr = st.date_input("Zakres dat (UTC)", value=(dmin, dmax), min_value=dmin, max_value=dmax)
        if isinstance(dr, tuple) and len(dr) == 2:
            d0, d1 = dr
        else:
            d0, d1 = dmin, dmax
        ev = st.multiselect(
            "Typ eventu (LAN / ONLINE)",
            options=ev_options,
            default=ev_options,
        )
        st.markdown("---")
        focus = st.selectbox(
            "Druzyna (profil + KPI dla niej)",
            options=[None] + all_teams,
            format_func=lambda x: "— wszystkie mecze (ogolnie) —" if x is None else str(x),
            index=0,
        )
        st.markdown("---")
        st.markdown("**Progi rankingow**")
        min_meczy_wr = st.number_input(
            "Min. meczow (winrate / sila druzyn)", min_value=1, value=15, step=1
        )
        min_wyst_gracz = st.number_input(
            "Min. wystapien gracza (Top graczy)", min_value=5, value=30, step=1
        )

    ev_filter = ev if len(ev) > 0 else ev_options

    m = matches_f[
        (matches_f["date_parsed"].dt.date >= d0)
        & (matches_f["date_parsed"].dt.date <= d1)
        & (matches_f["event_type"].astype(str).isin(ev_filter))
    ].copy()
    m = add_real_names(m)

    match_ids = set(m["hltv_url"].map(extract_hltv_match_page_id).dropna().astype(int))
    rounds_f = rounds[
        rounds["match_url"].map(extract_hltv_match_page_id).isin(match_ids)
    ].copy()

    st.subheader("Przeglad")
    st.caption(f"Okres: {d0} → {d1} · typ eventu: {', '.join(ev_filter)}")
    fav_overall = pr.favorite_winrate_overall(m)
    score_dist = pr.map_score_distribution(m)
    pct_20 = (
        float(score_dist.loc[score_dist["typ"] == "2-0 (pewna)", "udzial"].iloc[0])
        if not score_dist.empty and (score_dist["typ"] == "2-0 (pewna)").any()
        else None
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mecze (po filtrach)", f"{len(m):,}")
    c2.metric("Unikalnych druzyn", len(unique_teams(m)))
    c3.metric(
        "Niespodzianki (faworyt przegral)",
        f"{1 - fav_overall:.1%}" if fav_overall is not None else "—",
        help="Udzial meczow, w ktorych druzyna z nizszym sredanim ratingiem wygrala. "
        "Im wyzej, tym mniej przewidywalna scena.",
    )
    c4.metric(
        "Pewne zwyciestwa 2-0",
        f"{pct_20:.1%}" if pct_20 is not None else "—",
        help="Udzial rozstrzygnietych bo3 zakonczonych 2-0 (reszta to decyder 2-1).",
    )

    tab_teams, tab_pred, tab_predict, tab_players, tab_maps, tab_profile = st.tabs(
        [
            "Druzyny (inwestycja)",
            "Predykcyjnosc",
            "Predykcja meczu",
            "Gracze",
            "Mapy",
            "Druzyna · profil",
        ]
    )

    with tab_teams:
        st.markdown("#### Kto wygrywa i czy sila skladu to potwierdza")

        d_wr = pr.top_teams_by_winrate(m, min_meczy_wr)
        if d_wr.empty:
            st.warning(f"Za malo danych przy progu min. {min_meczy_wr} meczow — zmniejsz prog w panelu.")
        else:
            d_wr = d_wr.copy()
            d_wr["etykieta"] = (d_wr["winrate"] * 100).round(0).astype(int).astype(str) + "% (" + d_wr["mecze"].astype(str) + " m.)"
            fig = px.bar(
                d_wr,
                x="winrate",
                y="druzyna",
                orientation="h",
                text="etykieta",
                color_discrete_sequence=[ACCENT],
                title=f"Najskuteczniejsze druzyny: udzial wygranych meczow (min. {min_meczy_wr} meczow)",
                labels={"winrate": "winrate", "druzyna": ""},
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"}, xaxis_tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

        d_sw = pr.team_strength_vs_winrate(m, min_meczy_wr)
        if len(d_sw) >= 5:
            fig2 = px.scatter(
                d_sw,
                x="avg_rating",
                y="winrate",
                size="mecze",
                hover_name="druzyna",
                color_discrete_sequence=[BLUE],
                title="Sila skladu vs realny winrate — punkty nad linia to druzyny przebijajace swoja „papierowa” sile",
                labels={"avg_rating": "sredni rating rosteru", "winrate": "winrate", "mecze": "mecze"},
            )
            a, b = np.polyfit(d_sw["avg_rating"], d_sw["winrate"], 1)
            xs = np.linspace(d_sw["avg_rating"].min(), d_sw["avg_rating"].max(), 50)
            fig2.add_trace(
                go.Scatter(
                    x=xs, y=a * xs + b, mode="lines",
                    line=dict(dash="dash", color="gray"), name="trend",
                )
            )
            fig2.update_yaxes(tickformat=".0%")
            corr = d_sw["avg_rating"].corr(d_sw["winrate"])
            st.plotly_chart(fig2, use_container_width=True)
            st.caption(
                f"Korelacja sila↔winrate = {corr:.2f} (umiarkowanie dodatnia). "
                "Druzyny wyraznie powyzej linii dowoza wiecej, niz wynikaloby z ratingu — to cel taniego scoutingu."
            )

        d_ev = pr.team_winrate_by_event(m, min_per_segment=5, top_n=10)
        if not d_ev.empty:
            fig3 = px.bar(
                d_ev,
                x="druzyna",
                y="winrate",
                color="event_type",
                barmode="group",
                color_discrete_map=EVENT_COLORS,
                title="LAN vs Online: ktore druzyny „dowoza” na turniejach stacjonarnych (min. 5 meczow/segment)",
                labels={"winrate": "winrate", "druzyna": "", "event_type": "typ eventu"},
            )
            fig3.update_yaxes(tickformat=".0%")
            fig3.update_xaxes(tickangle=45)
            st.plotly_chart(fig3, use_container_width=True)

        d_cnt = pr.top_teams_by_match_count(m)
        if not d_cnt.empty:
            fig4 = px.bar(
                d_cnt,
                x="mecze",
                y="druzyna",
                orientation="h",
                color_discrete_sequence=[GREEN],
                title="Najaktywniejsze druzyny w probie (liczba rozegranych meczow)",
                labels={"mecze": "mecze", "druzyna": ""},
            )
            fig4.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig4, use_container_width=True)

        d_h2h = pr.top_head_to_head_pairs(m)
        if not d_h2h.empty:
            fig5 = px.bar(
                d_h2h,
                x="mecze",
                y="para",
                orientation="h",
                color_discrete_sequence=[ACCENT],
                title="Najczestsze rywalizacje (pary druzyn) — naturalni kandydaci na produkty/transmisje",
                labels={"mecze": "liczba spotkan", "para": ""},
            )
            fig5.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig5, use_container_width=True)

    with tab_pred:
        st.markdown("#### Czy sila przed meczem przewiduje wynik")

        d_cal = pr.favorite_winrate_by_rating_edge(m)
        if not d_cal.empty:
            d_cal = d_cal.copy()
            p = d_cal["faworyt_winrate"].clip(0, 1)
            d_cal["se"] = np.sqrt(p * (1 - p) / d_cal["mecze"].clip(lower=1))
            fig_a = px.line(
                d_cal,
                x="bucket",
                y="faworyt_winrate",
                markers=True,
                error_y="se",
                color_discrete_sequence=[ACCENT],
                title="Im wieksza przewaga ratingu przed meczem, tym wyzsza szansa faworyta",
                labels={"bucket": "przewaga ratingu (|team1 − team2|)", "faworyt_winrate": "winrate faworyta"},
            )
            fig_a.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="rzut moneta (50%)")
            fig_a.update_yaxes(tickformat=".0%", range=[0.45, 0.75])
            fig_a.update_traces(text=d_cal["mecze"], textposition="top center")
            st.plotly_chart(fig_a, use_container_width=True)
            st.caption(
                "Przy minimalnej przewadze wynik to niemal rzut moneta (~51%); przy duzej (>0.12) "
                "faworyt wygrywa ~68% meczow. Rating jest sensowna, ale nie pewna podstawa typowania."
            )

        st.markdown("##### Niespodzianki (faworyt wg ratingu przegral)")
        ups = pr.upsets_table(m)
        if fav_overall is not None:
            st.metric("Udzial niespodzianek w probie", f"{1 - fav_overall:.1%}")
        if not ups.empty:
            ups_show = ups.copy()
            for c in ("team1_avg_RATING", "team2_avg_RATING", "rating_diff"):
                if c in ups_show.columns:
                    ups_show[c] = ups_show[c].round(3)
            if "team1_win_flag" in ups_show.columns:
                ups_show["zwyciezca"] = np.where(
                    ups_show["team1_win_flag"] == 1, ups_show["team1_name"], ups_show["team2_name"]
                )
            cols = [c for c in ["date_parsed", "team1_name", "team2_name", "zwyciezca", "tournament", "rating_diff"] if c in ups_show.columns]
            st.dataframe(ups_show[cols], use_container_width=True, hide_index=True)
            st.caption(
                "Co druga-trzecia faworyzowana druzyna potrafi przegrac — to argument za dywersyfikacja "
                "portfela druzyn/typow zamiast stawiania na jednego lidera."
            )

        if not score_dist.empty:
            fig_d = go.Figure(
                data=[
                    go.Pie(
                        labels=score_dist["typ"],
                        values=score_dist["liczba"],
                        hole=0.45,
                        marker=dict(colors=[ACCENT, BLUE]),
                    )
                ]
            )
            fig_d.update_layout(title="Jak pewne sa zwyciestwa: 2-0 vs decyder 2-1 (rozstrzygniete bo3)")
            st.plotly_chart(fig_d, use_container_width=True)

    with tab_predict:
        render_match_prediction(matches, focus)

    with tab_players:
        st.markdown("#### Najlepsi zawodnicy — cel scoutingu")
        d_pl = pr.top_players_by_rating(m, min_wyst_gracz)
        if d_pl.empty:
            st.warning(f"Za malo danych przy progu min. {min_wyst_gracz} wystapien — zmniejsz prog w panelu.")
        else:
            fig_p = px.bar(
                d_pl,
                x="sredni_rating",
                y="gracz",
                orientation="h",
                error_x="sem",
                color_discrete_sequence=[ACCENT],
                title=f"Top zawodnicy po srednim ratingu (min. {min_wyst_gracz} wystapien; wasy = blad standardowy)",
                labels={"sredni_rating": "sredni rating", "gracz": ""},
            )
            fig_p.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_p, use_container_width=True)
            st.caption(
                "Nakladajace sie wasy oznaczaja, ze roznica miedzy graczami jest w granicach bledu — "
                "nie przeplacaj za pojedyncze miejsce w rankingu."
            )

    with tab_maps:
        st.markdown("#### Pula map")
        d_maps = pr.top_maps_from_rounds(rounds_f)
        if d_maps.empty:
            st.info("Brak danych rund po filtrach.")
        else:
            fig_m = px.bar(
                d_maps,
                x="liczba",
                y="mapa",
                orientation="h",
                color_discrete_sequence=[GREEN],
                title="Najczesciej grane mapy w probie (priorytet treningu i analizy)",
                labels={"liczba": "wystapienia", "mapa": ""},
            )
            fig_m.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_m, use_container_width=True)

        if focus is not None:
            st.markdown(f"##### Przewaga {focus} w rundach mapy 1 (wg nazwy mapy)")
            win_cols = [c for c in rounds_f.columns if c.startswith("map1_round") and c.endswith("_winner")]
            if win_cols and "map1_name" in rounds_f.columns and len(rounds_f) > 0:
                rsub = rounds_f[
                    ["match_url", "map1_name", "team1_name", "team2_name"] + win_cols
                ].dropna(subset=["map1_name"]).copy()

                def margin_for_focus(row: pd.Series) -> float | None:
                    s = row[win_cols].astype(str).str.lower()
                    t1_wins = (s == "team1").sum()
                    t2_wins = (s == "team2").sum()
                    if row["team1_name"] == focus:
                        return float(t1_wins - t2_wins)
                    if row["team2_name"] == focus:
                        return float(t2_wins - t1_wins)
                    return None

                rsub["margines"] = rsub.apply(margin_for_focus, axis=1)
                rsub = rsub.dropna(subset=["margines"])
                if not rsub.empty:
                    fig_box = px.box(
                        rsub,
                        x="map1_name",
                        y="margines",
                        color_discrete_sequence=[ACCENT],
                        title=f"{focus}: wygrane minus przegrane rundy (mapa 1) wg mapy — dodatnie = przewaga",
                        labels={"map1_name": "mapa", "margines": "przewaga w rundach"},
                    )
                    fig_box.update_xaxes(tickangle=45)
                    fig_box.add_hline(y=0, line_dash="dash", line_color="gray")
                    st.plotly_chart(fig_box, use_container_width=True)
                else:
                    st.info("Brak danych rund mapy 1 dla tej druzyny po filtrach.")
        else:
            st.caption("Wybierz druzyne w panelu, aby zobaczyc jej przewage rund na poszczegolnych mapach.")

    with tab_profile:
        if focus is None:
            st.info("Wybierz **konkretna druzyne** w panelu po lewej, aby zobaczyc jej profil: forma w czasie, LAN/Online, bilans i przeciwnicy.")
        else:
            mf = perspective_focus(m, focus)
            if mf.empty:
                st.warning("Brak meczow tej druzyny po filtrach.")
            else:
                st.markdown(f"#### Profil: {focus}")
                cwr1, cwr2 = st.columns(2)
                cwr1.metric("Winrate (wybrane mecze)", f"{mf['wygrala_wybrana'].mean():.1%}")
                cwr2.metric("Liczba meczow", f"{len(mf):,}")

                wm = pr.winrate_monthly(mf)
                if len(wm) > 3:
                    fig_t = px.line(
                        wm, x="okres", y="winrate", markers=True,
                        color_discrete_sequence=[ACCENT],
                        title=f"Forma {focus} w czasie — winrate na koniec miesiaca",
                    )
                    fig_t.update_yaxes(tickformat=".0%")
                    fig_t.add_hline(y=0.5, line_dash="dash", line_color="gray")
                    st.plotly_chart(fig_t, use_container_width=True)
                else:
                    st.caption("Za malo miesiecy z meczami, aby pokazac wiarygodny trend formy.")

                ca, cb = st.columns(2)
                with ca:
                    evw = pr.winrate_by_event_type(mf)
                    if not evw.empty:
                        fig_e = px.bar(
                            evw, x="event_type", y="winrate", text="mecze",
                            color="event_type", color_discrete_map=EVENT_COLORS,
                            title=f"{focus}: winrate LAN vs Online",
                            labels={"winrate": "winrate", "event_type": "typ eventu"},
                        )
                        fig_e.update_yaxes(tickformat=".0%")
                        st.plotly_chart(fig_e, use_container_width=True)
                with cb:
                    fig_donut = go.Figure(
                        data=[
                            go.Pie(
                                labels=["Wygrana", "Przegrana"],
                                values=[int(mf["wygrala_wybrana"].sum()), int((~mf["wygrala_wybrana"]).sum())],
                                hole=0.4, marker=dict(colors=[GREEN, ACCENT]),
                            )
                        ]
                    )
                    fig_donut.update_layout(title=f"Bilans {focus}")
                    st.plotly_chart(fig_donut, use_container_width=True)

                top_opp = (
                    mf.groupby("przeciwnik", as_index=False)
                    .agg(mecze=("przeciwnik", "count"), wygrane=("wygrala_wybrana", "sum"))
                    .assign(wr=lambda x: x["wygrane"] / x["mecze"])
                    .query("mecze >= 3")
                    .sort_values("mecze", ascending=False)
                    .head(15)
                )
                if not top_opp.empty:
                    fig_opp = px.bar(
                        top_opp, x="przeciwnik", y="wr",
                        color_discrete_sequence=[BLUE],
                        title=f"{focus}: winrate vs najczesciej spotykani przeciwnicy (min. 3 mecze)",
                        labels={"wr": "winrate", "przeciwnik": ""},
                    )
                    fig_opp.update_yaxes(tickformat=".0%")
                    fig_opp.update_xaxes(tickangle=45)
                    fig_opp.add_hline(y=0.5, line_dash="dash", line_color="gray")
                    st.plotly_chart(fig_opp, use_container_width=True)

                with st.expander(f"Mecze druzyny {focus} (tabela)"):
                    tbl = mf[["date_parsed", "przeciwnik", "wygrala_wybrana", "przewaga_rating_wybranej", "tournament"]].copy()
                    tbl["wynik"] = np.where(tbl["wygrala_wybrana"], "Wygrana", "Porazka")
                    tbl = tbl.drop(columns=["wygrala_wybrana"]).sort_values("date_parsed", ascending=False)
                    st.dataframe(tbl, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
