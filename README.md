# cs2_data_visualization

Projekt analityczny dla danych profesjonalnych meczów CS2 z HLTV. Repo zawiera:

- loader danych CSV,
- funkcje agregujące statystyki meczów, drużyn, graczy i map,
- model predykcji wyniku meczu drużyna A vs drużyna B,
- dashboard Streamlit z wykresami Plotly,
- notebook eksploracyjny.

Źródło danych: zbiory Kaggle użytkownika [griffindesroches](https://www.kaggle.com/griffindesroches/datasets).

## Struktura projektu

```text
app/dashboard.py                         # aplikacja Streamlit
src/cs2_project/loaders.py               # ścieżki do danych, wczytywanie CSV, joiny
src/cs2_project/matches_utils.py         # normalizacja i perspektywa drużyny
src/cs2_project/presentation.py          # agregacje pod wykresy dashboardu
src/cs2_project/prediction.py            # trening, predykcja i wyjaśnienia modelu
notebooks/cs2_hltv_analysis.ipynb        # analiza eksploracyjna
docs/prezentacja_zarzad.md               # opis biznesowy wyników
data/raw/                                # lokalne pliki CSV z Kaggle
models/match_predictor.joblib            # artefakt modelu tworzony przy uruchomieniu
```

## Dane wejściowe

Dashboard i notebook używają trzech plików CSV:

| Klucz w kodzie | Plik | Rola |
| --- | --- | --- |
| `matches_modeling` | `cs2_newestcombinedmatches_team1_reference_reduced2.csv` | główny zbiór meczowy z nazwami drużyn, wynikiem, statystykami składów i cechami modelu |
| `rounds` | `combined_round_by_round_with_map_names_cleaned.csv` | dane runda po rundzie oraz nazwy map |
| `timeseries` | `newest_ts_ds.csv` | dane czasowe używane w analizach łączenia tabel |

Opcjonalny plik:

| Klucz w kodzie | Plik | Rola |
| --- | --- | --- |
| `matches_full` | `cs2_newestcombinedmatches.csv` | pełny zbiór meczowy; nie jest wymagany przez dashboard |

Mapowanie nazw plików jest zdefiniowane w `DATA_FILES` w `src/cs2_project/loaders.py`.

## Pobranie danych

Pliki powinny znajdować się bezpośrednio w `data/raw/`, bez dodatkowych podfolderów:

```bash
mkdir -p data/raw
cd data/raw
kaggle datasets download -d griffindesroches/cs2-hltv-professional-match-statistics-dataset --unzip
kaggle datasets download -d griffindesroches/cs2-professional-round-by-round-statistics-dataset --unzip
kaggle datasets download -d griffindesroches/cs2-professional-hltv-match-data-time-series --unzip
```

Po rozpakowaniu minimalny zestaw powinien zawierać:

```text
data/raw/cs2_newestcombinedmatches_team1_reference_reduced2.csv
data/raw/combined_round_by_round_with_map_names_cleaned.csv
data/raw/newest_ts_ds.csv
```

## Środowisko

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Na Windows:

```bat
.venv\Scripts\activate
pip install -r requirements.txt
```

## Uruchomienie

Dashboard:

```bash
streamlit run app/dashboard.py
```

Domyślny adres lokalny:

```text
http://localhost:8501
```

Notebook:

```bash
jupyter notebook notebooks/cs2_hltv_analysis.ipynb
```

Trening modelu z CLI:

```bash
python -m cs2_project.prediction
```

Jeżeli pakiet nie jest widoczny z terminala, uruchom z ustawionym `PYTHONPATH`:

```bash
PYTHONPATH=src python -m cs2_project.prediction
```

## Jak działa wczytywanie danych

`load_core_tables()` wczytuje trzy tabele wymagane przez aplikację:

```python
from cs2_project.loaders import load_core_tables

tables = load_core_tables()
matches = tables["matches_modeling"]
rounds = tables["rounds"]
timeseries = tables["timeseries"]
```

`load_all_tables()` próbuje wczytać wszystkie pliki z `DATA_FILES`, także `matches_full`.

Łączenie tabel:

- `merge_matches_with_rounds()` wyciąga identyfikator meczu HLTV z URL (`/matches/<id>`) i łączy mecze z tabelą rund po `match_page_id`.
- `merge_matches_with_timeseries()` normalizuje datę do dnia UTC, buduje klucz pary drużyn niezależny od kolejności (`team_pair_key`) i łączy po `date_norm + team_pair_key`.

## Jak czytać główną tabelę meczową

Najważniejsze kolumny w `matches_modeling`:

| Kolumna | Znaczenie |
| --- | --- |
| `date` | data meczu; w kodzie parsowana do `date_parsed` jako UTC |
| `team1_name`, `team2_name` | nazwy drużyn w kolejności z pliku |
| `team1_win_flag` | `1`, jeśli wygrała `team1`; `0`, jeśli wygrała `team2` |
| `score_team1`, `score_team2` | wynik zapisany dla obu stron; część danych zawiera wyniki map, część wyniki pojedynczych map |
| `team1_avg_RATING`, `team2_avg_RATING` | średni rating składu |
| `rating_diff` | różnica ratingów `team1 - team2` |
| `adr_diff` | różnica ADR `team1 - team2` |
| `event_type` | typ eventu, normalizowany w dashboardzie do `LAN` / `ONLINE` |
| `hltv_url` | URL meczu używany do wyciągania `match_page_id` |

Funkcja `prepare_matches()` dodaje `date_parsed` i konwertuje wybrane kolumny liczbowe przez `pd.to_numeric(..., errors="coerce")`.

Funkcja `add_real_names()` dodaje:

- `zwyciezca`,
- `pokonany`,
- `wynik_slownie`.

Funkcja `perspective_focus(df, focus)` przepisuje mecze na perspektywę jednej drużyny:

- `przeciwnik`,
- `wygrala_wybrana`,
- `przewaga_rating_wybranej`,
- `przewaga_adr_wybranej`.

## Dane rundowe

Tabela `rounds` zawiera kolumny:

- `match_url`,
- `team1_name`, `team2_name`,
- `map1_name`, `map2_name`, `map3_name`,
- kolumny rund w formacie podobnym do `map1_round<N>_winner`.

Wartości zwycięzcy rundy są zapisywane jako `team1` albo `team2`, czyli odnoszą się do stron z danego wiersza, a nie do nazw drużyn jako tekstu. Dashboard wykorzystuje to m.in. do liczenia przewagi rund wybranej drużyny na mapie 1.

## Model predykcji

Kod modelu znajduje się w `src/cs2_project/prediction.py`.

Cel modelu: oszacowanie prawdopodobieństwa zwycięstwa `team_a` nad `team_b`.

Target treningowy:

```text
team1_win_flag
```

Cechy wejściowe są liczone jako różnice:

```text
diff_<cecha> = team1_<cecha> - team2_<cecha>
```

Lista bazowych cech jest w `FEATURE_BASES`:

- średni rating, ADR, KAST, KPR, DPR,
- odchylenie ratingu składu,
- najlepszy i najsłabszy gracz,
- winrate per mapa,
- forma `past3`.

W treningu porównywane są trzy modele:

- regresja logistyczna ze standaryzacją,
- Random Forest,
- Gradient Boosting.

Split testowy jest czasowy: ostatnie `test_size` obserwacji trafia do testu. Zbiór treningowy jest augmentowany symetrycznie przez dodanie `(-X, 1-y)`, żeby ograniczyć bias pozycji `team1`.

Artefakt modelu zawiera m.in.:

- wytrenowany model,
- listę cech,
- profile drużyn,
- metryki najlepszego modelu,
- metryki wszystkich kandydatów,
- baseline oparty na wyższym ratingu,
- globalną ważność cech,
- skalę cech do lokalnych wyjaśnień.

Plik modelu jest zapisywany pod:

```text
models/match_predictor.joblib
```

`load_or_train()` wczytuje ten plik, jeśli istnieje. Gdy pliku nie ma albo `retrain=True`, model jest trenowany od nowa.

## Predykcja i wyjaśnienia

Przykład:

```python
from cs2_project.loaders import load_core_tables
from cs2_project.prediction import load_or_train, predict_matchup, explain_prediction

matches = load_core_tables()["matches_modeling"]
artifact = load_or_train(matches)

result = predict_matchup(artifact, "Vitality", "Natus Vincere")
explanation = explain_prediction(artifact, "Vitality", "Natus Vincere")
```

`predict_matchup()` liczy wynik dla układu A vs B oraz B vs A i uśrednia prawdopodobieństwa, żeby zachować symetrię wyniku.

`explain_prediction()` zwraca lokalne wkłady cech:

- wartość dodatnia działa na korzyść drużyny A,
- wartość ujemna działa na korzyść drużyny B.

Dla modeli liniowych używany jest wkład z wag regresji. Dla modeli drzewiastych używany jest SHAP, jeśli jest dostępny. W przeciwnym razie funkcja korzysta z fallbacku opartego na standaryzowanej różnicy i globalnej ważności cech.

## Dashboard

Dashboard ładuje dane przez `load_core_tables()`, przygotowuje mecze przez `prepare_matches()` i filtruje je po:

- zakresie dat,
- typie eventu,
- wybranej drużynie,
- progach minimalnej liczby meczów lub wystąpień.

Zakładki:

| Zakładka | Zawartość |
| --- | --- |
| `Drużyny (inwestycja)` | winrate drużyn, siła składu vs winrate, LAN vs Online, aktywność, pary H2H |
| `Predykcyjność` | skuteczność faworyta wg ratingu, niespodzianki, rozkład 2-0 vs 2-1 |
| `Predykcja meczu` | wybór dwóch drużyn, prawdopodobieństwa, radar profilu, wkłady cech, metryki modelu |
| `Gracze` | najlepsi gracze po średnim ratingu i liczbie wystąpień |
| `Mapy` | najczęściej grane mapy i przewaga rund wybranej drużyny |
| `Drużyna · profil` | miesięczny winrate, LAN/Online, bilans i przeciwnicy wybranej drużyny |

## Ograniczenia danych i modelu

- Część kolumn wynikowych ma mieszane znaczenie: wyniki serii BO3 i wyniki pojedynczych map są zapisane w tych samych polach. Funkcja `map_score_distribution()` filtruje tylko wyniki `<= 2` i odrzuca forfeity `1-0`.
- Profile drużyn w modelu są budowane z najnowszych dostępnych obserwacji w danych. To daje praktyczny profil siły, ale nie jest pełną symulacją historyczną "as-of-match".
- Model nie używa kolumn wynikowych ani jawnie przeciekających zmiennych z listy `LEAKAGE_FORBIDDEN`.
- Predykcja powinna być czytana jako wsparcie analityczne, a nie pewny typ meczu.

## Typowy przepływ pracy

1. Pobierz CSV do `data/raw/`.
2. Utwórz środowisko i zainstaluj zależności.
3. Uruchom dashboard przez `streamlit run app/dashboard.py`.
4. Przy pierwszym wejściu w predykcję model zapisze artefakt w `models/`.
5. Po zmianie danych usuń `models/match_predictor.joblib` albo wywołaj `load_or_train(..., retrain=True)`.
