"""Model predykcji meczu CS2 na roznicach profili druzyn."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import joblib
except ImportError:  # pragma: no cover
    from sklearn.externals import joblib  # type: ignore

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "match_predictor.joblib"

FEATURE_BASES: list[str] = [
    "avg_RATING",
    "avg_ADR",
    "avg_KAST",
    "avg_KPR",
    "avg_DPR",
    "rating_std",
    "top_player",
    "weakest_player",
    "mirage",
    "inferno",
    "nuke",
    "dust2",
    "overpass",
    "train",
    "ancient",
    "vertigo",
    "anubis",
    "past3",
]

LOWER_IS_BETTER: frozenset[str] = frozenset({"avg_DPR", "rating_std", "weakest_player"})

FEATURE_LABELS: dict[str, str] = {
    "avg_RATING": "Rating skladu",
    "avg_ADR": "ADR (obrazenia/runde)",
    "avg_KAST": "KAST %",
    "avg_KPR": "Zabojstwa/runde (KPR)",
    "avg_DPR": "Smierci/runde (DPR)",
    "rating_std": "Spojnosc skladu",
    "top_player": "Gwiazda (najlepszy gracz)",
    "weakest_player": "Najslabszy gracz",
    "mirage": "Winrate Mirage",
    "inferno": "Winrate Inferno",
    "nuke": "Winrate Nuke",
    "dust2": "Winrate Dust2",
    "overpass": "Winrate Overpass",
    "train": "Winrate Train",
    "ancient": "Winrate Ancient",
    "vertigo": "Winrate Vertigo",
    "anubis": "Winrate Anubis",
    "past3": "Forma (ostatnie mecze)",
}

TARGET = "team1_win_flag"

LEAKAGE_FORBIDDEN: frozenset[str] = frozenset(
    {
        "winner",
        "score_team1",
        "score_team2",
        "decider_map",
        "original_winner_side",
        "team1_totalwinrate",
        "team2_totalwinrate",
        "team1_totallossrate",
        "team2_totallossrate",
        "team1_online_winrate",
        "team2_online_winrate",
        "team1_lan_winrate",
        "team2_lan_winrate",
        "team1_overall_winrate",
        "team2_overall_winrate",
        "team1_wins",
        "team2_wins",
        "team1_losses",
        "team2_losses",
        "team1_head2head_percentage",
        "team2_head2head_percentage",
    }
)


def build_team_profiles(matches: pd.DataFrame) -> pd.DataFrame:
    """Zwraca najnowszy profil kazdej druzyny dla cech z FEATURE_BASES."""
    df = matches.copy()
    df["_date"] = pd.to_datetime(df.get("date"), utc=True, errors="coerce")

    frames = []
    for side in ("team1", "team2"):
        cols = {f"{side}_{b}": b for b in FEATURE_BASES}
        present = {k: v for k, v in cols.items() if k in df.columns}
        if not present:
            continue
        sub = df[[f"{side}_name", "_date", *present.keys()]].rename(
            columns={f"{side}_name": "team", **present}
        )
        frames.append(sub)

    if not frames:
        return pd.DataFrame(columns=["team", *FEATURE_BASES]).set_index("team")

    long = pd.concat(frames, ignore_index=True)
    long = long.dropna(subset=["team"])
    for b in FEATURE_BASES:
        if b in long.columns:
            long[b] = pd.to_numeric(long[b], errors="coerce")

    long = long.sort_values("_date")
    prof = long.groupby("team").tail(1).set_index("team")
    prof = prof[[b for b in FEATURE_BASES if b in prof.columns]]
    prof = prof.fillna(prof.median(numeric_only=True))
    return prof.sort_index()


def _diff_columns(features: list[str]) -> list[str]:
    return [f"diff_{b}" for b in features]


def diff_vector(profile_a: pd.Series, profile_b: pd.Series, features: list[str]) -> pd.DataFrame:
    """Wektor cech (A - B) jako 1-wierszowy DataFrame z nazwami diff_<base>."""
    data = {f"diff_{b}": [float(profile_a[b]) - float(profile_b[b])] for b in features}
    return pd.DataFrame(data)


def build_training_frame(matches: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Buduje X, y i daty na podstawie roznic team1_<base> - team2_<base>."""
    df = matches.copy()
    feats = [b for b in FEATURE_BASES if f"team1_{b}" in df.columns and f"team2_{b}" in df.columns]

    X = pd.DataFrame(index=df.index)
    for b in feats:
        t1 = pd.to_numeric(df[f"team1_{b}"], errors="coerce")
        t2 = pd.to_numeric(df[f"team2_{b}"], errors="coerce")
        X[f"diff_{b}"] = t1 - t2

    y = pd.to_numeric(df[TARGET], errors="coerce")
    dates = pd.to_datetime(df.get("date"), utc=True, errors="coerce")

    mask = X.notna().all(axis=1) & y.notna()
    X, y, dates = X.loc[mask], y.loc[mask].astype(int), dates.loc[mask]

    bad = [c for c in X.columns if c.replace("diff_", "") in LEAKAGE_FORBIDDEN or c in LEAKAGE_FORBIDDEN]
    if bad:
        raise ValueError(f"Cechy z listy leakage trafily do X: {bad}")

    return X.reset_index(drop=True), y.reset_index(drop=True), dates.reset_index(drop=True)


def make_symmetric(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Dodaje lustrzany rekord (-X, 1-y) dla kazdego meczu."""
    X_mirror = -X
    y_mirror = 1 - y
    X_aug = pd.concat([X, X_mirror], ignore_index=True)
    y_aug = pd.concat([y, y_mirror], ignore_index=True)
    return X_aug, y_aug


def _eval_metrics(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }


def _baseline_higher_rating(X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any] | None:
    """Baseline: typuj druzyne z wyzszym srednim ratingiem skladu (diff_avg_RATING > 0)."""
    if "diff_avg_RATING" not in X_test.columns:
        return None
    pred = (X_test["diff_avg_RATING"] > 0).astype(int)
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
    }


def _feature_importance(model: Any, model_name: str, features: list[str]) -> dict[str, float]:
    """Globalna waznosc cech: feature_importances_ (drzewa) lub |coef| (LogReg)."""
    diff_cols = _diff_columns(features)
    if hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_, dtype=float)
    elif isinstance(model, Pipeline) and hasattr(model[-1], "coef_"):
        imp = np.abs(np.ravel(model[-1].coef_))
    elif hasattr(model, "coef_"):
        imp = np.abs(np.ravel(model.coef_))
    else:
        imp = np.ones(len(diff_cols), dtype=float)
    if imp.sum() > 0:
        imp = imp / imp.sum()
    return {b: float(v) for b, v in zip(features, imp)}


def train_model(matches: pd.DataFrame, test_size: float = 0.2) -> dict[str, Any]:
    """Trenuje kandydatow, porownuje test czasowy i zwraca artefakt najlepszego."""
    X, y, dates = build_training_frame(matches)
    features = [c.replace("diff_", "") for c in X.columns]

    order = dates.sort_values().index
    X, y = X.loc[order].reset_index(drop=True), y.loc[order].reset_index(drop=True)

    n_test = max(1, int(len(X) * test_size))
    split = len(X) - n_test
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    X_train_sym, y_train_sym = make_symmetric(X_train, y_train)

    candidates: dict[str, Any] = {
        "Regresja logistyczna": Pipeline(
            [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=5, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }

    all_metrics: dict[str, dict[str, Any]] = {}
    fitted: dict[str, Any] = {}
    for name, est in candidates.items():
        est.fit(X_train_sym, y_train_sym)
        fitted[name] = est
        metrics = _eval_metrics(est, X_test, y_test)
        cv = cross_val_score(
            est, X_train_sym, y_train_sym, cv=5, scoring="accuracy", n_jobs=-1
        )
        metrics["cv_accuracy_mean"] = float(cv.mean())
        metrics["cv_accuracy_std"] = float(cv.std())
        all_metrics[name] = metrics

    best_name = max(
        all_metrics,
        key=lambda n: (all_metrics[n]["roc_auc"], all_metrics[n]["accuracy"]),
    )
    best_model = fitted[best_name]

    baseline = _baseline_higher_rating(X_test, y_test)
    importance = _feature_importance(best_model, best_name, features)
    scale = X_train_sym.std(ddof=0).replace(0, 1.0)
    feature_scale = {b: float(scale[f"diff_{b}"]) for b in features}

    return {
        "model": best_model,
        "model_name": best_name,
        "features": features,
        "diff_columns": _diff_columns(features),
        "profiles": build_team_profiles(matches),
        "metrics": all_metrics[best_name],
        "all_models_metrics": all_metrics,
        "baseline": baseline,
        "importance": importance,
        "feature_scale": feature_scale,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "class_balance": float(y.mean()),
        "trained_at": pd.Timestamp.utcnow().isoformat(),
    }


def load_or_train(
    matches: pd.DataFrame, path: Path | str = MODEL_PATH, retrain: bool = False
) -> dict[str, Any]:
    """Wczytuje model z dysku albo trenuje i zapisuje, jesli pliku brak (lub retrain)."""
    path = Path(path)
    if path.exists() and not retrain:
        try:
            return joblib.load(path)
        except Exception:  # pragma: no cover
            pass
    artifact = train_model(matches)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)
    return artifact


def predict_matchup(artifact: dict[str, Any], team_a: str, team_b: str) -> dict[str, Any]:
    """Prawdopodobienstwo wygranej A i B. Symetryzowane: srednia z (A,B) i odwrotnosci (B,A)."""
    profiles: pd.DataFrame = artifact["profiles"]
    model = artifact["model"]
    features: list[str] = artifact["features"]

    if team_a not in profiles.index or team_b not in profiles.index:
        missing = [t for t in (team_a, team_b) if t not in profiles.index]
        raise KeyError(f"Brak profilu dla: {missing}")

    pa, pb = profiles.loc[team_a], profiles.loc[team_b]
    x_ab = diff_vector(pa, pb, features)
    x_ba = diff_vector(pb, pa, features)

    p_ab = float(model.predict_proba(x_ab)[0, 1])
    p_ba = float(model.predict_proba(x_ba)[0, 1])
    proba_a = (p_ab + (1.0 - p_ba)) / 2.0
    proba_b = 1.0 - proba_a

    return {
        "team_a": team_a,
        "team_b": team_b,
        "proba_a": proba_a,
        "proba_b": proba_b,
        "winner": team_a if proba_a >= proba_b else team_b,
        "diff_vector": x_ab.iloc[0].to_dict(),
    }


def _linear_estimator(model: Any) -> Any | None:
    """Zwraca koncowy estymator liniowy (z coef_) lub None — obsluguje Pipeline."""
    est = model[-1] if isinstance(model, Pipeline) else model
    return est if hasattr(est, "coef_") else None


def _linear_contributions(model: Any, x_ab: pd.DataFrame, features: list[str]) -> dict[str, float] | None:
    """Wklad cech do logitu dla modelu liniowego."""
    lin = _linear_estimator(model)
    if lin is None:
        return None
    coef = np.ravel(lin.coef_).astype(float)
    x = x_ab.to_numpy(dtype=float).ravel()
    if isinstance(model, Pipeline) and hasattr(model[0], "mean_"):
        mean = np.asarray(model[0].mean_, dtype=float)
        scale = np.asarray(model[0].scale_, dtype=float)
        scale = np.where(scale == 0, 1.0, scale)
        contrib = coef * (x - mean) / scale
    else:
        contrib = coef * x
    return {b: float(v) for b, v in zip(features, contrib[: len(features)])}


def explain_prediction(
    artifact: dict[str, Any], team_a: str, team_b: str, top_n: int = 8
) -> dict[str, Any]:
    """Zwraca lokalne wklady cech dla pary A vs B."""
    profiles: pd.DataFrame = artifact["profiles"]
    model = artifact["model"]
    features: list[str] = artifact["features"]
    importance: dict[str, float] = artifact["importance"]
    scale: dict[str, float] = artifact.get("feature_scale", {})

    pa, pb = profiles.loc[team_a], profiles.loc[team_b]
    x_ab = diff_vector(pa, pb, features)

    contributions = _linear_contributions(model, x_ab, features)
    method = "dokladny wklad liniowy (coef × standaryzowana roznica)"

    if contributions is None:
        try:
            import shap  # type: ignore

            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(x_ab)
            arr = sv[1] if isinstance(sv, list) else sv
            vals = np.ravel(np.asarray(arr))
            contributions = {b: float(v) for b, v in zip(features, vals[: len(features)])}
            method = "SHAP (TreeExplainer)"
        except Exception:
            row = x_ab.iloc[0]
            contributions = {
                b: (float(row[f"diff_{b}"]) / scale.get(b, 1.0)) * importance.get(b, 0.0)
                for b in features
            }
            method = "standaryzowana roznica × waznosc"

    signed = {b: (-v if b in LOWER_IS_BETTER else v) for b, v in contributions.items()}
    ranked = sorted(signed.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_n]

    return {
        "method": method,
        "contributions": dict(ranked),
        "global_importance": importance,
    }


def _format_metrics(name: str, m: dict[str, Any]) -> str:
    return (
        f"  {name:22s} acc={m['accuracy']:.3f} prec={m['precision']:.3f} "
        f"rec={m['recall']:.3f} f1={m['f1']:.3f} auc={m['roc_auc']:.3f} "
        f"cv={m['cv_accuracy_mean']:.3f}±{m['cv_accuracy_std']:.3f}"
    )


def main() -> None:  # pragma: no cover
    import sys

    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from cs2_project.loaders import load_tables

    matches = load_tables(["matches_modeling"])["matches_modeling"]
    art = train_model(matches)

    print(f"Najlepszy model: {art['model_name']}")
    print(f"Trening: {art['n_train']} meczow | test: {art['n_test']} meczow")
    print("Metryki na tescie:")
    for name, m in art["all_models_metrics"].items():
        mark = " <-- wybrany" if name == art["model_name"] else ""
        print(_format_metrics(name, m) + mark)
    if art["baseline"]:
        b = art["baseline"]
        print(f"  {'Baseline (wyzszy rating)':22s} acc={b['accuracy']:.3f} f1={b['f1']:.3f}")
    cm = art["metrics"]["confusion_matrix"]
    print(f"Macierz pomylek wybranego (wiersze=prawda 0/1, kolumny=pred 0/1): {cm}")
    print("Top 8 cech (waznosc):")
    for b, v in sorted(art["importance"].items(), key=lambda kv: kv[1], reverse=True)[:8]:
        print(f"  {FEATURE_LABELS.get(b, b):28s} {v:.3f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(art, MODEL_PATH)
    print(f"Zapisano model -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
