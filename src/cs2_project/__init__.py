"""CS2 HLTV analysis helpers for notebooks and scripts."""

from .loaders import (
    CORE_TABLE_KEYS,
    DATA_FILES,
    extract_hltv_match_page_id,
    load_all_tables,
    load_core_tables,
    load_tables,
    merge_matches_with_rounds,
    merge_matches_with_timeseries,
    normalize_team_pair_key,
    project_root,
)
from .prediction import (
    FEATURE_BASES,
    FEATURE_LABELS,
    MODEL_PATH,
    build_team_profiles,
    build_training_frame,
    explain_prediction,
    load_or_train,
    predict_matchup,
    train_model,
)

__all__ = [
    "CORE_TABLE_KEYS",
    "DATA_FILES",
    "extract_hltv_match_page_id",
    "load_all_tables",
    "load_core_tables",
    "load_tables",
    "merge_matches_with_rounds",
    "merge_matches_with_timeseries",
    "normalize_team_pair_key",
    "project_root",
    "FEATURE_BASES",
    "FEATURE_LABELS",
    "MODEL_PATH",
    "build_team_profiles",
    "build_training_frame",
    "explain_prediction",
    "load_or_train",
    "predict_matchup",
    "train_model",
]
