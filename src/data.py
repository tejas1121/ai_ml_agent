"""OpenML-CC18 dataset discovery, loading, and train/test splitting."""

from __future__ import annotations

import difflib
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import openml
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def _config() -> dict[str, Any]:
    with CONFIG_PATH.open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def _configure_cache() -> Path:
    cache_path = PROJECT_ROOT / _config()["paths"]["cache"]
    cache_path.mkdir(parents=True, exist_ok=True)
    openml.config.set_root_cache_directory(str(cache_path))
    return cache_path


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


@lru_cache(maxsize=1)
def _suite_dataset_index() -> list[tuple[int, str]]:
    """Fetch the suite's dataset names once, caching their metadata locally."""
    _configure_cache()
    suite = openml.study.get_suite(int(_config()["openml_suite_id"]))
    records: list[tuple[int, str]] = []
    for dataset_id in suite.data:
        dataset = openml.datasets.get_dataset(
            int(dataset_id),
            download_data=False,
            download_qualities=False,
            download_features_meta_data=False,
        )
        records.append((int(dataset_id), dataset.name))
    return records


def _resolve_dataset(name: str, records: list[tuple[int, str]]) -> tuple[int, str]:
    normalized = _normalize_name(name)
    for dataset_id, suite_name in records:
        if _normalize_name(suite_name) == normalized:
            return dataset_id, suite_name

    names = [suite_name for _, suite_name in records]
    nearest = difflib.get_close_matches(name, names, n=1, cutoff=0.55)
    if not nearest:
        raise ValueError(f"Dataset {name!r} was not found in OpenML suite 99.")
    suite_name = nearest[0]
    dataset_id = next(dataset_id for dataset_id, item_name in records if item_name == suite_name)
    print(f"Dataset {name!r} not found in suite 99; using closest match {suite_name!r}.")
    return dataset_id, suite_name


def _load_openml_data(name: str) -> tuple[pd.DataFrame, pd.Series]:
    cache_path = PROJECT_ROOT / "data" / "cache" / f"{name}.pkl"
    if cache_path.exists():
        features, target = pd.read_pickle(cache_path)
        return features, target

    records = _suite_dataset_index()
    dataset_id, suite_name = _resolve_dataset(name, records)
    dataset = openml.datasets.get_dataset(
        dataset_id,
        download_data=True,
        download_qualities=False,
        download_features_meta_data=True,
    )
    target_name = dataset.default_target_attribute
    if not target_name:
        raise ValueError(f"OpenML dataset {suite_name!r} has no default target.")
    features, target, _, _ = dataset.get_data(
        target=target_name,
        dataset_format="dataframe",
    )
    if target is None:
        raise ValueError(f"OpenML dataset {suite_name!r} did not return its target.")

    valid_target = target.notna()
    features = features.loc[valid_target].copy()
    target = target.loc[valid_target]
    labels = LabelEncoder().fit_transform(target)
    if np.unique(labels).size != 2:
        raise ValueError(
            f"Expected a binary target for {suite_name!r}; found "
            f"{np.unique(labels).size} classes."
        )
    target_series = pd.Series(
        labels, index=features.index, name=target_name, dtype="int64"
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle((features, target_series), cache_path)
    return features, target_series


def _summary(name: str, features: pd.DataFrame, target: pd.Series) -> str:
    numeric_count = int(features.select_dtypes(include="number").shape[1])
    categorical_count = features.shape[1] - numeric_count
    missing_rate = float(features.isna().to_numpy().mean()) if features.size else 0.0
    balance = target.value_counts(normalize=True).sort_index()
    return (
        f"{name}: rows={len(features)}, columns={features.shape[1]}, "
        f"numeric={numeric_count}, categorical={categorical_count}, "
        f"missing_rate={missing_rate:.4f}, "
        f"class_balance={{0: {balance.get(0, 0.0):.4f}, 1: {balance.get(1, 0.0):.4f}}}"
    )


def print_dataset_summaries() -> None:
    """Download/cache and print the requested dataset dimensions and balance."""
    records = _suite_dataset_index()
    for name in _config()["datasets"]:
        dataset_id, suite_name = _resolve_dataset(name, records)
        del dataset_id
        features, target = _load_openml_data(suite_name)
        print(_summary(name, features, target))
        del features, target


def load_dataset(
    name: str, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load a binary dataset and return a stratified train/test split."""
    features, target = _load_openml_data(name)
    train_size = 1.0 - float(_config()["test_size"])
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        train_size=train_size,
        random_state=seed,
        stratify=target,
    )
    return X_train, X_test, y_train, y_test


def get_cv_splits(
    X_train: pd.DataFrame, y_train: pd.Series, seed: int
) -> StratifiedKFold:
    """Return the training-only shuffled stratified five-fold splitter."""
    return StratifiedKFold(
        n_splits=int(_config()["cv_folds"]),
        shuffle=True,
        random_state=seed,
    )


if __name__ == "__main__":
    print_dataset_summaries()
