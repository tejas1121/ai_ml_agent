"""Training-partition loading for candidate development and selection."""

from __future__ import annotations

import pandas as pd

from src.data import load_dataset


def load_train_only(
    dataset: str, seed: int
) -> tuple[pd.DataFrame, pd.Series]:
    """Return only training features and labels for one dataset split."""
    X_train, _, y_train, _ = load_dataset(dataset, seed)
    return X_train, y_train
