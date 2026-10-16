"""Dataset profiles and fixed-order numerical meta-features."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def profile_dataset(X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
    """Summarize feature shape, types, missingness, and binary class balance."""
    numeric = X.select_dtypes(include="number")
    categorical = X.select_dtypes(exclude="number")
    skew = numeric.skew(axis=0).abs().replace([np.inf, -np.inf], np.nan).dropna()
    cardinalities = categorical.nunique(dropna=True)
    class_fractions = y.value_counts(normalize=True)

    return {
        "n_rows": int(X.shape[0]),
        "n_cols": int(X.shape[1]),
        "n_numeric": int(numeric.shape[1]),
        "n_categorical": int(categorical.shape[1]),
        "missing_rate": float(X.isna().to_numpy().mean()) if X.size else 0.0,
        "class_balance": float(class_fractions.min()) if not class_fractions.empty else 0.0,
        "mean_abs_skewness": float(skew.mean()) if not skew.empty else 0.0,
        "mean_categorical_cardinality": (
            float(cardinalities.mean()) if not cardinalities.empty else 0.0
        ),
        "columns": [
            {"name": str(column), "dtype": str(dtype)}
            for column, dtype in X.dtypes.items()
        ],
    }


def meta_features(profile: dict[str, Any]) -> list[float]:
    """Return [rows, columns, numeric, categorical, missing, minority, abs_skew, cat_cardinality]."""
    keys = (
        "n_rows",
        "n_cols",
        "n_numeric",
        "n_categorical",
        "missing_rate",
        "class_balance",
        "mean_abs_skewness",
        "mean_categorical_cardinality",
    )
    return [float(profile[key]) for key in keys]


def profile_to_text(profile: dict[str, Any]) -> str:
    """Format a compact profile suitable for inclusion in a future prompt."""
    columns_text = "; ".join(
        f"{column['name']} ({column['dtype']})" for column in profile["columns"]
    ) or "(no columns)"
    return (
        "Dataset profile\n"
        f"- Rows: {profile['n_rows']}\n"
        f"- Columns: {profile['n_cols']}\n"
        f"- Numeric features: {profile['n_numeric']}\n"
        f"- Categorical features: {profile['n_categorical']}\n"
        f"- Missing-value rate: {profile['missing_rate']:.4f}\n"
        f"- Minority-class fraction: {profile['class_balance']:.4f}\n"
        f"- Mean absolute numeric skewness: {profile['mean_abs_skewness']:.4f}\n"
        "- Mean categorical cardinality: "
        f"{profile['mean_categorical_cardinality']:.4f}\n"
        f"- Column names and dtypes: {columns_text}"
    )
