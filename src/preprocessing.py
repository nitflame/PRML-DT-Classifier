"""Dataset validation, preprocessing, and reproducible train/test splitting."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if df.columns.duplicated().any():
        raise ValueError("Column names must be unique after trimming whitespace.")
    return df.replace(r"^\s*$", np.nan, regex=True)


def preprocess(frame: pd.DataFrame, target: str, numeric_imputation: str = "median",
               categorical_imputation: str = "mode", categorical_constant: str = "Unknown",
               encoding: str = "Label Encoding"):
    df = clean_frame(frame)
    if target not in df or df[target].isna().all():
        raise ValueError("Choose a target column containing at least one non-missing value.")
    df = df.loc[df[target].notna()].copy()
    y = df.pop(target).astype(str)
    X = df.copy()
    categorical = list(X.select_dtypes(exclude=np.number).columns)
    numeric = [c for c in X.columns if c not in categorical]
    for col in numeric:
        if X[col].isna().any():
            val = X[col].mean() if numeric_imputation == "mean" else X[col].median()
            X[col] = X[col].fillna(val)
    for col in categorical:
        if X[col].isna().any():
            mode = X[col].mode(dropna=True)
            val = categorical_constant if categorical_imputation == "constant" or mode.empty else mode.iloc[0]
            X[col] = X[col].fillna(val).astype(str)
        else:
            X[col] = X[col].astype(str)
    if encoding == "One-Hot Encoding":
        X = pd.get_dummies(X, columns=categorical, dtype=float)
        feature_kinds = {c: "numeric" for c in X.columns}
    else:
        # Keep nominal fields nominal for the multiway ID3 splits; C4.5 can
        # still treat numeric source columns as continuous thresholds.
        X = X.copy()
        for c in categorical:
            values = sorted(X[c].astype(str).unique())
            X[c] = pd.Categorical(X[c].astype(str), categories=values).codes
        feature_kinds = {c: ("categorical" if c in categorical else "numeric") for c in X.columns}
    return X.reset_index(drop=True), y.reset_index(drop=True), feature_kinds


def split_data(X, y, test_size=0.2, random_seed=42):
    counts = y.value_counts()
    stratify = y if len(counts) > 1 and counts.min() >= 2 else None
    return train_test_split(X, y, test_size=test_size, random_state=random_seed, stratify=stratify)
