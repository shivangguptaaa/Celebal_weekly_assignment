"""Cleaning and leakage-removal, as a scikit-learn transformer.

Fitting learns imputation values and rare-category lists from whatever
data it's given (the training fold only). Transform reapplies those
exact values everywhere else - the held-out test set, a single lead
form, a batch upload - so nothing downstream ever recomputes a median
or a rare-category list from a partial or 1-row frame.

Being a proper transformer means it can sit as the first step inside
the same Pipeline as the model, so train.py and scoring.py share one
fitted object with no separate cleaning step to keep in sync.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

LEAKAGE_COLUMNS: list[str] = [
    "Tags",
    "Lead Quality",
    "Lead Profile",
    "Asymmetrique Activity Index",
    "Asymmetrique Profile Index",
    "Asymmetrique Activity Score",
    "Asymmetrique Profile Score",
    "Last Notable Activity",
    "Prospect ID",
    "Lead Number",
]

# Near-constant, one dominant category, basically no signal.
LOW_INFORMATION_COLUMNS: list[str] = [
    "What matters most to you in choosing a course",
]

TARGET_COLUMN = "Converted"
SELECT_PLACEHOLDER = "Select"

CATEGORICAL_UNKNOWN_THRESHOLD = 0.01
RARE_CATEGORY_THRESHOLD = 0.01

HIGH_CARDINALITY_COLUMNS: list[str] = [
    "Lead Source",
    "Country",
    "Specialization",
    "How did you hear about X Education",
    "What is your current occupation",
    "City",
]


def drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = [c for c in LEAKAGE_COLUMNS + LOW_INFORMATION_COLUMNS if c in df.columns]
    return df.drop(columns=cols_to_drop)


def replace_select_placeholder(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace(SELECT_PLACEHOLDER, pd.NA)


class DataCleaner(BaseEstimator, TransformerMixin):
    """Drops leakage columns, imputes missing values, collapses rare
    categories - using statistics learned once from whatever it's fit on."""

    def __init__(
        self,
        unknown_threshold: float = CATEGORICAL_UNKNOWN_THRESHOLD,
        rare_threshold: float = RARE_CATEGORY_THRESHOLD,
    ):
        self.unknown_threshold = unknown_threshold
        self.rare_threshold = rare_threshold

    def fit(self, X: pd.DataFrame, y=None) -> "DataCleaner":
        df = drop_leakage_columns(X)
        df = replace_select_placeholder(df)

        self.numeric_medians_: dict[str, float] = {}
        self.categorical_fill_: dict[str, dict] = {}

        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                self.numeric_medians_[col] = float(df[col].median())
            else:
                missing_frac = df[col].isna().mean()
                if missing_frac > self.unknown_threshold:
                    self.categorical_fill_[col] = {"strategy": "unknown"}
                else:
                    mode = df[col].mode(dropna=True)
                    fill_val = mode.iloc[0] if not mode.empty else "Unknown"
                    self.categorical_fill_[col] = {"strategy": "mode", "value": fill_val}

        # rare categories are computed AFTER missing-value fill, so "Unknown"
        # is treated as a normal category when checking frequency
        filled = self._apply_missing(df)
        self.rare_category_map_: dict[str, list[str]] = {}
        for col in HIGH_CARDINALITY_COLUMNS:
            if col not in filled.columns:
                continue
            freqs = filled[col].value_counts(normalize=True)
            self.rare_category_map_[col] = freqs[freqs < self.rare_threshold].index.tolist()

        return self

    def _apply_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col, median_val in self.numeric_medians_.items():
            if col in df.columns:
                df[col] = df[col].fillna(median_val)
        for col, rule in self.categorical_fill_.items():
            if col not in df.columns:
                continue
            fill_val = "Unknown" if rule["strategy"] == "unknown" else rule["value"]
            df[col] = df[col].fillna(fill_val)
        return df

    def _apply_rare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col, rare_values in self.rare_category_map_.items():
            if col not in df.columns or not rare_values:
                continue
            df[col] = df[col].where(~df[col].isin(rare_values), "Other")
        return df

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        # make sure every column this cleaner was fit on exists, even for a
        # partial single-lead input, so imputation has something to fill
        for col in list(self.numeric_medians_) + list(self.categorical_fill_):
            if col not in df.columns:
                df[col] = np.nan

        df = drop_leakage_columns(df)
        df = replace_select_placeholder(df)
        df = self._apply_missing(df)
        df = self._apply_rare(df)
        return df


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    raw = pd.read_csv("data/Lead_Scoring.csv")
    X = raw.drop(columns=[TARGET_COLUMN])
    y = raw[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    cleaner = DataCleaner().fit(X_train)
    cleaned_train = cleaner.transform(X_train)
    cleaned_test = cleaner.transform(X_test)

    leaked = [c for c in LEAKAGE_COLUMNS if c in cleaned_train.columns]
    assert not leaked, f"Leakage columns survived cleaning: {leaked}"

    for name, frame in [("train", cleaned_train), ("test", cleaned_test)]:
        nulls = frame.isna().sum()
        nulls = nulls[nulls > 0]
        assert nulls.empty, f"Nulls remain in {name} after cleaning:\n{nulls}"

    print(f"Raw shape:           {raw.shape}")
    print(f"Cleaned train shape: {cleaned_train.shape}")
    print(f"Cleaned test shape:  {cleaned_test.shape}")
    print(f"Columns dropped: {sorted(set(X.columns) - set(cleaned_train.columns))}")
