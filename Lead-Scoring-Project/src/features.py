"""Feature engineering: binary mapping, variance filtering, ColumnTransformer."""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BINARY_COLUMNS: list[str] = [
    "Do Not Email",
    "Do Not Call",
    "Search",
    "Magazine",
    "Newspaper Article",
    "X Education Forums",
    "Newspaper",
    "Digital Advertisement",
    "Through Recommendations",
    "Receive More Updates About Our Courses",
    "Update me on Supply Chain Content",
    "Get updates on DM Content",
    "I agree to pay the amount through cheque",
    "A free copy of Mastering The Interview",
]

NUMERIC_COLUMNS: list[str] = [
    "TotalVisits",
    "Total Time Spent on Website",
    "Page Views Per Visit",
]

CATEGORICAL_COLUMNS: list[str] = [
    "Lead Origin",
    "Lead Source",
    "Last Activity",
    "Country",
    "Specialization",
    "How did you hear about X Education",
    "What is your current occupation",
    "City",
]

# If one value dominates a binary column past this share, it's carrying
# basically no signal, so drop it instead of hardcoding a fixed list.
NEAR_ZERO_VARIANCE_THRESHOLD = 0.01


class BinaryMapper(BaseEstimator, TransformerMixin):
    """Maps Yes/No columns to 1/0. Fills a missing column with 'No' first,
    so a partial single-lead input doesn't break the ColumnTransformer
    downstream that expects these columns to exist."""

    def fit(self, X: pd.DataFrame, y=None) -> "BinaryMapper":
        self.columns_ = [c for c in BINARY_COLUMNS if c in X.columns]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        for col in self.columns_:
            if col not in df.columns:
                df[col] = "No"
            df[col] = df[col].map({"Yes": 1, "No": 0}).fillna(0).astype(int)
        return df


def drop_near_zero_variance(
    df: pd.DataFrame, candidate_columns: list[str], threshold: float = NEAR_ZERO_VARIANCE_THRESHOLD
) -> list[str]:
    kept = []
    for col in candidate_columns:
        if col not in df.columns:
            continue
        top_share = df[col].value_counts(normalize=True).iloc[0]
        if top_share <= 1 - threshold:
            kept.append(col)
    return kept


def get_feature_columns(prepared_df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """Resolve final numeric/binary/categorical columns.

    `prepared_df` must already be cleaned + binary-mapped (i.e. the
    output of DataCleaner + BinaryMapper on the training fold).
    """
    numeric_cols = [c for c in NUMERIC_COLUMNS if c in prepared_df.columns]
    binary_cols = drop_near_zero_variance(prepared_df, BINARY_COLUMNS)
    categorical_cols = [c for c in CATEGORICAL_COLUMNS if c in prepared_df.columns]
    return numeric_cols, binary_cols, categorical_cols


def make_preprocessor(numeric_cols: list[str], binary_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_cols),
            ("binary", "passthrough", binary_cols),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )


def build_preprocessor(prepared_df: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str], list[str]]:
    """Decide columns from a prepared training frame and build a fresh
    (unfitted) ColumnTransformer sized to that schema."""
    numeric_cols, binary_cols, categorical_cols = get_feature_columns(prepared_df)
    preprocessor = make_preprocessor(numeric_cols, binary_cols, categorical_cols)
    return preprocessor, numeric_cols, binary_cols, categorical_cols


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    from clean_data import TARGET_COLUMN, DataCleaner

    raw = pd.read_csv("data/Lead_Scoring.csv")
    X = raw.drop(columns=[TARGET_COLUMN])
    y = raw[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    cleaner = DataCleaner().fit(X_train)
    binarizer = BinaryMapper()

    X_train_prepared = binarizer.fit_transform(cleaner.transform(X_train))
    preprocessor, numeric_cols, binary_cols, categorical_cols = build_preprocessor(X_train_prepared)

    X_train_transformed = preprocessor.fit_transform(X_train_prepared)

    print(f"Numeric columns:     {numeric_cols}")
    print(f"Binary columns kept: {binary_cols}")
    print(f"Dropped (near-zero variance): {sorted(set(BINARY_COLUMNS) - set(binary_cols))}")
    print(f"Categorical columns: {categorical_cols}")
    print(f"Transformed matrix shape: {X_train_transformed.shape}")
