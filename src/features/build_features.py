"""Feature engineering utilities for F1 finishing time regression."""

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

CATEGORICAL_VARS = ["team_name", "driver_ref", "track", "year"]
NUMERIC_VARS = ["grid", "pit_stops"]  # extend as needed

def make_preprocessor(df: pd.DataFrame) -> Pipeline:
    """Create a preprocessing pipeline for numeric + categorical features."""

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_VARS),
            ("cat", categorical_transformer, CATEGORICAL_VARS),
        ]
    )

    return preprocessor

def get_feature_target(df: pd.DataFrame):
    """Return X (features) and y (target) from the modeling dataset."""
    y = df["finishing_time_ms"]
    X = df[CATEGORICAL_VARS + NUMERIC_VARS]
    return X, y
