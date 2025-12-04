"""Utilities for loading Ergast F1 data into pandas DataFrames."""

from pathlib import Path
import pandas as pd
from ..config import RAW_DATA_DIR, PROCESSED_DATA_DIR

def load_raw_csv(name: str) -> pd.DataFrame:
    """Load a raw CSV from data/raw by stem name (without extension)."""
    path = RAW_DATA_DIR / f"{name}.csv"
    return pd.read_csv(path)

def load_processed(name: str) -> pd.DataFrame:
    """Load a processed CSV from data/processed by stem name."""
    path = PROCESSED_DATA_DIR / f"{name}.csv"
    return pd.read_csv(path)

def save_processed(df: pd.DataFrame, name: str) -> Path:
    """Save processed DataFrame to data/processed and return the path."""
    path = PROCESSED_DATA_DIR / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
