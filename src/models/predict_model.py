"""Helper functions for loading a trained model and making predictions."""

from pathlib import Path
import joblib
import pandas as pd

def load_model(path: Path):
    return joblib.load(path)

def predict(model, X: pd.DataFrame):
    return model.predict(X)
