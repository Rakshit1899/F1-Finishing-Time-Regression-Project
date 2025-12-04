"""Evaluation helpers for regression models."""

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

def regression_metrics(y_true, y_pred):
    return {
        "r2": r2_score(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred, squared=False),
        "mae": mean_squared_error(y_true, y_pred, squared=False),
    }
