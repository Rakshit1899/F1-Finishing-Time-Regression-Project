"""Visualization utilities for EDA and model diagnostics."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_finishing_time_hist(df: pd.DataFrame, col: str = "finishing_time_ms"):
    plt.figure()
    sns.histplot(df[col], bins=30, kde=True)
    plt.title("Distribution of Finishing Times")
    plt.xlabel(col)
    plt.ylabel("Count")

def plot_residuals(y_true, y_pred):
    residuals = y_true - y_pred
    plt.figure()
    sns.scatterplot(x=y_pred, y=residuals)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Predicted")
