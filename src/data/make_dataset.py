"""Scripts for building the modeling dataset from Ergast F1 tables."""

import pandas as pd
from .load_data import load_raw_csv, save_processed

def build_modeling_dataset() -> pd.DataFrame:
    """Merge Ergast tables into a race-driver level dataset.

    Expected raw tables (as CSVs in `data/raw`):
    - `races.csv`
    - `results.csv`
    - `drivers.csv`
    - `constructors.csv`
    - `pit_stops.csv` (optional, for pit-based features)
    - `circuits.csv` (optional, for track-level attributes)
    """
    races = load_raw_csv("races")
    results = load_raw_csv("results")
    drivers = load_raw_csv("drivers")
    constructors = load_raw_csv("constructors")

    # Basic merges
    df = results.merge(races, on="raceId", how="left", suffixes=("", "_race"))
    df = df.merge(drivers, on="driverId", how="left", suffixes=("", "_driver"))
    df = df.merge(constructors, on="constructorId", how="left", suffixes=("", "_constructor"))

    # Example: construct a numeric finishing-time-like response
    # If you have `milliseconds` in results, use that as response.
    if "milliseconds" in df.columns:
        df["finishing_time_ms"] = df["milliseconds"]
    else:
        # Placeholder if column not available – user should adapt.
        df["finishing_time_ms"] = df["positionOrder"].astype("float")  # crude proxy

    # Example basic predictors
    df["year"] = df["year"]
    df["grid"] = df["grid"]
    df["track"] = df["name_race"] if "name_race" in df.columns else df.get("name")
    df["team_name"] = df["name_constructor"]
    df["driver_ref"] = df["driverRef"]

    # Drop obvious missing response
    df = df.dropna(subset=["finishing_time_ms"])

    save_processed(df, "f1_modeling_dataset")
    return df
