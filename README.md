# Predicting Formula 1 Driver Finishing Times Using Regression

This project analyzes Formula 1 race data (Ergast F1 database) to model and predict
driver finishing times using regression techniques.

## Structure

- `data/` – Raw and processed datasets
- `notebooks/` – Exploratory analysis, feature engineering, modeling
- `src/` – Reusable Python modules (data loading, feature building, modeling, viz)
- `results/` – Saved models, figures, evaluation tables
- `reports/` – Written report, slide outlines, figures for the report
- `scripts/` – CLI-style scripts to run data prep and training pipelines

## Quick Start

1. Create and activate a virtual environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download/prepare data:

   ```bash
   python scripts/download_data.py
   python scripts/run_all.py
   ```

4. Use the notebooks in `notebooks/` for EDA, modeling, and diagnostics.
