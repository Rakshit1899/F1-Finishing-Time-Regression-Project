"""Script to download Ergast F1 CSVs or remind the user to place them in data/raw.

For simplicity, this is a placeholder. You can either:
- Use `requests` to hit the Ergast API and save responses; or
- Download a Kaggle/CSV dump manually and place files in `data/raw/`.
"""

from pathlib import Path
from src.config import RAW_DATA_DIR

def main():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("Place your Ergast CSV files (races.csv, results.csv, drivers.csv, constructors.csv, ...) in:")
    print(RAW_DATA_DIR.resolve())

if __name__ == "__main__":
    main()
