"""Train the baseline regression model only."""

from src.config import ensure_directories
from src.models.train_model import train_baseline_model

def main():
    ensure_directories()
    metrics = train_baseline_model()
    print("Baseline model metrics:")
    print(metrics)

if __name__ == "__main__":
    main()
