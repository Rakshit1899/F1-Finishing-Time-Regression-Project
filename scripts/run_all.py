"""Run the end-to-end pipeline: build dataset and train baseline model."""

from src.config import ensure_directories
from src.data.make_dataset import build_modeling_dataset
from src.models.train_model import train_baseline_model

def main():
    ensure_directories()
    print("Building modeling dataset...")
    df = build_modeling_dataset()
    print(f"Dataset built with {len(df)} rows.")

    print("Training baseline model...")
    metrics = train_baseline_model()
    print("Training complete.")
    print(metrics)

if __name__ == "__main__":
    main()
