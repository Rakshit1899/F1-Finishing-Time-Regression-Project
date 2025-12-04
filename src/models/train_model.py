"""Training routines for baseline and extended regression models."""

from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import Pipeline

from ..config import MODELS_DIR
from ..data.make_dataset import build_modeling_dataset
from ..features.build_features import make_preprocessor, get_feature_target

def train_baseline_model(test_size: float = 0.2, random_state: int = 42):
    df = build_modeling_dataset()
    X, y = get_feature_target(df)
    preprocessor = make_preprocessor(df)

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "baseline_linear_regression.joblib"
    joblib.dump(model, model_path)

    return {
        "model_path": model_path,
        "r2": r2,
        "rmse": rmse,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
