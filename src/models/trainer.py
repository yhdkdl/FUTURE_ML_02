import joblib
import pandas as pd
from pathlib import Path

from scipy.sparse import spmatrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from src.utils.config import MODELS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_logistic_regression() -> LogisticRegression:
    return LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
        C=1.0,
        solver="lbfgs",
    )


def build_random_forest() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        max_depth=None,
        min_samples_split=5,
    )


def build_linear_svc() -> LinearSVC:
    return LinearSVC(
        class_weight="balanced",
        random_state=42,
        max_iter=2000,
        C=1.0,
    )


def train_model(model, X_train: spmatrix, y_train: pd.Series) -> object:
    model_name = type(model).__name__
    logger.info(f"Training {model_name}...")
    model.fit(X_train, y_train)
    logger.info(f"{model_name} training complete")
    return model


def save_model(model, filename: str) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    filepath = MODELS_DIR / filename
    joblib.dump(model, filepath)
    logger.info(f"Model saved to {filepath}")
    return filepath


def load_model(filename: str) -> object:
    filepath = MODELS_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"No model found at {filepath}")
    model = joblib.load(filepath)
    logger.info(f"Model loaded from {filepath}")
    return model