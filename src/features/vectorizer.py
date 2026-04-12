from pathlib import Path

import joblib
import pandas as pd
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer

from src.utils.config import MODELS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

# These values are deliberate defaults for ticket text vectorization.
TFIDF_PARAMS = {
    "max_features": 5000,
    "ngram_range": (1, 2),
    "min_df": 3,
    "max_df": 0.70,
    "sublinear_tf": True,
}


def build_vectorizer() -> TfidfVectorizer:
    logger.info("Building TF-IDF vectorizer with params:")
    for key, val in TFIDF_PARAMS.items():
        logger.info(f"  {key}: {val}")
    return TfidfVectorizer(**TFIDF_PARAMS)


def fit_vectorizer(
    vectorizer: TfidfVectorizer,
    texts: pd.Series,
) -> TfidfVectorizer:
    logger.info(f"Fitting vectorizer on {len(texts)} documents...")
    vectorizer.fit(texts)
    logger.info(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    return vectorizer


def transform_texts(
    vectorizer: TfidfVectorizer,
    texts: pd.Series,
) -> spmatrix:
    logger.info(f"Transforming {len(texts)} documents...")
    matrix = vectorizer.transform(texts)
    logger.info(f"Feature matrix shape: {matrix.shape}")
    return matrix


def fit_transform_texts(
    vectorizer: TfidfVectorizer,
    texts: pd.Series,
) -> spmatrix:
    logger.info(f"Fit-transforming {len(texts)} documents...")
    matrix = vectorizer.fit_transform(texts)
    logger.info(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    logger.info(f"Feature matrix shape: {matrix.shape}")
    return matrix


def save_vectorizer(vectorizer: TfidfVectorizer, filename: str) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    filepath = MODELS_DIR / filename
    joblib.dump(vectorizer, filepath)
    logger.info(f"Vectorizer saved to {filepath}")
    return filepath


def load_vectorizer(filename: str) -> TfidfVectorizer:
    filepath = MODELS_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"No vectorizer found at {filepath}")
    vectorizer = joblib.load(filepath)
    logger.info(f"Vectorizer loaded from {filepath}")
    return vectorizer


def get_top_features_per_class(
    vectorizer: TfidfVectorizer,
    X: spmatrix,
    y: pd.Series,
    n: int = 15,
) -> dict:
    feature_names = vectorizer.get_feature_names_out()
    top_features = {}

    for label in y.unique():
        class_mask = y == label
        class_matrix = X[class_mask.values]

        mean_scores = class_matrix.mean(axis=0).A1
        top_indices = mean_scores.argsort()[::-1][:n]
        top_words = [feature_names[i] for i in top_indices]
        top_features[label] = top_words

    return top_features