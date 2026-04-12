import joblib
import pandas as pd
from dataclasses import dataclass
from typing import Optional

from src.data.preprocessor import clean_text
from src.utils.config import MODELS_DIR, PRIORITY_ORDER
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PredictionResult:
    raw_text: str
    cleaned_text: str
    category: str
    priority: str
    priority_level: int
    is_valid: bool
    warning: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "raw_text": self.raw_text,
            "cleaned_text": self.cleaned_text,
            "category": self.category,
            "priority": self.priority,
            "priority_level": self.priority_level,
            "is_valid": self.is_valid,
            "warning": self.warning,
        }

    def __str__(self) -> str:
        warning_str = f"\n  Warning: {self.warning}" if self.warning else ""
        preview = self.raw_text[:80] + ("..." if len(self.raw_text) > 80 else "")
        return (
            f"\n{'=' * 55}\n"
            f"  Ticket Classification Result\n"
            f"{'=' * 55}\n"
            f"  Input:    {preview}\n"
            f"  Category: {self.category}\n"
            f"  Priority: {self.priority} (Level {self.priority_level}/3)\n"
            f"  Valid:    {self.is_valid}"
            f"{warning_str}\n"
            f"{'=' * 55}"
        )


class TicketPredictor:
    # Minimum cleaned text length to make a reliable prediction.
    MIN_TEXT_LENGTH = 10

    def __init__(self):
        self._vectorizer = None
        self._category_model = None
        self._priority_model = None
        self._is_loaded = False

    def load(self) -> "TicketPredictor":
        logger.info("Loading models and vectorizer...")

        vectorizer_path = MODELS_DIR / "tfidf_vectorizer.pkl"
        category_path = MODELS_DIR / "category_classifier.pkl"
        priority_path = MODELS_DIR / "priority_classifier.pkl"

        for path in [vectorizer_path, category_path, priority_path]:
            if not path.exists():
                raise FileNotFoundError(
                    f"Required model file not found: {path}\n"
                    f"Please run Sprint 3-5 notebooks first."
                )

        self._vectorizer = joblib.load(vectorizer_path)
        self._category_model = joblib.load(category_path)
        self._priority_model = joblib.load(priority_path)
        self._is_loaded = True

        logger.info("All models loaded successfully")
        return self

    def _validate_input(self, text: str) -> tuple[bool, Optional[str]]:
        if not text or not text.strip():
            return False, "Input text is empty"

        cleaned = clean_text(text)
        if len(cleaned) < self.MIN_TEXT_LENGTH:
            return False, f"Text too short after cleaning ({len(cleaned)} chars)"

        return True, None

    def predict(self, raw_text: str) -> PredictionResult:
        if not self._is_loaded:
            raise RuntimeError("Models not loaded. Call predictor.load() first.")

        is_valid, warning = self._validate_input(raw_text)
        cleaned = clean_text(raw_text)

        if not is_valid:
            return PredictionResult(
                raw_text=raw_text,
                cleaned_text=cleaned,
                category="Unknown",
                priority="Medium",
                priority_level=1,
                is_valid=False,
                warning=warning,
            )

        features = self._vectorizer.transform([cleaned])

        category = self._category_model.predict(features)[0]
        priority = self._priority_model.predict(features)[0]
        priority_level = PRIORITY_ORDER.get(priority, 1)

        return PredictionResult(
            raw_text=raw_text,
            cleaned_text=cleaned,
            category=category,
            priority=priority,
            priority_level=priority_level,
            is_valid=True,
            warning=None,
        )

    def predict_batch(self, texts: list[str]) -> list[PredictionResult]:
        logger.info(f"Running batch prediction on {len(texts)} tickets...")
        results = [self.predict(text) for text in texts]
        logger.info("Batch prediction complete")
        return results

    def predict_dataframe(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        results = self.predict_batch(df[text_column].astype(str).tolist())

        df = df.copy()
        df["predicted_category"] = [r.category for r in results]
        df["predicted_priority"] = [r.priority for r in results]
        df["priority_level"] = [r.priority_level for r in results]
        df["prediction_valid"] = [r.is_valid for r in results]
        df["prediction_warning"] = [r.warning for r in results]

        return df


_predictor_instance: Optional[TicketPredictor] = None


def get_predictor() -> TicketPredictor:
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = TicketPredictor().load()
    return _predictor_instance


def predict_ticket(raw_text: str) -> PredictionResult:
    predictor = get_predictor()
    return predictor.predict(raw_text)
