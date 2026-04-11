import pandas as pd

from src.utils.config import (
    CATEGORY_TARGET,
    CATEGORY_TO_PRIORITY,
    PRIORITY_ORDER,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

PRIORITY_COLUMN = "priority"


def assign_priority_labels(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Assigning rule-based priority labels...")

    df = df.copy()
    df[PRIORITY_COLUMN] = df[CATEGORY_TARGET].map(CATEGORY_TO_PRIORITY)

    unmapped = df[PRIORITY_COLUMN].isnull().sum()
    if unmapped > 0:
        logger.warning(f"{unmapped} tickets could not be mapped - defaulting to Medium")
        df[PRIORITY_COLUMN] = df[PRIORITY_COLUMN].fillna("Medium")

    logger.info("Priority distribution:")
    for priority, count in df[PRIORITY_COLUMN].value_counts().items():
        logger.info(f"  {priority}: {count}")

    return df


def get_priority_order() -> dict:
    return PRIORITY_ORDER


def validate_priority_labels(df: pd.DataFrame) -> bool:
    valid_priorities = set(PRIORITY_ORDER.keys())
    actual_priorities = set(df[PRIORITY_COLUMN].unique())
    invalid = actual_priorities - valid_priorities

    if invalid:
        logger.error(f"Invalid priority labels found: {invalid}")
        return False

    logger.info("All priority labels are valid")
    return True
