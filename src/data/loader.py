import pandas as pd
from src.utils.logger import get_logger
from src.utils.config import RAW_DATA_DIR, TEXT_COLUMN, CATEGORY_TARGET, ID_COLUMN

logger = get_logger(__name__)


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("Loading Zenodo IT Support Ticket dataset...")

    X_train = pd.read_csv(RAW_DATA_DIR / "X_train.csv")
    X_test  = pd.read_csv(RAW_DATA_DIR / "X_test.csv")
    y_train = pd.read_csv(RAW_DATA_DIR / "y_train.csv")
    y_test  = pd.read_csv(RAW_DATA_DIR / "y_test.csv")

    train_df = X_train.merge(y_train, on=ID_COLUMN)
    test_df  = X_test.merge(y_test,  on=ID_COLUMN)

    logger.info(f"Train set: {len(train_df)} tickets")
    logger.info(f"Test set:  {len(test_df)} tickets")

    return train_df, test_df


def get_basic_info(df: pd.DataFrame, split_name: str = "dataset") -> None:
    logger.info(f"=== {split_name.upper()} INFO ===")
    logger.info(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    logger.info("Missing values:")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        logger.info("  None")
    else:
        for col, count in missing.items():
            logger.info(f"  {col}: {count}")

    logger.info("Category distribution:")
    for cat, count in df[CATEGORY_TARGET].value_counts().items():
        logger.info(f"  {cat}: {count}")