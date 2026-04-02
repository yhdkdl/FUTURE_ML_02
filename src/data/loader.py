import pandas as pd
from pathlib import Path
from src.utils.logger import get_logger
from src.utils.config import RAW_DATA_DIR

logger = get_logger(__name__)

def load_raw_data(filename: str = "customer_support_tickets.csv") -> pd.DataFrame:
    filepath = RAW_DATA_DIR / filename
    
    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset not found at {filepath}. "
            f"Please download it and place it in data/raw/"
        )
    
    logger.info(f"Loading dataset from {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    
    return df


def get_basic_info(df: pd.DataFrame) -> None:
    logger.info("=== DATASET SHAPE ===")
    logger.info(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    
    logger.info("=== COLUMN NAMES ===")
    for col in df.columns:
        logger.info(f"  - {col}")

    logger.info("=== MISSING VALUES ===")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        logger.info("  No missing values found")
    else:
        for col, count in missing.items():
            logger.info(f"  {col}: {count} missing")

    logger.info("=== DATA TYPES ===")
    for col, dtype in df.dtypes.items():
        logger.info(f"  {col}: {dtype}")