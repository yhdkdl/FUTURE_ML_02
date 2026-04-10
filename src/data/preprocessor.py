import re
import string

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from src.utils.config import TEXT_COLUMN
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Initialize once at module level to avoid repeated initialization overhead.
STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def to_lowercase(text: str) -> str:
    return text.lower()


def remove_punctuation(text: str) -> str:
    return text.translate(str.maketrans("", "", string.punctuation))


def remove_numbers(text: str) -> str:
    return re.sub(r"\d+", "", text)

def remove_placeholders(text: str) -> str:
    # Removes unfilled template tags like {product_purchased}, {order_id}
    return re.sub(r"\{[^}]*\}", "", text)


def remove_extra_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def remove_stopwords(text: str) -> str:
    tokens = word_tokenize(text)
    filtered = [word for word in tokens if word not in STOP_WORDS]
    return " ".join(filtered)


def lemmatize_text(text: str) -> str:
    tokens = word_tokenize(text)
    lemmatized = [LEMMATIZER.lemmatize(word) for word in tokens]
    return " ".join(lemmatized)


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)

    text = to_lowercase(text)
    text = remove_placeholders(text)  
    text = remove_punctuation(text)
    text = remove_numbers(text)
    text = remove_extra_whitespace(text)
    text = remove_stopwords(text)
    text = lemmatize_text(text)
    text = remove_extra_whitespace(text)

    return text


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Starting text preprocessing pipeline...")

    df = df.copy()

    logger.info("Dropping rows with missing ticket text...")
    before = len(df)
    df = df.dropna(subset=[TEXT_COLUMN])
    after = len(df)
    if before != after:
        logger.info(f"Dropped {before - after} rows with missing text")

    logger.info("Cleaning text column...")
    df["clean_text"] = df[TEXT_COLUMN].apply(clean_text)

    logger.info("Removing rows where clean text is empty after processing...")
    before = len(df)
    df = df[df["clean_text"].str.strip() != ""]
    after = len(df)
    if before != after:
        logger.info(f"Dropped {before - after} rows with empty clean text")

    logger.info(f"Preprocessing complete. {len(df)} rows remaining.")
    return df