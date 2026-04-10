import re
import string

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from src.utils.config import TEXT_COLUMN
from src.utils.logger import get_logger

logger = get_logger(__name__)

BOILERPLATE_PHRASES = [
    r"i(?:'m| am) having an issue with the",
    r"i'm facing a problem with my",
    r"please assist",
    r"please help",
    r"i've tried troubleshooting steps mentioned in the user manual",
    r"but the issue persists",
    r"your billing zip code is",
    r"we appreciate that you have requested",
    r"please double check your email address",
    r"the issue persists",
    r"the issue remains unresolved",
    r"i've already contacted customer support multiple times",
    r"i've performed a factory reset",
    r"i've recently updated the firmware",
    r"it was working fine until yesterday",
    r"sometimes it works fine but other times",
    r"i've checked for any available software updates",
]

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

def remove_boilerplate(text: str) -> str:
    for phrase in BOILERPLATE_PHRASES:
        text = re.sub(phrase, " ", text, flags=re.IGNORECASE)
    return text



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
    text = remove_boilerplate(text)   
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