"""
Data Preprocessing Module for Fake News Detection System
========================================================
Handles data loading, column normalization, missing value handling,
duplicate removal, text cleaning, stop-word removal, and stemming.
"""

import os
import re
import string
import logging
from functools import lru_cache
import pandas as pd
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Standard comprehensive English stopwords list (failsafe offline support + scikit-learn standard)
DEFAULT_STOPWORDS = frozenset([
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't",
    "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm",
    "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more",
    "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
    "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's",
    "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they",
    "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't",
    "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom",
    "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've",
    "your", "yours", "yourself", "yourselves", "mr", "mrs", "said", "also", "would", "one", "two"
])

# Attempt to load NLTK stopwords if available, merge with default
try:
    import nltk
    from nltk.corpus import stopwords
    nltk_stops = set(stopwords.words("english"))
    STOPWORDS = DEFAULT_STOPWORDS.union(nltk_stops)
except Exception:
    STOPWORDS = DEFAULT_STOPWORDS

# Initialize Porter Stemmer with LRU caching for high throughput
try:
    from nltk.stem import PorterStemmer
    _stemmer = PorterStemmer()
    @lru_cache(maxsize=100000)
    def cached_stem(word: str) -> str:
        return _stemmer.stem(word)
except Exception:
    @lru_cache(maxsize=100000)
    def cached_stem(word: str) -> str:
        # Fallback simple suffix trimmer if nltk is absent
        if len(word) > 4 and word.endswith("ing"):
            return word[:-3]
        if len(word) > 3 and word.endswith("ed"):
            return word[:-2]
        if len(word) > 3 and word.endswith("es"):
            return word[:-2]
        if len(word) > 3 and word.endswith("s"):
            return word[:-1]
        return word

# Compiled regular expressions for fast text normalization
RE_HTML = re.compile(r"<[^>]+>")
RE_URL = re.compile(r"https?://\S+|www\.\S+")
RE_NON_ALPHA = re.compile(r"[^a-zA-Z\s]")
RE_WHITESPACE = re.compile(r"\s+")


def clean_text(text: str, apply_stemming: bool = True) -> str:
    """
    Applies the full NLP text cleaning pipeline to a single string:
    1. Lowercasing
    2. HTML tag removal
    3. URL removal
    4. Non-alphabetic character & punctuation removal
    5. Tokenization & whitespace normalization
    6. Stop-word removal
    7. Stemming (Porter Stemmer with LRU cache)

    Parameters
    ----------
    text : str
        Input raw text string.
    apply_stemming : bool, default=True
        Whether to apply stemming to tokens.

    Returns
    -------
    str : Cleaned and preprocessed text string.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Remove HTML tags
    text = RE_HTML.sub(" ", text)

    # 3. Remove URLs
    text = RE_URL.sub(" ", text)

    # 4. Remove punctuation & special characters (keep only a-z and whitespace)
    text = RE_NON_ALPHA.sub(" ", text)

    # 5. Tokenize via whitespace splitting
    tokens = text.split()

    # 6. Stop-word removal & short token filtering (keep tokens >= 2 chars)
    filtered = [w for w in tokens if len(w) >= 2 and w not in STOPWORDS]

    # 7. Stemming
    if apply_stemming:
        processed_tokens = [cached_stem(w) for w in filtered]
    else:
        processed_tokens = filtered

    return " ".join(processed_tokens)


def load_dataset(data_path: str = "data/WELFake_Dataset.csv") -> pd.DataFrame:
    """
    Loads and inspects the WELFake dataset safely, checking column names
    and providing informative error messages if the file is missing.

    Parameters
    ----------
    data_path : str
        Path to WELFake_Dataset.csv

    Returns
    -------
    pd.DataFrame : Loaded raw DataFrame
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found at '{data_path}'.\n"
            f"Please download 'WELFake_Dataset.csv' from Kaggle:\n"
            f"https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification\n"
            f"and place it inside the 'data/' folder."
        )

    logger.info(f"Loading dataset from '{data_path}'...")
    df = pd.read_csv(data_path)
    logger.info(f"Raw dataset loaded. Shape: {df.shape}")
    return df


def preprocess_dataframe(df: pd.DataFrame, sample_size: int = None, apply_stemming: bool = True) -> pd.DataFrame:
    """
    Preprocesses the WELFake dataset according to project specifications:
    - Normalizes column names (case-insensitive detection of title, text, label)
    - Removes index/serial columns (e.g. Unnamed: 0)
    - Handles missing values (fills NaN in title/text, drops NaN in label)
    - Drops duplicate rows
    - Combines Title and Text into 'combined_text'
    - Applies clean_text to 'combined_text'
    - Filters out empty preprocessed records

    Parameters
    ----------
    df : pd.DataFrame
        Input raw dataframe.
    sample_size : int, optional
        If specified, stratifies or samples rows for faster training experimentation.
    apply_stemming : bool, default=True
        Whether to apply stemming during text cleaning.

    Returns
    -------
    pd.DataFrame : Cleaned DataFrame with 'combined_text', 'clean_text', and 'label'.
    """
    df = df.copy()

    # Map column names case-insensitively
    col_mapping = {}
    for col in df.columns:
        c_lower = col.strip().lower()
        if c_lower in ["title", "headline"]:
            col_mapping[col] = "title"
        elif c_lower in ["text", "article", "content"]:
            col_mapping[col] = "text"
        elif c_lower in ["label", "target", "class"]:
            col_mapping[col] = "label"

    df.rename(columns=col_mapping, inplace=True)

    # Validate required columns
    required_cols = ["title", "text", "label"]
    for req in required_cols:
        if req not in df.columns:
            raise KeyError(f"Expected column '{req}' not found in dataset columns: {list(df.columns)}")

    # Drop serial / unnamed index columns
    drop_cols = [c for c in df.columns if "unnamed" in c.lower() or c.lower() in ["id", "index", "serial", "s.no"]]
    if drop_cols:
        logger.info(f"Dropping unnecessary columns: {drop_cols}")
        df.drop(columns=drop_cols, inplace=True, errors="ignore")

    # Drop missing labels
    initial_len = len(df)
    df.dropna(subset=["label"], inplace=True)
    df["label"] = df["label"].astype(int)

    # Align labels to project specification: 0 = Fake News, 1 = Real News
    # In the raw WELFake CSV, 1 corresponds to 37,106 fake news rows and 0 corresponds to 35,028 real news rows.
    # Inverting aligns it with the project standard (0 -> Fake News, 1 -> Real News)
    if (df["label"] == 1).sum() > (df["label"] == 0).sum():
        logger.info("Aligning raw labels to project convention: 0 = Fake News, 1 = Real News")
        df["label"] = 1 - df["label"]

    # Handle missing values in title and text by filling with empty string
    df["title"] = df["title"].fillna("").astype(str)
    df["text"] = df["text"].fillna("").astype(str)

    # Remove duplicates
    df.drop_duplicates(subset=["title", "text"], inplace=True)
    logger.info(f"Removed duplicates and invalid labels: from {initial_len} to {len(df)} records.")

    # Combine Title and Text: combined_text = Title + " " + Text
    logger.info("Combining 'title' and 'text' into 'combined_text'...")
    df["combined_text"] = df["title"] + " " + df["text"]

    # Filter out records where combined_text is just whitespace
    df = df[df["combined_text"].str.strip().str.len() > 10].reset_index(drop=True)

    # Optional sampling for rapid validation if requested
    if sample_size and sample_size < len(df):
        logger.info(f"Sampling {sample_size} records stratified by label...")
        df = df.groupby("label", group_keys=False).apply(
            lambda x: x.sample(min(len(x), sample_size // 2), random_state=42)
        ).reset_index(drop=True)

    # Apply text cleaning
    logger.info("Applying NLP text cleaning pipeline (lowercasing, url/html removal, stop-words, stemming)...")
    df["clean_text"] = df["combined_text"].apply(lambda t: clean_text(t, apply_stemming=apply_stemming))

    # Drop records that became empty after cleaning
    df = df[df["clean_text"].str.strip().str.len() > 5].reset_index(drop=True)
    logger.info(f"Preprocessing completed. Final valid records: {len(df)}")
    logger.info(f"Class distribution:\n{df['label'].value_counts(normalize=True).to_dict()}")

    return df
