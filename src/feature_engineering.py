"""
Feature Engineering Module for Fake News Detection System
=========================================================
Implements TF-IDF feature extraction, parameter configuration,
vectorizer serialization, and inference transformation.
"""

import os
import logging
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def build_tfidf_vectorizer(
    max_features: int = 50000,
    ngram_range: tuple = (1, 2),
    min_df: int = 2,
    max_df: float = 0.95,
    sublinear_tf: bool = True
) -> TfidfVectorizer:
    """
    Initializes and configures the TF-IDF Vectorizer with optimal NLP parameters.

    Parameters
    ----------
    max_features : int, default=50000
        Maximum vocabulary size to retain top TF-IDF unigrams and bigrams.
    ngram_range : tuple, default=(1, 2)
        Lower and upper boundary of the range of n-values for different n-grams.
    min_df : int or float, default=2
        Ignore terms with document frequency strictly lower than given threshold.
    max_df : float, default=0.95
        Ignore terms with document frequency strictly higher than 95% of documents.
    sublinear_tf : bool, default=True
        Apply sublinear tf scaling (1 + log(tf)) to prevent dominant term frequencies.

    Returns
    -------
    TfidfVectorizer : Configured TF-IDF vectorizer instance.
    """
    logger.info(
        f"Initializing TfidfVectorizer with max_features={max_features}, "
        f"ngram_range={ngram_range}, min_df={min_df}, max_df={max_df}, sublinear_tf={sublinear_tf}"
    )
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf
    )
    return vectorizer


def fit_and_transform_tfidf(vectorizer: TfidfVectorizer, X_train, X_test=None):
    """
    Fits TF-IDF vectorizer strictly on training data to prevent data leakage,
    and transforms both training and test sets.

    Parameters
    ----------
    vectorizer : TfidfVectorizer
        Configured TF-IDF vectorizer.
    X_train : iterable of str
        Training text documents.
    X_test : iterable of str, optional
        Testing text documents.

    Returns
    -------
    tuple : (X_train_tfidf, X_test_tfidf [if X_test provided else None], fitted_vectorizer)
    """
    logger.info(f"Fitting TfidfVectorizer on {len(X_train)} training samples...")
    X_train_tfidf = vectorizer.fit_transform(X_train)
    logger.info(f"Training TF-IDF matrix shape: {X_train_tfidf.shape}")

    X_test_tfidf = None
    if X_test is not None:
        logger.info(f"Transforming {len(X_test)} test samples using fitted vectorizer...")
        X_test_tfidf = vectorizer.transform(X_test)
        logger.info(f"Test TF-IDF matrix shape: {X_test_tfidf.shape}")

    return X_train_tfidf, X_test_tfidf, vectorizer


def save_vectorizer(vectorizer: TfidfVectorizer, filepath: str = "models/tfidf_vectorizer.pkl"):
    """
    Persists the fitted TfidfVectorizer to disk using joblib.

    Parameters
    ----------
    vectorizer : TfidfVectorizer
        Fitted TF-IDF vectorizer.
    filepath : str
        Target destination path for serialized file.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    logger.info(f"Saving TfidfVectorizer to '{filepath}'...")
    joblib.dump(vectorizer, filepath, compress=3)
    logger.info("TfidfVectorizer saved successfully.")


def load_vectorizer(filepath: str = "models/tfidf_vectorizer.pkl") -> TfidfVectorizer:
    """
    Loads a serialized TfidfVectorizer from disk.

    Parameters
    ----------
    filepath : str
        Path to serialized vectorizer file.

    Returns
    -------
    TfidfVectorizer : Loaded vectorizer instance.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"TF-IDF vectorizer file not found at '{filepath}'. "
            f"Please run the training pipeline (python train.py) to train and save the vectorizer."
        )
    return joblib.load(filepath)
