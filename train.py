"""
Fake News Detection System - End-to-End Training Pipeline
=========================================================
Executes data ingestion, text preprocessing, TF-IDF feature extraction,
train-test split, training of five ML classifiers, evaluation, diagnostic
visualizations, best model selection, and model persistence.

Usage:
------
    python train.py
"""

import os
import sys
import time
import argparse
import logging
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_preprocessing import load_dataset, preprocess_dataframe
from src.feature_engineering import build_tfidf_vectorizer, fit_and_transform_tfidf, save_vectorizer
from src.train_models import train_and_evaluate_all_models

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def run_pipeline(
    data_path: str = "data/WELFake_Dataset.csv",
    sample_size: int = None,
    max_features: int = 50000,
    test_size: float = 0.20,
    random_state: int = 42
):
    """
    Executes the entire end-to-end Machine Learning pipeline.
    """
    print("\n" + "=" * 70)
    print("   FAKE-NEWS DETECTION SYSTEM USING MACHINE LEARNING")
    print("           END-TO-END TRAINING PIPELINE")
    print("=" * 70)

    start_total_time = time.time()

    # Step 1 & 2: Dataset Loading and Validation
    print(f"\n[Step 1/7] Loading dataset from: {data_path}")
    if not os.path.exists(data_path):
        print(f"\n[ERROR] Dataset file not found at '{data_path}'!")
        print("Please place 'WELFake_Dataset.csv' inside the 'data/' directory.")
        print("Dataset source: https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification")
        sys.exit(1)

    df_raw = load_dataset(data_path)
    total_raw_records = len(df_raw)
    print(f"Dataset loaded successfully.")
    print(f"Total raw records: {total_raw_records}")
    print(f"Dataset columns: {list(df_raw.columns)}")

    # Step 3 & 4: Data Cleaning and Text Preprocessing
    print(f"\n[Step 2/7] Preprocessing dataset (cleaning, lowercasing, stop-words, stemming)...")
    clean_cache_path = os.path.join("data", "preprocessed_cache.parquet")

    # Use cached preprocessed dataset if available to save time on repeated runs
    if os.path.exists(clean_cache_path):
        print(f"Found cached preprocessed dataset at '{clean_cache_path}'. Loading...")
        df_clean = pd.read_parquet(clean_cache_path)
        print(f"Loaded {len(df_clean)} preprocessed records from cache.")
    else:
        df_clean = preprocess_dataframe(df_raw, sample_size=sample_size, apply_stemming=True)
        try:
            df_clean.to_parquet(clean_cache_path, index=False)
            logger.info(f"Cached preprocessed dataset to '{clean_cache_path}'")
        except Exception as e:
            logger.warning(f"Could not write cache file: {e}")

    total_clean = len(df_clean)
    fake_count = int((df_clean["label"] == 0).sum())
    real_count = int((df_clean["label"] == 1).sum())

    print(f"\nData Preprocessing Summary:")
    print(f"- Valid records after cleaning : {total_clean:,}")
    print(f"- Fake News (Class 0)          : {fake_count:,} ({fake_count/total_clean*100:.1f}%)")
    print(f"- Real News (Class 1)          : {real_count:,} ({real_count/total_clean*100:.1f}%)")

    # Step 5: Train / Test Split (Stratified 80/20)
    print(f"\n[Step 3/7] Splitting data into Training ({(1-test_size)*100:.0f}%) and Testing ({test_size*100:.0f}%)...")
    X = df_clean["clean_text"].values
    y = df_clean["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    print(f"Training samples : {len(X_train):,}")
    print(f"Testing samples  : {len(X_test):,}")

    # Step 6: TF-IDF Feature Extraction
    print(f"\n[Step 4/7] Extracting TF-IDF features (max_features={max_features}, ngram_range=(1,2))...")
    vectorizer = build_tfidf_vectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )
    X_train_tfidf, X_test_tfidf, vectorizer = fit_and_transform_tfidf(vectorizer, X_train, X_test)

    # Save TF-IDF Vectorizer
    vectorizer_path = "models/tfidf_vectorizer.pkl"
    save_vectorizer(vectorizer, vectorizer_path)
    print(f"TF-IDF vectorizer saved to '{vectorizer_path}'")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_):,} features")

    # Step 7: Train and Evaluate 5 Machine Learning Models
    print(f"\n[Step 5/7] Training and evaluating 5 Machine Learning models...")
    best_model, best_name, evaluation_results, comp_df = train_and_evaluate_all_models(
        X_train_tfidf=X_train_tfidf,
        y_train=y_train,
        X_test_tfidf=X_test_tfidf,
        y_test=y_test,
        vectorizer=vectorizer,
        models_dir="models",
        reports_dir="reports"
    )

    # Save additional training metadata
    dataset_summary = {
        "total_articles": total_raw_records,
        "clean_articles": total_clean,
        "fake_count": fake_count,
        "real_count": real_count,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "vocabulary_size": len(vectorizer.vocabulary_)
    }
    joblib.dump(dataset_summary, "models/dataset_summary.pkl")

    # Step 8: Final Summary Output
    total_elapsed = time.time() - start_total_time
    print("\n" + "=" * 70)
    print("                    FINAL MODEL COMPARISON")
    print("=" * 70)
    print(comp_df.to_string(index=False))

    best_f1 = comp_df.loc[comp_df["Model"] == best_name, "F1 Score"].values[0]
    best_acc = comp_df.loc[comp_df["Model"] == best_name, "Accuracy"].values[0]

    print("\n" + "=" * 70)
    print(f"Best Model Selected : {best_name}")
    print(f"Best Model F1 Score : {best_f1 * 100:.2f}%")
    print(f"Best Model Accuracy : {best_acc * 100:.2f}%")
    print(f"Primary Criterion   : F1-score")
    print("=" * 70)
    print("Artifacts generated and saved:")
    print(" - models/logistic_regression.pkl")
    print(" - models/naive_bayes.pkl")
    print(" - models/svm.pkl")
    print(" - models/random_forest.pkl")
    print(" - models/gradient_boosting.pkl")
    print(" - models/best_model.pkl")
    print(" - models/tfidf_vectorizer.pkl")
    print(" - models/model_metadata.pkl")
    print(" - reports/model_comparison.csv")
    print(" - reports/confusion_matrix.png")
    print(" - reports/confusion_matrix_best.png")
    print(" - reports/roc_curve.png")
    print(" - reports/feature_importance.png")
    print(f"\nPipeline successfully completed in {total_elapsed / 60:.2f} minutes.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Fake News Detection System")
    parser.add_argument("--data_path", type=str, default="data/WELFake_Dataset.csv", help="Path to WELFake CSV")
    parser.add_argument("--sample_size", type=int, default=None, help="Sample size for rapid testing (None = full)")
    parser.add_argument("--max_features", type=int, default=40000, help="Max TF-IDF features")
    args = parser.parse_args()

    run_pipeline(
        data_path=args.data_path,
        sample_size=args.sample_size,
        max_features=args.max_features
    )
