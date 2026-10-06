"""
Model Training and Orchestration Module
=======================================
Trains the 5 machine learning models, serializes model artifacts,
invokes evaluation routines, selects the optimal best model,
and persists model metadata.
"""

import os
import time
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from src.evaluate_models import (
    evaluate_single_model,
    generate_comparison_report,
    plot_confusion_matrices,
    plot_best_confusion_matrix,
    plot_roc_curves,
    plot_feature_importance,
    select_best_model
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_model_instances() -> dict:
    """
    Constructs instances of the five required machine learning models
    with hyperparameters tuned for sparse TF-IDF text classification.
    """
    models = {
        "Logistic Regression": LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42,
            solver="lbfgs"
        ),
        "Naive Bayes": MultinomialNB(
            alpha=0.1
        ),
        "Support Vector Machine (SVM)": CalibratedClassifierCV(
            estimator=LinearSVC(
                C=1.0,
                max_iter=2000,
                random_state=42,
                dual=False
            ),
            cv=3
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=35,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=50,
            max_depth=3,
            max_features=60,
            subsample=0.8,
            random_state=42
        )
    }
    return models


def train_and_evaluate_all_models(
    X_train_tfidf,
    y_train,
    X_test_tfidf,
    y_test,
    vectorizer,
    models_dir: str = "models",
    reports_dir: str = "reports"
) -> tuple:
    """
    Trains all 5 models, evaluates them on test data, generates comparison reports
    and diagnostic plots, selects the best model, and persists artifacts.

    Parameters
    ----------
    X_train_tfidf : sparse matrix
        TF-IDF features for training.
    y_train : array-like
        Ground truth labels for training (0=Fake, 1=Real).
    X_test_tfidf : sparse matrix
        TF-IDF features for testing.
    y_test : array-like
        Ground truth labels for testing.
    vectorizer : TfidfVectorizer
        Fitted TF-IDF vectorizer.
    models_dir : str
        Directory to save models.
    reports_dir : str
        Directory to save evaluation reports.

    Returns
    -------
    tuple : (best_model, best_model_name, evaluation_results, comparison_df)
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    models = get_model_instances()
    trained_models = {}
    evaluation_results = []

    # Mapping of model names to file names
    filename_map = {
        "Logistic Regression": "logistic_regression.pkl",
        "Naive Bayes": "naive_bayes.pkl",
        "Support Vector Machine (SVM)": "svm.pkl",
        "Random Forest": "random_forest.pkl",
        "Gradient Boosting": "gradient_boosting.pkl"
    }

    print("\n" + "=" * 60)
    print("TRAINING AND EVALUATING MACHINE LEARNING MODELS")
    print("=" * 60)

    for name, model in models.items():
        print(f"\n---> Training: {name}...")
        start_time = time.time()
        model.fit(X_train_tfidf, y_train)
        train_duration = time.time() - start_time
        print(f"     Completed training in {train_duration:.2f} seconds.")

        # Save individual model
        filename = filename_map[name]
        save_path = os.path.join(models_dir, filename)
        joblib.dump(model, save_path, compress=3)
        logger.info(f"Saved {name} to '{save_path}'")
        trained_models[name] = model

        # Evaluate on test set
        metrics = evaluate_single_model(model, X_test_tfidf, y_test, model_name=name)
        evaluation_results.append(metrics)

    # Generate comparison table
    comparison_df = generate_comparison_report(evaluation_results, reports_dir=reports_dir)

    # Plot Confusion Matrices and ROC Curves
    print("\nGenerating diagnostic plots...")
    plot_confusion_matrices(evaluation_results, reports_dir=reports_dir)
    plot_roc_curves(evaluation_results, y_test, reports_dir=reports_dir)

    # Select Best Model based on F1-Score
    best_result = select_best_model(evaluation_results)
    best_model_name = best_result["Model"]
    best_model = trained_models[best_model_name]

    # Save best model confusion matrix
    plot_best_confusion_matrix(best_result, reports_dir=reports_dir)

    # Save feature importance for best model
    try:
        plot_feature_importance(best_model, vectorizer, model_name=best_model_name, reports_dir=reports_dir)
    except Exception as e:
        logger.warning(f"Could not generate feature importance plot: {e}")
        # Fallback to Logistic Regression for feature explainability if best model is non-linear
        if "Logistic Regression" in trained_models:
            plot_feature_importance(
                trained_models["Logistic Regression"],
                vectorizer,
                model_name="Logistic Regression (Interpretability)",
                reports_dir=reports_dir
            )

    # Save Best Model explicitly
    best_model_path = os.path.join(models_dir, "best_model.pkl")
    joblib.dump(best_model, best_model_path, compress=3)
    logger.info(f"Saved best model ({best_model_name}) to '{best_model_path}'")

    # Save Model Metadata
    metadata = {
        "best_model_name": best_model_name,
        "best_model_f1": best_result["F1 Score"],
        "best_model_accuracy": best_result["Accuracy"],
        "best_model_roc_auc": best_result["ROC-AUC"],
        "comparison_table": comparison_df.to_dict(orient="records"),
        "preprocessing_config": {
            "lowercase": True,
            "strip_html": True,
            "strip_urls": True,
            "remove_punctuation": True,
            "stemming": "PorterStemmer",
            "stopword_removal": True
        },
        "tfidf_config": {
            "max_features": vectorizer.max_features,
            "ngram_range": vectorizer.ngram_range,
            "sublinear_tf": vectorizer.sublinear_tf
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    metadata_path = os.path.join(models_dir, "model_metadata.pkl")
    joblib.dump(metadata, metadata_path)
    logger.info(f"Saved model metadata to '{metadata_path}'")

    return best_model, best_model_name, evaluation_results, comparison_df
