"""
Model Evaluation and Explainability Module
==========================================
Calculates evaluation metrics, confusion matrices, ROC curves,
model comparison table, and feature importance explainability.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Ensure clean plotting style
sns.set_theme(style="whitegrid")


def evaluate_single_model(model, X_test, y_test, model_name: str) -> dict:
    """
    Evaluates a trained classifier on the test dataset.

    Parameters
    ----------
    model : estimator
        Trained scikit-learn model.
    X_test : array-like or sparse matrix
        Test features.
    y_test : array-like
        True test labels (0=Fake, 1=Real).
    model_name : str
        Human-readable model name.

    Returns
    -------
    dict : Dictionary containing predictions, probabilities, and computed metrics.
    """
    y_pred = model.predict(X_test)

    # Obtain probability estimates or decision function scores for ROC-AUC
    y_prob = None
    if hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
        except Exception:
            y_prob = None
    elif hasattr(model, "decision_function"):
        try:
            df_scores = model.decision_function(X_test)
            # Min-max scale or sigmoid for ROC-AUC
            y_prob = 1 / (1 + np.exp(-df_scores))
        except Exception:
            y_prob = None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    roc_auc = None
    if y_prob is not None:
        try:
            roc_auc = roc_auc_score(y_test, y_prob)
        except Exception:
            roc_auc = None

    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "Model": model_name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1,
        "ROC-AUC": roc_auc if roc_auc is not None else float("nan"),
        "y_pred": y_pred,
        "y_prob": y_prob,
        "confusion_matrix": cm
    }

    auc_str = f"{roc_auc:.4f}" if (roc_auc is not None and not np.isnan(roc_auc)) else "N/A"
    logger.info(
        f"[{model_name}] Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | "
        f"F1: {f1:.4f} | ROC-AUC: {auc_str}"
    )

    return metrics


def generate_comparison_report(evaluation_results: list, reports_dir: str = "reports") -> pd.DataFrame:
    """
    Creates a comparison DataFrame from model evaluation metrics and saves to CSV.

    Parameters
    ----------
    evaluation_results : list of dict
        Results returned by evaluate_single_model.
    reports_dir : str
        Directory to store the comparison CSV.

    Returns
    -------
    pd.DataFrame : Clean comparison table sorted by F1 Score.
    """
    os.makedirs(reports_dir, exist_ok=True)
    rows = []
    for res in evaluation_results:
        rows.append({
            "Model": res["Model"],
            "Accuracy": round(res["Accuracy"], 4),
            "Precision": round(res["Precision"], 4),
            "Recall": round(res["Recall"], 4),
            "F1 Score": round(res["F1 Score"], 4),
            "ROC-AUC": round(res["ROC-AUC"], 4) if not np.isnan(res["ROC-AUC"]) else "N/A"
        })

    df_comp = pd.DataFrame(rows)
    df_comp.sort_values(by="F1 Score", ascending=False, inplace=True)
    df_comp.reset_index(drop=True, inplace=True)

    csv_path = os.path.join(reports_dir, "model_comparison.csv")
    df_comp.to_csv(csv_path, index=False)
    logger.info(f"Model comparison table saved to '{csv_path}'")
    return df_comp


def plot_confusion_matrices(evaluation_results: list, reports_dir: str = "reports"):
    """
    Plots a grid of confusion matrices for all evaluated models.

    Parameters
    ----------
    evaluation_results : list of dict
        Results from evaluate_single_model.
    reports_dir : str
        Directory to save confusion_matrix.png.
    """
    os.makedirs(reports_dir, exist_ok=True)
    n_models = len(evaluation_results)
    cols = 3
    rows = (n_models + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(16, 5 * rows))
    axes = np.array(axes).flatten()

    for idx, res in enumerate(evaluation_results):
        ax = axes[idx]
        cm = res["confusion_matrix"]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["Fake (0)", "Real (1)"],
            yticklabels=["Fake (0)", "Real (1)"],
            annot_kws={"size": 13, "weight": "bold"}
        )
        ax.set_title(f"{res['Model']}\nF1: {res['F1 Score']:.4f} | Acc: {res['Accuracy']:.4f}", fontsize=13, weight="bold")
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("True Label", fontsize=11)

    # Hide extra unused subplots
    for idx in range(n_models, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    out_path = os.path.join(reports_dir, "confusion_matrix.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info(f"Confusion matrices plot saved to '{out_path}'")


def plot_best_confusion_matrix(best_result: dict, reports_dir: str = "reports"):
    """
    Plots an annotated confusion matrix specifically for the selected best model.
    """
    os.makedirs(reports_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    cm = best_result["confusion_matrix"]

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        ax=ax,
        xticklabels=["Fake (0)", "Real (1)"],
        yticklabels=["Fake (0)", "Real (1)"],
        annot_kws={"size": 15, "weight": "bold"}
    )

    tn, fp, fn, tp = cm.ravel()
    subtitle = f"TN: {tn:,} | FP: {fp:,} | FN: {fn:,} | TP: {tp:,}\nAccuracy: {best_result['Accuracy']*100:.2f}% | F1 Score: {best_result['F1 Score']*100:.2f}%"
    ax.set_title(f"Best Model Confusion Matrix: {best_result['Model']}\n{subtitle}", fontsize=13, weight="bold", pad=15)
    ax.set_xlabel("Predicted Class", fontsize=12)
    ax.set_ylabel("Actual Class", fontsize=12)

    plt.tight_layout()
    out_path = os.path.join(reports_dir, "confusion_matrix_best.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info(f"Best model confusion matrix saved to '{out_path}'")


def plot_roc_curves(evaluation_results: list, y_test, reports_dir: str = "reports"):
    """
    Generates and saves a combined ROC curve comparison plot for all models with probability/decision scores.
    """
    os.makedirs(reports_dir, exist_ok=True)
    plt.figure(figsize=(9, 7))

    for res in evaluation_results:
        y_prob = res.get("y_prob")
        if y_prob is not None and not np.isnan(res["ROC-AUC"]):
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            plt.plot(
                fpr, tpr,
                lw=2,
                label=f"{res['Model']} (AUC = {res['ROC-AUC']:.4f})"
            )

    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Classifier (AUC = 0.5000)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=12)
    plt.title("Receiver Operating Characteristic (ROC) Comparison", fontsize=14, weight="bold")
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    out_path = os.path.join(reports_dir, "roc_curve.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info(f"ROC curve plot saved to '{out_path}'")


def plot_feature_importance(best_model, vectorizer, model_name: str, top_n: int = 20, reports_dir: str = "reports"):
    """
    Extracts and visualizes the most influential TF-IDF features.
    Provides clear academic interpretability:
    - Features associated with Fake News (Class 0)
    - Features associated with Real News (Class 1)
    """
    os.makedirs(reports_dir, exist_ok=True)
    feature_names = np.array(vectorizer.get_feature_names_out())

    # Handle linear models (Logistic Regression, LinearSVC)
    coef = None
    if hasattr(best_model, "coef_"):
        coef = best_model.coef_[0]
    elif hasattr(best_model, "estimator") and hasattr(best_model.estimator, "coef_"):
        coef = best_model.estimator.coef_[0]
    elif hasattr(best_model, "calibrated_classifiers_"):
        # CalibratedClassifierCV
        coefs = [clf.estimator.coef_[0] for clf in best_model.calibrated_classifiers_ if hasattr(clf.estimator, "coef_")]
        if coefs:
            coef = np.mean(coefs, axis=0)

    if coef is not None:
        # Top positive coefficients -> Real News (1)
        # Top negative coefficients -> Fake News (0)
        top_real_idx = np.argsort(coef)[-top_n:]
        top_fake_idx = np.argsort(coef)[:top_n]

        top_real_words = feature_names[top_real_idx]
        top_real_weights = coef[top_real_idx]

        top_fake_words = feature_names[top_fake_idx]
        top_fake_weights = np.abs(coef[top_fake_idx])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        y_pos1 = np.arange(top_n)
        ax1.barh(y_pos1, top_fake_weights, color="#d9534f", align="center")
        ax1.set_yticks(y_pos1)
        ax1.set_yticklabels(top_fake_words, fontsize=11)
        ax1.invert_yaxis()
        ax1.set_xlabel("Relative Predictive Weight (|Negative Coefficient|)", fontsize=11)
        ax1.set_title(f"Top {top_n} Tokens Indicative of FAKE News", fontsize=13, weight="bold", color="#a94442")

        y_pos2 = np.arange(top_n)
        ax2.barh(y_pos2, top_real_weights, color="#5cb85c", align="center")
        ax2.set_yticks(y_pos2)
        ax2.set_yticklabels(top_real_words, fontsize=11)
        ax2.invert_yaxis()
        ax2.set_xlabel("Relative Predictive Weight (Positive Coefficient)", fontsize=11)
        ax2.set_title(f"Top {top_n} Tokens Indicative of REAL News", fontsize=13, weight="bold", color="#3c763d")

        fig.suptitle(
            f"Feature Explainability Analysis ({model_name})\n"
            "(These are linguistic tokens that influenced the model's classification patterns)",
            fontsize=14, weight="bold"
        )
        plt.tight_layout()
        out_path = os.path.join(reports_dir, "feature_importance.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        logger.info(f"Feature importance visualization saved to '{out_path}'")
        return

    # Handle tree-based models (Random Forest, Gradient Boosting)
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
        top_idx = np.argsort(importances)[-top_n:]
        top_words = feature_names[top_idx]
        top_scores = importances[top_idx]

        fig, ax = plt.subplots(figsize=(10, 8))
        y_pos = np.arange(top_n)
        ax.barh(y_pos, top_scores, color="#337ab7", align="center")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_words, fontsize=11)
        ax.invert_yaxis()
        ax.set_xlabel("Feature Importance (Gini / Split Gain)", fontsize=11)
        ax.set_title(
            f"Top {top_n} Most Influential Tokens ({model_name})\n"
            "(Tokens that contributed most to tree decision splits)",
            fontsize=13, weight="bold"
        )
        plt.tight_layout()
        out_path = os.path.join(reports_dir, "feature_importance.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        logger.info(f"Tree feature importance visualization saved to '{out_path}'")


def select_best_model(evaluation_results: list) -> dict:
    """
    Selects the best performing model strictly based on:
    1. Primary: F1-score
    2. Secondary: Accuracy
    3. Tertiary: ROC-AUC

    Parameters
    ----------
    evaluation_results : list of dict
        Metrics from evaluate_single_model.

    Returns
    -------
    dict : Evaluation result of the best model.
    """
    def sorting_key(item):
        f1 = item["F1 Score"]
        acc = item["Accuracy"]
        auc = item["ROC-AUC"] if not np.isnan(item["ROC-AUC"]) else 0.0
        return (f1, acc, auc)

    sorted_results = sorted(evaluation_results, key=sorting_key, reverse=True)
    best = sorted_results[0]
    logger.info(
        f"Selected Best Model: {best['Model']} with F1 Score={best['F1 Score']:.4f}, "
        f"Accuracy={best['Accuracy']:.4f}, ROC-AUC={best['ROC-AUC']:.4f}"
    )
    return best
