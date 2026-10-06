"""
Prediction and Inference Pipeline Module
========================================
Handles end-to-end inference for user-provided headlines and articles:
input validation, text cleaning, TF-IDF vectorization, model scoring,
confidence computation, and linguistic explainability.
"""

import os
import logging
import numpy as np
import joblib

from src.data_preprocessing import clean_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DISCLAIMER_TEXT = (
    "This prediction is based on statistical patterns learned from the WELFake training dataset "
    "and should not be treated as a substitute for professional fact-checking or journalistic verification."
)


class NewsPredictor:
    """
    Wrapper for loading the serialized model, TF-IDF vectorizer,
    and executing the full inference pipeline.
    """

    def __init__(
        self,
        model_path: str = "models/best_model.pkl",
        vectorizer_path: str = "models/tfidf_vectorizer.pkl",
        metadata_path: str = "models/model_metadata.pkl"
    ):
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.metadata_path = metadata_path

        self.model = None
        self.vectorizer = None
        self.metadata = None
        self.load_artifacts()

    def load_artifacts(self):
        """Loads model, vectorizer, and metadata from disk."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file not found at '{self.model_path}'. "
                f"Please run the training pipeline first: python train.py"
            )
        if not os.path.exists(self.vectorizer_path):
            raise FileNotFoundError(
                f"TF-IDF vectorizer not found at '{self.vectorizer_path}'. "
                f"Please run the training pipeline first: python train.py"
            )

        self.model = joblib.load(self.model_path)
        self.vectorizer = joblib.load(self.vectorizer_path)

        if os.path.exists(self.metadata_path):
            try:
                self.metadata = joblib.load(self.metadata_path)
            except Exception:
                self.metadata = {}
        else:
            self.metadata = {}

        logger.info("Predictor artifacts loaded successfully.")

    def explain_prediction(self, preprocessed_text: str, top_k: int = 6) -> list:
        """
        Extracts the most influential words from the user's article based on
        TF-IDF weights and model coefficients.

        Returns
        -------
        list of dict : List of influential words with their association and importance.
        """
        if not preprocessed_text or not hasattr(self.vectorizer, "vocabulary_"):
            return []

        tokens = set(preprocessed_text.split())
        vocab = self.vectorizer.vocabulary_
        idf = self.vectorizer.idf_

        # Extract model coefficients if linear / calibrated
        coef = None
        if hasattr(self.model, "coef_"):
            coef = self.model.coef_[0]
        elif hasattr(self.model, "estimator") and hasattr(self.model.estimator, "coef_"):
            coef = self.model.estimator.coef_[0]
        elif hasattr(self.model, "calibrated_classifiers_"):
            coefs = [clf.estimator.coef_[0] for clf in self.model.calibrated_classifiers_ if hasattr(clf.estimator, "coef_")]
            if coefs:
                coef = np.mean(coefs, axis=0)

        word_influences = []
        for word in tokens:
            if word in vocab:
                idx = vocab[word]
                word_idf = idf[idx]
                if coef is not None:
                    weight = coef[idx]
                    impact = "Real News indicator" if weight > 0 else "Fake News indicator"
                    score = abs(weight) * word_idf
                else:
                    impact = "Influential term"
                    score = word_idf

                word_influences.append({
                    "word": word,
                    "impact": impact,
                    "score": float(score)
                })

        # Sort by score descending
        word_influences.sort(key=lambda x: x["score"], reverse=True)
        return word_influences[:top_k]

    def predict(self, title: str, text: str) -> dict:
        """
        Predicts whether a news item is Fake (0) or Real (1).

        Parameters
        ----------
        title : str
            News headline or title.
        text : str
            News article body text.

        Returns
        -------
        dict : Prediction details including label, verdict, confidence, explanation, and model name.
        """
        # 1. Validation
        title_clean = title.strip() if isinstance(title, str) else ""
        text_clean = text.strip() if isinstance(text, str) else ""

        if not title_clean and not text_clean:
            return {
                "success": False,
                "error": "Both news headline and article text are empty. Please provide news content to verify."
            }

        combined_raw = f"{title_clean} {text_clean}".strip()
        if len(combined_raw) < 15:
            return {
                "success": False,
                "error": "The provided text is too brief to analyze reliably. Please provide a more detailed headline or article snippet."
            }

        # 2. Text Preprocessing
        clean_input = clean_text(combined_raw, apply_stemming=True)
        if len(clean_input.strip()) == 0:
            return {
                "success": False,
                "error": "After removing stop-words and non-alphabetic symbols, no recognizable vocabulary remained for analysis."
            }

        # 3. TF-IDF Transformation
        tfidf_features = self.vectorizer.transform([clean_input])

        # 4. Model Prediction & Probability
        pred_label = int(self.model.predict(tfidf_features)[0])

        confidence = 0.0
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(tfidf_features)[0]
            confidence = float(probs[pred_label])
        elif hasattr(self.model, "decision_function"):
            score = float(self.model.decision_function(tfidf_features)[0])
            prob_real = 1.0 / (1.0 + np.exp(-score))
            confidence = prob_real if pred_label == 1 else (1.0 - prob_real)
        else:
            confidence = 0.85

        verdict = "REAL NEWS" if pred_label == 1 else "FAKE NEWS"
        model_name = self.metadata.get("best_model_name", type(self.model).__name__)

        # 5. Extract Explainable Tokens
        top_features = self.explain_prediction(clean_input, top_k=6)

        return {
            "success": True,
            "prediction": verdict,
            "label": pred_label,
            "confidence_percentage": round(confidence * 100, 2),
            "model_used": model_name,
            "cleaned_text_preview": clean_input[:200] + ("..." if len(clean_input) > 200 else ""),
            "influential_tokens": top_features,
            "disclaimer": DISCLAIMER_TEXT
        }
