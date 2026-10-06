# FAKE-NEWS DETECTION SYSTEM USING MACHINE LEARNING

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Framework Streamlit](https://img.shields.io/badge/framework-Streamlit-red.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/library-scikit--learn-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end, production-ready Natural Language Processing (NLP) and Machine Learning system for automated detection, classification, and explainability of fake and real news articles.

---

## 1. Project Title
**Fake-News Detection System Using Machine Learning**

---

## 2. Project Overview
Misinformation and digital disinformation campaigns present significant threats to societal cohesion, democratic institutions, and public health. This system provides a transparent, automated screening pipeline that ingests news headlines and article bodies, processes textual data using modern NLP techniques, extracts statistical n-gram representations via TF-IDF vectorization, and predicts authenticity using five distinct supervised machine learning classifiers.

---

## 3. Problem Statement
Manual fact-checking and editorial verification require extensive time and domain expertise, making it impossible to audit the sheer volume of digital content published daily. An automated, computationally efficient, and interpretable classification system is required to quickly flag suspicious, sensationalist, or fabricated narratives for downstream human review.

---

## 4. Project Objectives
- Ingest and preprocess the large-scale **WELFake benchmark dataset** (72,134 news articles).
- Standardize unstructured text using regex-based sanitization, lowercasing, stop-word elimination, and Porter Stemming.
- Generate high-dimensional feature spaces using **TF-IDF vectorization** (unigrams and bigrams with sublinear term-frequency scaling).
- Train, evaluate, and benchmark **five core machine learning algorithms**:
  1. Logistic Regression
  2. Multinomial Naive Bayes
  3. Support Vector Machine (LinearSVC with Probability Calibration)
  4. Random Forest Classifier
  5. Gradient Boosting Classifier
- Provide automated selection of the best-performing model based primarily on **F1-score** on unseen test data.
- Offer model explainability by extracting salient linguistic tokens that drove the classification.
- Deploy an intuitive, responsive **Streamlit Web Application** featuring live news prediction, confidence meters, performance dashboards, and academic viva reference material.

---

## 5. Dataset Information
- **Dataset Name:** WELFake – Fake and Real News Dataset
- **Total Articles:** 72,134 records
- **Class Balance:**
  - **Fake News (`0`):** 37,106 records (51.4%)
  - **Real News (`1`):** 35,028 records (48.6%)
- **Data Attributes:**
  - `Serial Number`: Unique numeric identifier
  - `Title`: Headline of the news article
  - `Text`: Complete news body
  - `Label`: Binary ground-truth target (`0` = Fake News, `1` = Real News)

---

## 6. Dataset Source & Link
- **Source:** Kaggle Fake News Classification Benchmark
- **Curated By:** Saurabh Shahane
- **Kaggle Dataset URL:** [https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification)

---

## 7. Key Features
- **Deterministic NLP Preprocessing:** Unified text transformation pipeline used identically across training and inference to avoid train-test data leakage.
- **Leak-Free Train-Test Split:** Stratified 80/20 train/test partition ensures balanced class distributions in both subsets.
- **Multi-Model Comparison:** Evaluates Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
- **Diagnostic Visualizations:** Generates publication-ready confusion matrices, combined ROC curves, and linguistic feature importance charts.
- **Token-Level Explainability:** Highlights influential vocabulary tokens in the user's input that swung the classification verdict.
- **Fast Interactive Web UI:** Built with Streamlit with `@st.cache_resource` for instant response times.
- **Academic Fact-Checking Disclaimer:** Clear contextual messaging to ensure ethical AI deployment.

---

## 8. Technology Stack
- **Language:** Python 3.10 / 3.11 / 3.14
- **Data Manipulation:** `pandas`, `numpy`, `pyarrow`
- **Machine Learning:** `scikit-learn`, `scipy`, `joblib`
- **Natural Language Processing:** `nltk`
- **Visualization:** `matplotlib`, `seaborn`
- **Web Application:** `streamlit`

---

## 9. Machine Learning Algorithms
1. **Logistic Regression:** Linear probabilistic model with L-BFGS optimization and L2 regularization.
2. **Multinomial Naive Bayes:** Bayesian probabilistic classifier with Laplace smoothing ($\alpha=0.1$).
3. **Support Vector Machine (LinearSVC):** Convex maximum-margin hyperplane classifier calibrated via `CalibratedClassifierCV` for reliable confidence probabilities.
4. **Random Forest Classifier:** Bagged ensemble of decorrelated decision trees using Gini impurity splits.
5. **Gradient Boosting Classifier:** Sequential ensemble that minimizes logistic loss through gradient descent over decision trees.

---

## 10. Project Workflow
```
[ WELFake_Dataset.csv ]
         │
         ▼
[ Data Cleaning: Deduplication, Missing Value Imputation ]
         │
         ▼
[ NLP Preprocessing: Lowercase, Regex Strip, Stop-words, Porter Stemmer ]
         │
         ▼
[ Stratified 80/20 Train-Test Split ]
         │
         ▼
[ TF-IDF Feature Extraction (Max Features: 40,000, N-grams: 1-2) ]
         │
         ├────────────────────────────────────────┐
         │                                        │
         ▼                                        ▼
[ Train 5 Machine Learning Models ]     [ Transform Test Set ]
         │                                        │
         └───────────────────┬────────────────────┘
                             │
                             ▼
               [ Model Evaluation & Benchmarks ]
                             │
                             ▼
                 [ Automatic Best Model Selection ]
                             │
                             ▼
            [ Serialize Models & Metadata (.pkl) ]
                             │
                             ▼
               [ Interactive Streamlit Web App ]
```

---

## 11. Project Directory Structure
```
Fake News Detection/
│
├── data/
│   ├── WELFake_Dataset.csv          # Raw benchmark dataset (Kaggle)
│   └── preprocessed_cache.parquet   # Cached cleaned text for fast execution
│
├── notebooks/
│   └── fake_news_detection.ipynb    # Walkthrough Jupyter Notebook for demonstrations
│
├── src/
│   ├── data_preprocessing.py        # Text cleaning, normalization, and label alignment
│   ├── feature_engineering.py       # TF-IDF vectorization and serialization
│   ├── train_models.py              # Model training orchestration and selection
│   ├── evaluate_models.py           # Evaluation metrics, confusion matrix, ROC plots
│   └── prediction.py                # Single-article inference and token explainability
│
├── models/
│   ├── logistic_regression.pkl      # Trained Logistic Regression model
│   ├── naive_bayes.pkl              # Trained Naive Bayes model
│   ├── svm.pkl                      # Trained Support Vector Machine model
│   ├── random_forest.pkl            # Trained Random Forest model
│   ├── gradient_boosting.pkl        # Trained Gradient Boosting model
│   ├── tfidf_vectorizer.pkl         # Fitted TF-IDF Vectorizer
│   ├── best_model.pkl               # Selected best performing model
│   ├── model_metadata.pkl           # Hyperparameters and evaluation scores
│   └── dataset_summary.pkl          # Corpus record metrics
│
├── reports/
│   ├── model_comparison.csv         # Comparative metrics table
│   ├── confusion_matrix.png         # Multi-model confusion matrix grid
│   ├── confusion_matrix_best.png    # Annotated best model confusion matrix
│   ├── roc_curve.png                # Combined ROC curves with AUC scores
│   └── feature_importance.png       # Linguistic keyword explainability chart
│
├── app.py                           # Streamlit Web Application
├── train.py                         # Complete command-line training pipeline
├── requirements.txt                 # Project dependencies
├── README.md                        # Documentation and research overview
└── .gitignore                       # Git exclusion rules
```

---

## 12. Installation & Setup

### Step 1: Clone Repository / Open Project Directory
```bash
cd "c:/Fake News Detection"
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 13. Dataset Setup
The dataset `WELFake_Dataset.csv` should be placed inside the `data/` folder:
```
data/WELFake_Dataset.csv
```
If the file is not present, you can download it directly from Kaggle:
[https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification)

---

## 14. Training Instructions
To execute the end-to-end training pipeline, run:
```bash
python train.py
```
This script will:
1. Ingest and validate the dataset.
2. Clean and preprocess all articles.
3. Perform stratified 80/20 train/test splitting.
4. Fit and save the TF-IDF vectorizer.
5. Train all five classification models.
6. Evaluate test metrics and generate diagnostic plots.
7. Select and save `best_model.pkl` and `model_metadata.pkl`.

---

## 15. Running the Streamlit Application
Launch the web interface locally:
```bash
streamlit run app.py
```
Open your web browser at `http://localhost:8501`.

---

## 16. Deployment Instructions (Streamlit Community Cloud)
1. Initialize a Git repository and commit the project files:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Fake News Detection System"
   ```
2. Create a new repository on GitHub (e.g., `fake-news-detection`).
3. Link and push your code:
   ```bash
   git remote add origin https://github.com/<your-username>/fake-news-detection.git
   git branch -M main
   git push -u origin main
   ```
4. Go to [share.streamlit.io](https://share.streamlit.io/), link your GitHub account, select the repository, set main file path to `app.py`, and click **Deploy**.

---

## 17. Project Limitations
- **Corpus Specificity:** Predictions reflect linguistic correlations in the WELFake dataset.
- **Evolving Misinformation:** Satires, sarcasm, and novel misinformation narratives not represented in training data may lead to false classifications.
- **Text-Only Heuristic:** The model relies on linguistic stylometry and vocabulary distributions; it cannot verify real-world facts or query external databases.
- **Model Drift:** Periodic retraining is required as news reporting styles and topics evolve over time.

---

## 18. Future Enhancements
- Integration with external fact-checking APIs (e.g., ClaimReview, Google Fact Check Tools).
- Deep learning transformer models (BERT, RoBERTa) for contextual embeddings.
- Multimodal verification (cross-modal image and text authenticity checks).
- Source credibility and domain authority graph modeling.
- Multilingual detection support across diverse regional languages.
