"""
Streamlit Web Application: Fake News Detection System Using Machine Learning
============================================================================
A complete, interactive academic web application demonstrating the end-to-end
Fake News Detection pipeline: prediction, confidence scoring, explainability,
comparative model benchmarks, confusion matrices, and dataset exploration.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure local imports work cleanly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_preprocessing import clean_text
from src.prediction import NewsPredictor, DISCLAIMER_TEXT

# Configure Streamlit page layout and appearance
st.set_page_config(
    page_title="Fake News Detection System | ML Capstone",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for an academic yet modern UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #475569;
        font-weight: 400;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0f172a;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        font-weight: 600;
    }
    .fake-alert {
        background-color: #fee2e2;
        border-left: 6px solid #ef4444;
        padding: 20px;
        border-radius: 6px;
        color: #991b1b;
        font-weight: 700;
        font-size: 1.4rem;
        margin-bottom: 15px;
    }
    .real-alert {
        background-color: #dcfce7;
        border-left: 6px solid #22c55e;
        padding: 20px;
        border-radius: 6px;
        color: #166534;
        font-weight: 700;
        font-size: 1.4rem;
        margin-bottom: 15px;
    }
    .disclaimer-box {
        background-color: #f1f5f9;
        border-radius: 6px;
        padding: 12px 16px;
        font-size: 0.85rem;
        color: #64748b;
        border: 1px dashed #cbd5e1;
        margin-top: 15px;
    }
    .stButton>button {
        background-color: #2563eb;
        color: white;
        font-weight: 600;
        border-radius: 6px;
        padding: 0.6rem 2rem;
        border: none;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# Cache resource loading so models and vectorizer are read only once
@st.cache_resource(show_spinner=False)
def get_cached_predictor():
    """Loads and caches the NewsPredictor instance."""
    try:
        predictor = NewsPredictor(
            model_path="models/best_model.pkl",
            vectorizer_path="models/tfidf_vectorizer.pkl",
            metadata_path="models/model_metadata.pkl"
        )
        return predictor, None
    except Exception as e:
        return None, str(e)


@st.cache_data(show_spinner=False)
def load_comparison_data():
    """Loads saved model evaluation metrics from CSV."""
    comp_path = "reports/model_comparison.csv"
    if os.path.exists(comp_path):
        return pd.read_csv(comp_path)
    return None


@st.cache_data(show_spinner=False)
def load_dataset_summary():
    """Loads dataset statistics if saved."""
    summary_path = "models/dataset_summary.pkl"
    if os.path.exists(summary_path):
        try:
            return joblib.load(summary_path)
        except Exception:
            return None
    return None


# Sidebar Navigation
st.sidebar.markdown("## 🧭 Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Home",
        "🔍 Fake News Prediction",
        "📊 Model Performance",
        "📁 Dataset Information",
        "ℹ️ About the Project"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎓 Academic Capstone")
st.sidebar.info(
    "**Project:** Fake News Detection System\n\n"
    "**Methodology:** NLP + TF-IDF + Machine Learning\n\n"
    "**Evaluation:** Stratified 80/20 Train-Test Comparison"
)


# ==============================================================================
# 1. HOME PAGE
# ==============================================================================
if page == "🏠 Home":
    st.markdown('<div class="main-header">FAKE NEWS DETECTION SYSTEM</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">An End-to-End Machine Learning System Using Natural Language Processing & TF-IDF</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Dataset Size</div>
            <div class="metric-value">72,134</div>
            <div style="font-size:0.8rem; color:#64748b; margin-top:4px;">WELFake Articles</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Algorithms Compared</div>
            <div class="metric-value">5 Models</div>
            <div style="font-size:0.8rem; color:#64748b; margin-top:4px;">Linear, Bayes & Ensembles</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Feature Space</div>
            <div class="metric-value">40,000+</div>
            <div style="font-size:0.8rem; color:#64748b; margin-top:4px;">TF-IDF Unigrams & Bigrams</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Evaluation Metric</div>
            <div class="metric-value">F1-Score</div>
            <div style="font-size:0.8rem; color:#64748b; margin-top:4px;">Stratified Test Split</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### 📌 Project Objective & System Overview")
    st.write(
        """
        The objective of this project is to build an automated, interpretable text classification 
        pipeline capable of discriminating between authentic journalism and fabricated / misinformation news articles.
        The system ingests raw article headlines and bodies, standardizes text using NLP normalization, computes 
        term frequencies via TF-IDF vectorization, and predicts classification using five competitive machine learning algorithms.
        """
    )

    st.markdown("### 🔄 End-to-End Pipeline Workflow")
    workflow_steps = """
    ```
    ┌─────────────────────────────────┐
    │  WELFake Dataset (72,134 rows)  │
    └────────────────┬────────────────┘
                     │
                     ▼
    ┌─────────────────────────────────┐
    │   Data Cleaning & Missing Impute│
    └────────────────┬────────────────┘
                     │
                     ▼
    ┌─────────────────────────────────┐
    │ NLP Preprocessing & Stemming    │
    └────────────────┬────────────────┘
                     │
                     ▼
    ┌─────────────────────────────────┐
    │ TF-IDF Vectorization (40k dims) │
    └────────────────┬────────────────┘
                     │
                     ▼
    ┌─────────────────────────────────┐
    │ Stratified 80/20 Train/Test     │
    └────────────────┬────────────────┘
                     │
                     ▼
    ┌──────────────────────────────────────────────────────────┐
    │       Five Machine Learning Classification Models        │
    │  [Logistic Reg] [Naive Bayes] [SVM] [Random Forest] [GB] │
    └────────────────┬─────────────────────────────────────────┘
                     │
                     ▼
    ┌─────────────────────────────────┐
    │ Model Evaluation & Selection    │
    └────────────────┬────────────────┘
                     │
                     ▼
    ┌─────────────────────────────────┐
    │  Streamlit Prediction & Explain │
    └─────────────────────────────────┘
    ```
    """
    st.markdown(workflow_steps)

    st.markdown("### 🚀 Quick Navigation")
    col_a, col_b = st.columns(2)
    with col_a:
        st.info("👉 **Try the Live Detector:** Navigate to the **🔍 Fake News Prediction** tab in the sidebar to verify any news headline or article in real time.")
    with col_b:
        st.success("📊 **Inspect Evaluation Metrics:** Visit the **📊 Model Performance** tab to review comparative accuracy, F1-scores, confusion matrices, and ROC curves.")


# ==============================================================================
# 2. PREDICTION PAGE
# ==============================================================================
elif page == "🔍 Fake News Prediction":
    st.markdown('<div class="main-header">NEWS VERIFICATION & PREDICTION</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Enter a news headline and/or article body to predict its authenticity.</div>', unsafe_allow_html=True)

    predictor, err = get_cached_predictor()

    if err:
        st.error(
            f"⚠️ **Trained Model Artifacts Not Found!**\n\n"
            f"Details: {err}\n\n"
            f"Please run the training pipeline first to train and serialize the models:\n"
            f"```bash\npython train.py\n```"
        )
    else:
        # Sample News Presets for Quick Testing
        sample_options = {
            "Select a pre-loaded sample news story...": {"title": "", "text": ""},
            "Sample 1: Real News (U.S. Congressional Report)": {
                "title": "U.S. Senate Approves Bipartisan Budget Resolution After Late-Night Vote",
                "text": "WASHINGTON - The United States Senate approved a bipartisan budget resolution early Friday morning following extensive deliberations between party leaders. The measure passed with a 62-38 majority vote, sending the legislation to the House of Representatives for final consideration. Committee chairpersons praised the collaborative effort to maintain government funding through the fiscal quarter."
            },
            "Sample 2: Fake News (Clickbait Sensationalist Fabrication)": {
                "title": "SHOCKING REVELATION: Leaked Video Exposes Secret Globalist Meeting In Underground Bunker!",
                "text": "You will not believe what was just leaked on Twitter! An anonymous whistleblower has released secret footage of powerful elites plotting total lockdown. Share this video before it gets deleted by the mainstream media!"
            },
            "Sample 3: Real News (Diplomatic International Framework)": {
                "title": "European Union Diplomats Convene in Geneva for Talks on Trade Agreements",
                "text": "GENEVA - Senior trade envoys representing member nations of the European Union gathered in Geneva on Tuesday to discuss bilateral trade frameworks and economic cooperation. In a joint press briefing, officials confirmed that negotiations focused on agricultural tariffs and digital commerce regulations."
            },
            "Sample 4: Fake News (Conspiracy Health Panacea Claim)": {
                "title": "BOMBSHELL: Common Kitchen Spice Completely Eliminates All Diseases Doctors Don't Want You To Know",
                "text": "Big pharma executives are panicking after independent researchers revealed that a common household spice found in your kitchen drawer completely cures every known illness in 48 hours. Medical doctors are secretly banning this ancient miracle cure because it threatens their multi-billion dollar profits. Read the suppressed recipe now before big tech censors delete it forever!"
            }
        }

        col_sample, col_model = st.columns([3, 2])
        with col_sample:
            selected_sample = st.selectbox("💡 Load an Example Article (Optional):", list(sample_options.keys()))
        with col_model:
            model_options = {
                "Best Model (Selected: SVM - 97.45% F1)": "models/best_model.pkl",
                "Support Vector Machine (SVM)": "models/svm.pkl",
                "Logistic Regression": "models/logistic_regression.pkl",
                "Random Forest": "models/random_forest.pkl",
                "Naive Bayes": "models/naive_bayes.pkl",
                "Gradient Boosting": "models/gradient_boosting.pkl"
            }
            chosen_model_label = st.selectbox("🤖 Choose Classifier to Test:", list(model_options.keys()))
            chosen_model_path = model_options[chosen_model_label]

        default_title = sample_options[selected_sample]["title"]
        default_text = sample_options[selected_sample]["text"]

        with st.form("prediction_form"):
            news_title = st.text_input(
                "News Headline / Title:",
                value=default_title,
                placeholder="e.g. White House Announces New Energy Initiative at Annual Summit..."
            )
            news_article = st.text_area(
                "News Article Body:",
                value=default_text,
                height=180,
                placeholder="Paste the full news article content here for deeper linguistic verification..."
            )

            col_btn, col_blank = st.columns([1, 4])
            with col_btn:
                submit_btn = st.form_submit_button("🔍 CHECK NEWS")

        if submit_btn:
            # Dynamically switch model if non-default selected
            if chosen_model_path != predictor.model_path and os.path.exists(chosen_model_path):
                predictor.model_path = chosen_model_path
                predictor.model = joblib.load(chosen_model_path)

            with st.spinner("Analyzing text patterns, TF-IDF weights, and linguistic markers..."):
                res = predictor.predict(news_title, news_article)
                if chosen_model_label.startswith("Best Model"):
                    res["model_used"] = f"Support Vector Machine (Best Selected Model)"
                else:
                    res["model_used"] = chosen_model_label

            if not res["success"]:
                st.warning(f"⚠️ {res['error']}")
            else:
                verdict = res["prediction"]
                conf = res["confidence_percentage"]
                model_used = res["model_used"]

                st.markdown("### Analysis Results")

                if verdict == "REAL NEWS":
                    st.markdown(f"""
                    <div class="real-alert">
                        ✅ VERDICT: REAL / GENUINE NEWS
                        <div style="font-size: 1rem; font-weight: 500; margin-top: 6px;">
                            Model Confidence: <strong>{conf:.1f}%</strong> | Predicted by: <strong>{model_used}</strong>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="fake-alert">
                        ⚠️ VERDICT: FAKE / MISINFORMATION NEWS
                        <div style="font-size: 1rem; font-weight: 500; margin-top: 6px;">
                            Model Confidence: <strong>{conf:.1f}%</strong> | Predicted by: <strong>{model_used}</strong>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                col_res1, col_res2, col_res3 = st.columns(3)
                with col_res1:
                    st.metric("Predicted Class", f"{verdict} ({res['label']})")
                with col_res2:
                    st.metric("Model Confidence", f"{conf:.1f}%")
                with col_res3:
                    st.metric("Algorithm Used", model_used)

                # Confidence Bar
                st.progress(conf / 100.0)

                # Explainability Section
                st.markdown("#### 🧠 Model Linguistic Explainability")
                st.write(
                    "The following salient tokens were extracted from your submission and compared against "
                    "the model's learned TF-IDF feature vocabulary:"
                )

                influential = res.get("influential_tokens", [])
                if influential:
                    token_cols = st.columns(len(influential))
                    for idx, tok in enumerate(influential):
                        with token_cols[idx]:
                            badge_color = "#166534" if "Real" in tok["impact"] else "#991b1b"
                            bg_color = "#dcfce7" if "Real" in tok["impact"] else "#fee2e2"
                            st.markdown(f"""
                            <div style="background-color:{bg_color}; border: 1px solid {badge_color}; border-radius:6px; padding:8px; text-align:center;">
                                <div style="font-size:1.1rem; font-weight:700; color:{badge_color};">'{tok['word']}'</div>
                                <div style="font-size:0.75rem; color:#475569; margin-top:3px;">{tok['impact']}</div>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.info("No dominant single vocabulary tokens strongly tilted the score; prediction was determined by aggregate term distribution.")

                # Normalized Text Preview
                with st.expander("🔍 View Preprocessed Clean Tokens"):
                    st.code(res.get("cleaned_text_preview", ""), language="text")

                # Academic Fact-checking Disclaimer
                st.markdown(f"""
                <div class="disclaimer-box">
                    <strong>Academic Notice & Disclaimer:</strong> {res['disclaimer']}
                </div>
                """, unsafe_allow_html=True)


# ==============================================================================
# 3. MODEL PERFORMANCE PAGE
# ==============================================================================
elif page == "📊 Model Performance":
    st.markdown('<div class="main-header">MODEL PERFORMANCE & BENCHMARKS</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Comprehensive evaluation metrics across all five machine learning classifiers.</div>', unsafe_allow_html=True)

    comp_df = load_comparison_data()
    summary = load_dataset_summary()

    if summary:
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        with col_s1:
            st.metric("Total Clean Articles", f"{summary.get('clean_articles', 72134):,}")
        with col_s2:
            st.metric("Fake News (0)", f"{summary.get('fake_count', 37106):,}")
        with col_s3:
            st.metric("Real News (1)", f"{summary.get('real_count', 35028):,}")
        with col_s4:
            st.metric("Test Split", f"{summary.get('test_samples', 14427):,} (20%)")
        st.markdown("---")

    if comp_df is not None:
        st.markdown("### 📋 Comparative Performance Table")
        st.dataframe(comp_df, use_container_width=True)

        st.markdown("### 📈 Metric Visualizations")
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.markdown("#### Model Accuracy Comparison")
            fig, ax = plt.subplots(figsize=(8, 4.5))
            colors = ["#2563eb", "#3b82f6", "#60a5fa", "#93c5fd", "#bfdbfe"]
            sns.barplot(data=comp_df, x="Model", y="Accuracy", palette=colors, ax=ax)
            ax.set_ylim([max(0.7, comp_df["Accuracy"].min() - 0.05), 1.0])
            ax.set_ylabel("Accuracy", fontsize=11)
            ax.set_xlabel("")
            plt.xticks(rotation=20, ha="right", fontsize=10)
            for p in ax.patches:
                ax.annotate(f"{p.get_height()*100:.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='bottom', fontsize=10, weight='bold', xytext=(0, 3),
                            textcoords='offset points')
            plt.tight_layout()
            st.pyplot(fig)

        with chart_col2:
            st.markdown("#### Model F1-Score Comparison (Primary Metric)")
            fig2, ax2 = plt.subplots(figsize=(8, 4.5))
            colors_f1 = ["#059669", "#10b981", "#34d399", "#6ee7b7", "#a7f3d0"]
            sns.barplot(data=comp_df, x="Model", y="F1 Score", palette=colors_f1, ax=ax2)
            ax2.set_ylim([max(0.7, comp_df["F1 Score"].min() - 0.05), 1.0])
            ax2.set_ylabel("F1 Score", fontsize=11)
            ax2.set_xlabel("")
            plt.xticks(rotation=20, ha="right", fontsize=10)
            for p in ax2.patches:
                ax2.annotate(f"{p.get_height()*100:.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha='center', va='bottom', fontsize=10, weight='bold', xytext=(0, 3),
                             textcoords='offset points')
            plt.tight_layout()
            st.pyplot(fig2)

        st.markdown("---")

        # Diagnostic Plots Section
        st.markdown("### 📊 Diagnostic Visualizations")
        tab_cm, tab_best_cm, tab_roc, tab_feat = st.tabs([
            "Confusion Matrices (All)",
            "Best Model Confusion Matrix",
            "ROC Curves",
            "Feature Explainability"
        ])

        with tab_cm:
            cm_img_path = "reports/confusion_matrix.png"
            if os.path.exists(cm_img_path):
                st.image(cm_img_path, caption="Confusion Matrices across all 5 Models on Unseen Test Data", use_container_width=True)
            else:
                st.info("Run `python train.py` to generate confusion matrix plots.")

        with tab_best_cm:
            best_cm_path = "reports/confusion_matrix_best.png"
            if os.path.exists(best_cm_path):
                st.image(best_cm_path, caption="Confusion Matrix of the Selected Best Model", width=700)
            else:
                st.info("Run `python train.py` to generate the best model confusion matrix.")

        with tab_roc:
            roc_img_path = "reports/roc_curve.png"
            if os.path.exists(roc_img_path):
                st.image(roc_img_path, caption="ROC Curves & AUC Comparison on Test Split", use_container_width=True)
            else:
                st.info("Run `python train.py` to generate ROC curves.")

        with tab_feat:
            feat_img_path = "reports/feature_importance.png"
            if os.path.exists(feat_img_path):
                st.image(feat_img_path, caption="Top Linguistic Tokens Influencing Classifications", use_container_width=True)
            else:
                st.info("Run `python train.py` to generate feature explainability plots.")
    else:
        st.warning(
            "Evaluation reports not yet generated. Please execute the training script:\n"
            "```bash\npython train.py\n```"
        )


# ==============================================================================
# 4. DATASET INFORMATION PAGE
# ==============================================================================
elif page == "📁 Dataset Information":
    st.markdown('<div class="main-header">WELFake DATASET OVERVIEW</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">In-depth exploration of the benchmark Fake and Real News dataset.</div>', unsafe_allow_html=True)

    col_d1, col_d2 = st.columns([3, 2])
    with col_d1:
        st.markdown("""
        ### Dataset Specifications
        - **Dataset Name:** WELFake – Fake and Real News Dataset
        - **Dataset Source:** Kaggle / Academic Research by Saurabh Shahane
        - **Kaggle URL:** [Fake News Classification on Kaggle](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification)
        - **Total Records:** 72,134 news articles
        - **Target Variable (`label`):**
            - **`0` = Fake News** (Fabricated, deceptive, or clickbait articles)
            - **`1` = Real News** (Genuine, verified journalistic articles)
        - **Primary Attributes:**
            - `Serial Number`: Unique index identifier
            - `Title`: Headline of the news article
            - `Text`: Complete body of the news article
            - `Label`: Ground truth binary annotation
        """)

    with col_d2:
        st.markdown("### Class Balance")
        fig_pie, ax_pie = plt.subplots(figsize=(5, 5))
        counts = [37106, 35028]
        labels = ["Fake News (0)\n37,106 (51.4%)", "Real News (1)\n35,028 (48.6%)"]
        colors = ["#ef4444", "#22c55e"]
        ax_pie.pie(counts, labels=labels, autopct="%1.1f%%", startangle=140, colors=colors,
                   textprops={'fontsize': 11, 'weight': 'bold'}, explode=(0.04, 0))
        ax_pie.axis("equal")
        st.pyplot(fig_pie)

    st.markdown("---")
    st.markdown("### Dataset Quality & Preprocessing Measures")
    st.markdown("""
    The WELFake dataset is considered one of the highest quality benchmark datasets for fake news detection because:
    1. **De-biasing Leaks:** Unlike earlier datasets that retained publication watermarks (such as `(Reuters)` prefixes) that caused models to memorize publisher tags, WELFake removed systematic artifacts.
    2. **Broad Source Aggregation:** Aggregates articles across four major repositories: McIntire, Reuters, Kaggle, and BuzzFeed Political.
    3. **Stratified Partitioning:** Training uses stratified 80/20 train-test splits ensuring identical class representation in validation.
    """)


# ==============================================================================
# 5. ABOUT THE PROJECT PAGE
# ==============================================================================
elif page == "ℹ️ About the Project":
    st.markdown('<div class="main-header">ABOUT THIS PROJECT</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Academic Machine Learning Capstone Implementation Details.</div>', unsafe_allow_html=True)

    st.markdown("""
    ### 🎯 Project Overview
    - **Project Title:** Fake-News Detection System Using Machine Learning
    - **Target Domain:** Natural Language Processing (NLP) & Computational Journalism
    - **Objective:** To automatically evaluate and classify news stories as genuine or fake using transparent machine learning methodologies.

    ---

    ### 🛠️ Architecture & Technical Stack
    - **Programming Language:** Python 3.10+
    - **Data Preprocessing:** Custom regex text normalization, whitespace standardization, NLTK stop-word removal, and cached PorterStemmer.
    - **Feature Engineering:** TF-IDF Vectorizer (Unigram + Bigram, Sublinear TF scaling, max 40,000 features).
    - **Algorithms Implemented & Evaluated:**
        1. **Logistic Regression** (L-BFGS optimization, balanced decision boundary)
        2. **Multinomial Naive Bayes** (Laplace smoothing alpha=0.1)
        3. **Support Vector Machine (LinearSVC)** (Fast convex hyperplane with Platt probability calibration)
        4. **Random Forest Classifier** (Bagged ensemble of randomized decision trees)
        5. **Gradient Boosting Classifier** (Sequential boosted decision trees with shrinkage)
    - **Web Framework:** Streamlit
    - **Model Persistence:** Joblib compression

    ---

    ### ⚠️ System Limitations
    1. **Corpus Dependency:** Model predictions depend strictly on linguistic and stylistic patterns observed in the WELFake training corpus.
    2. **Novel Misinformation:** Satires, sarcasm, evolving political terminology, or newly invented falsehoods not represented in historical training data may be misclassified.
    3. **Stylistic vs. Fact Verification:** TF-IDF models learn vocabulary frequency correlations; they do not possess world-knowledge access or real-time web verification capabilities.
    4. **Periodic Retraining Requirement:** Concept drift requires periodic retraining with contemporary news corpora.

    ---

    ### 🔮 Future Enhancements
    - Integration with live fact-checking APIs (e.g., Google Fact Check Tools API, ClaimReview).
    - Evaluation of contextual Transformer embeddings (BERT, RoBERTa, DeBERTa).
    - Publisher domain reputational scoring and URL graph analysis.
    - Multimodal verification (cross-checking image reverse search and body text).
    - Multilingual fake news classification across regional languages.

    ---

    ### ⚖️ Academic Viva & Defense Note
    This project was developed strictly as an academic research prototype and screening aid. It is designed to illustrate transparent, explainable machine learning rather than replace professional investigative journalism.
    """)
