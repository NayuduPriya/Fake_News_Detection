"""
Streamlit Web Application: Fake News Detection System Using Machine Learning
============================================================================
An academic, presentation-friendly web application for B.Tech project review:
- 4 Streamlined Navigation Sections: Home, Fake News Detection, Model Insights, About & Dataset
- Real-time prediction with calibrated confidence scores and token explainability
- Empirical benchmark metrics across 5 machine learning models
- Publication-quality diagnostic plots: Confusion Matrix, ROC Curves, Feature Importance
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
    page_title="Fake News Detection System",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for a dark navy academic UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        font-weight: 500;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .metric-value {
        font-size: 1.7rem;
        font-weight: 800;
        color: #1e3a8a;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #64748b;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .fake-alert {
        background-color: #fef2f2;
        border-left: 6px solid #dc2626;
        padding: 18px;
        border-radius: 6px;
        color: #991b1b;
        font-weight: 700;
        font-size: 1.35rem;
        margin-bottom: 15px;
    }
    .real-alert {
        background-color: #f0fdf4;
        border-left: 6px solid #16a34a;
        padding: 18px;
        border-radius: 6px;
        color: #166534;
        font-weight: 700;
        font-size: 1.35rem;
        margin-bottom: 15px;
    }
    .disclaimer-box {
        background-color: #f8fafc;
        border-radius: 6px;
        padding: 12px 16px;
        font-size: 0.85rem;
        color: #64748b;
        border: 1px dashed #cbd5e1;
        margin-top: 15px;
    }
    .tech-card {
        background-color: #f1f5f9;
        border-radius: 6px;
        padding: 10px 14px;
        text-align: center;
        font-weight: 600;
        color: #1e293b;
        border: 1px solid #cbd5e1;
    }
    .stButton>button {
        background-color: #1e3a8a;
        color: white;
        font-weight: 600;
        border-radius: 6px;
        padding: 0.55rem 1.8rem;
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


# Sidebar Navigation state management
if "page" not in st.session_state:
    st.session_state.page = "Home"

def set_page(page_name):
    st.session_state.page = page_name

st.sidebar.markdown("## 🧭 Navigation")
nav_selection = st.sidebar.radio(
    "Select Section",
    ["Home", "Fake News Detection", "Model Insights", "About & Dataset"],
    index=["Home", "Fake News Detection", "Model Insights", "About & Dataset"].index(st.session_state.page)
)

if nav_selection != st.session_state.page:
    st.session_state.page = nav_selection

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎓 B.Tech Project Review")
st.sidebar.info(
    "**System:** Fake News Detection\n\n"
    "**Dataset:** WELFake (72,134 rows)\n\n"
    "**Best Model:** SVM (97.45% F1)\n\n"
    "**Split:** Stratified 80/20"
)


# ==============================================================================
# 1. HOME PAGE — Professional First Impression
# ==============================================================================
if st.session_state.page == "Home":
    st.markdown('<div class="main-header">FAKE NEWS DETECTION SYSTEM</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">NLP-Powered Machine Learning Classification for Automated Journalism Verification</div>', unsafe_allow_html=True)

    col_btn, col_blank = st.columns([1, 3])
    with col_btn:
        if st.button("🔍 Detect News Now"):
            set_page("Fake News Detection")
            st.rerun()

    st.markdown("---")

    st.markdown("### 📌 Problem & System Overview")
    st.write(
        "Misinformation and clickbait articles pose significant threats to public trust and digital media integrity. "
        "This project develops an automated text classification pipeline that standardizes raw headlines and articles using NLP, "
        "transforms text into high-dimensional TF-IDF feature vectors, and evaluates five competitive machine learning classifiers "
        "to accurately flag fake news."
    )

    st.markdown("### 🔄 End-to-End Project Workflow")
    workflow_box = """
    ```
    WELFake Dataset (72,134) ➔ Text Preprocessing ➔ TF-IDF (40k dims) ➔ 5 ML Models ➔ Model Evaluation ➔ Prediction
    ```
    """
    st.markdown(workflow_box)

    st.markdown("### 📊 Key Project Statistics")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Dataset Size</div>
            <div class="metric-value">72,134</div>
            <div style="font-size:0.75rem; color:#64748b; margin-top:2px;">WELFake Corpus</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Best Accuracy</div>
            <div class="metric-value">97.22%</div>
            <div style="font-size:0.75rem; color:#64748b; margin-top:2px;">Support Vector Machine</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Best F1-Score</div>
            <div class="metric-value">97.45%</div>
            <div style="font-size:0.75rem; color:#64748b; margin-top:2px;">Stratified Test Set</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Feature Dimension</div>
            <div class="metric-value">40,000</div>
            <div style="font-size:0.75rem; color:#64748b; margin-top:2px;">TF-IDF Unigrams & Bigrams</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 🛠️ Technology Stack")
    t1, t2, t3, t4, t5 = st.columns(5)
    with t1:
        st.markdown('<div class="tech-card">🐍 Python 3.10+</div>', unsafe_allow_html=True)
    with t2:
        st.markdown('<div class="tech-card">🐼 Pandas & NumPy</div>', unsafe_allow_html=True)
    with t3:
        st.markdown('<div class="tech-card">⚙️ Scikit-Learn</div>', unsafe_allow_html=True)
    with t4:
        st.markdown('<div class="tech-card">🔤 TF-IDF Vectorizer</div>', unsafe_allow_html=True)
    with t5:
        st.markdown('<div class="tech-card">👑 Streamlit App</div>', unsafe_allow_html=True)


# ==============================================================================
# 2. FAKE NEWS DETECTION — Main Feature
# ==============================================================================
elif st.session_state.page == "Fake News Detection":
    st.markdown('<div class="main-header">FAKE NEWS DETECTION</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Paste a news headline and article body below to analyze its authenticity in real time.</div>', unsafe_allow_html=True)

    predictor, err = get_cached_predictor()

    if err:
        st.error(
            f"⚠️ **Trained Model Artifacts Not Found!**\n\n"
            f"Details: {err}\n\n"
            f"Please run the training script first: `python train.py`"
        )
    else:
        # Sample News Presets for Viva Demonstration
        sample_options = {
            "Select a pre-loaded sample news story for live demo...": {"title": "", "text": ""},
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
            selected_sample = st.selectbox("💡 Quick Demo Preset:", list(sample_options.keys()))
        with col_model:
            model_options = {
                "Best Model (Support Vector Machine - 97.45% F1)": "models/best_model.pkl",
                "Logistic Regression": "models/logistic_regression.pkl",
                "Support Vector Machine (SVM)": "models/svm.pkl",
                "Random Forest": "models/random_forest.pkl",
                "Naive Bayes": "models/naive_bayes.pkl",
                "Gradient Boosting": "models/gradient_boosting.pkl"
            }
            chosen_model_label = st.selectbox("🤖 Choose Classifier Algorithm:", list(model_options.keys()))
            chosen_model_path = model_options[chosen_model_label]

        default_title = sample_options[selected_sample]["title"]
        default_text = sample_options[selected_sample]["text"]

        with st.form("prediction_form"):
            news_title = st.text_input(
                "News Headline / Title:",
                value=default_title,
                placeholder="e.g. U.S. Senate Approves Bipartisan Budget Resolution..."
            )
            news_article = st.text_area(
                "News Article Body:",
                value=default_text,
                height=180,
                placeholder="Paste full news article content here..."
            )

            col_btn, _ = st.columns([1, 4])
            with col_btn:
                submit_btn = st.form_submit_button("⚡ Analyze News")

        if submit_btn:
            # Dynamically load selected model if changed
            if chosen_model_path != predictor.model_path and os.path.exists(chosen_model_path):
                predictor.model_path = chosen_model_path
                predictor.model = joblib.load(chosen_model_path)

            with st.spinner("Executing NLP preprocessing, TF-IDF vectorization, and model inference..."):
                res = predictor.predict(news_title, news_article)
                if chosen_model_label.startswith("Best Model"):
                    res["model_used"] = "Support Vector Machine (Selected Best Model)"
                else:
                    res["model_used"] = chosen_model_label

            if not res["success"]:
                st.warning(f"⚠️ {res['error']}")
            else:
                verdict = res["prediction"]
                conf = res["confidence_percentage"]
                model_used = res["model_used"]

                st.markdown("### 📋 Prediction Results")

                if verdict == "REAL NEWS":
                    st.markdown(f"""
                    <div class="real-alert">
                        ✅ VERDICT: REAL / GENUINE NEWS
                        <div style="font-size: 0.95rem; font-weight: 500; margin-top: 4px;">
                            Model Confidence: <strong>{conf:.1f}%</strong> | Classifier: <strong>{model_used}</strong>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="fake-alert">
                        ⚠️ VERDICT: FAKE / MISINFORMATION NEWS
                        <div style="font-size: 0.95rem; font-weight: 500; margin-top: 4px;">
                            Model Confidence: <strong>{conf:.1f}%</strong> | Classifier: <strong>{model_used}</strong>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                c_m1, c_m2, c_m3 = st.columns(3)
                with c_m1:
                    st.metric("Predicted Verdict", verdict)
                with c_m2:
                    st.metric("Model Confidence Score", f"{conf:.1f}%")
                with c_m3:
                    st.metric("Classifier Name", model_used)

                st.progress(conf / 100.0)

                # Token Explainability
                st.markdown("#### 🧠 Model Token Explainability")
                st.write("Salient vocabulary terms extracted from the input text that contributed to this classification:")
                influential = res.get("influential_tokens", [])
                if influential:
                    token_cols = st.columns(min(len(influential), 6))
                    for idx, tok in enumerate(influential[:6]):
                        with token_cols[idx]:
                            is_real = "Real" in tok["impact"]
                            badge_color = "#166534" if is_real else "#991b1b"
                            bg_color = "#f0fdf4" if is_real else "#fef2f2"
                            st.markdown(f"""
                            <div style="background-color:{bg_color}; border: 1px solid {badge_color}; border-radius:6px; padding:6px; text-align:center;">
                                <div style="font-size:1rem; font-weight:700; color:{badge_color};">'{tok['word']}'</div>
                                <div style="font-size:0.7rem; color:#475569;">{tok['impact']}</div>
                            </div>
                            """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class="disclaimer-box">
                    <strong>Disclaimer:</strong> {res['disclaimer']}
                </div>
                """, unsafe_allow_html=True)


# ==============================================================================
# 3. MODEL INSIGHTS — Demonstrate the Technical Work
# ==============================================================================
elif st.session_state.page == "Model Insights":
    st.markdown('<div class="main-header">MODEL INSIGHTS & BENCHMARKS</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Empirical evaluation metrics and diagnostic visualizations across five machine learning algorithms.</div>', unsafe_allow_html=True)

    comp_df = load_comparison_data()
    summary = load_dataset_summary()

    if comp_df is not None:
        best_name = comp_df.iloc[0]["Model"]
        best_f1 = comp_df.iloc[0]["F1 Score"]
        best_acc = comp_df.iloc[0]["Accuracy"]

        st.success(
            f"🏆 **Selected Best Performing Model:** **{best_name}** "
            f"(F1-Score: **{best_f1*100:.2f}%** | Accuracy: **{best_acc*100:.2f}%**)"
        )

        st.markdown("### 📋 Model Comparison Table (Evaluated on 12,722 Unseen Test Samples)")
        st.dataframe(comp_df, use_container_width=True)

        st.markdown("---")
        st.markdown("### 📊 Evaluation Visualizations")

        tab1, tab2, tab3, tab4 = st.tabs([
            "1. Model Performance Chart",
            "2. Best Model Confusion Matrix",
            "3. ROC Curve",
            "4. Top Feature Explainability"
        ])

        with tab1:
            st.markdown("#### F1-Score & Accuracy Comparison Across Models")
            fig_comp, ax_comp = plt.subplots(figsize=(9, 4))
            comp_melted = comp_df.melt(id_vars=["Model"], value_vars=["F1 Score", "Accuracy"], var_name="Metric", value_name="Score")
            sns.barplot(data=comp_melted, x="Model", y="Score", hue="Metric", palette=["#1e3a8a", "#0d9488"], ax=ax_comp)
            ax_comp.set_ylim([0.75, 1.0])
            ax_comp.set_ylabel("Score", fontsize=11)
            ax_comp.set_xlabel("")
            plt.xticks(rotation=15, ha="right", fontsize=10)
            for p in ax_comp.patches:
                height = p.get_height()
                if not np.isnan(height) and height > 0:
                    ax_comp.annotate(f"{height*100:.1f}%", (p.get_x() + p.get_width() / 2., height),
                                     ha='center', va='bottom', fontsize=9, weight='bold', xytext=(0, 2),
                                     textcoords='offset points')
            plt.tight_layout()
            st.pyplot(fig_comp)

        with tab2:
            best_cm_path = "reports/confusion_matrix_best.png"
            if os.path.exists(best_cm_path):
                st.image(best_cm_path, caption=f"Confusion Matrix for Best Model ({best_name})", width=650)
            else:
                st.info("Run `python train.py` to generate confusion matrix visualization.")

        with tab3:
            roc_path = "reports/roc_curve.png"
            if os.path.exists(roc_path):
                st.image(roc_path, caption="Combined Receiver Operating Characteristic (ROC) Comparison", use_container_width=True)
            else:
                st.info("Run `python train.py` to generate ROC curves.")

        with tab4:
            feat_path = "reports/feature_importance.png"
            if os.path.exists(feat_path):
                st.image(feat_path, caption="Top Salient Vocabulary Tokens Influencing Classification", use_container_width=True)
            else:
                st.info("Run `python train.py` to generate feature explainability plots.")
    else:
        st.warning("Evaluation report not found. Run `python train.py` to execute evaluation pipeline.")


# ==============================================================================
# 4. ABOUT & DATASET — One Compact Section
# ==============================================================================
elif st.session_state.page == "About & Dataset":
    st.markdown('<div class="main-header">ABOUT THE PROJECT & DATASET</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Technical specs, dataset overview, and viva reference documentation.</div>', unsafe_allow_html=True)

    with st.expander("📌 Problem Statement & Objectives", expanded=True):
        st.markdown("""
        - **Problem:** The rapid proliferation of digital misinformation and clickbait requires automated, interpretable text verification.
        - **Objectives:** Build an end-to-end Machine Learning pipeline to preprocess news text, extract TF-IDF n-gram features, evaluate 5 classification algorithms, and select the optimal model.
        """)

    with st.expander("📊 WELFake Dataset Specifications", expanded=True):
        col_info, col_pie = st.columns([3, 2])
        with col_info:
            st.markdown("""
            - **Dataset:** WELFake – Fake and Real News Dataset
            - **Source:** Kaggle ([Fake News Classification Dataset](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification))
            - **Total Articles:** 72,134 records (63,606 clean non-duplicate records)
            - **Target Variable (`label`):**
                - `0` = Fake News (28,815 clean records / 37,106 raw)
                - `1` = Real News (34,791 clean records / 35,028 raw)
            - **Attributes:** `Serial Number`, `Title`, `Text`, `Label`
            """)
        with col_pie:
            fig_p, ax_p = plt.subplots(figsize=(4, 4))
            ax_p.pie([37106, 35028], labels=["Fake (0)\n37,106", "Real (1)\n35,028"], autopct="%1.1f%%",
                     colors=["#dc2626", "#16a34a"], startangle=140, textprops={'weight': 'bold', 'fontsize': 10})
            ax_p.axis("equal")
            st.pyplot(fig_p)

    with st.expander("⚙️ NLP Preprocessing & Feature Engineering"):
        st.markdown("""
        1. **Cleaning:** Lowercasing, HTML/URL stripping, regex punctuation removal.
        2. **Normalization:** NLTK English stop-words filtering and Porter Stemmer with LRU caching.
        3. **TF-IDF Vectorization:** `TfidfVectorizer(max_features=40000, ngram_range=(1,2), sublinear_tf=True)` fit strictly on training data.
        """)

    with st.expander("🤖 Machine Learning Algorithms Evaluated"):
        st.markdown("""
        1. **Logistic Regression:** Linear decision boundary with L-BFGS solver.
        2. **Multinomial Naive Bayes:** Probabilistic classifier with Laplace smoothing ($\alpha=0.1$).
        3. **Support Vector Machine (LinearSVC):** Convex max-margin hyperplane calibrated via `CalibratedClassifierCV`.
        4. **Random Forest:** Bagged ensemble of decorrelated decision trees (`max_depth=35`).
        5. **Gradient Boosting:** Sequential gradient-boosted trees (`n_estimators=50, max_depth=3`).
        """)

    with st.expander("⚠️ Project Limitations & Future Enhancements"):
        st.markdown("""
        - **Limitations:** Predictions reflect vocabulary correlations in the WELFake corpus; text-only models cannot verify real-world facts dynamically.
        - **Future Scope:** Integration with Google Fact Check API, fine-tuned BERT/RoBERTa transformers, and multi-modal image verification.
        """)
