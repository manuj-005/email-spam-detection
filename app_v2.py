
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import ConfusionMatrixDisplay
from src.text_utils import preprocess_text

# ========================================
# 1. APPLICATION CONFIGURATION
# ========================================

st.set_page_config(
    page_title="MailGuard AI",
    page_icon="🛡️",
    layout="wide"
)

BASE_DIR = Path(__file__).parent

# ========================================
# 2. LOAD MODEL AND RESULTS
# ========================================

@st.cache_resource
def load_model():
    return joblib.load(
        BASE_DIR / "models" / "final_best_model.pkl"
    )


@st.cache_data
def load_comparison():
    return pd.read_csv(
        BASE_DIR / "models" / "model_comparison.csv"
    )


@st.cache_data
def load_final_results():
    with open(
        BASE_DIR / "models" / "final_results.json",
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


model = load_model()
comparison = load_comparison()
final_results = load_final_results()

# ========================================
# 3. HEADER
# ========================================

st.title("🛡️ MailGuard AI")

st.write(
    "### Email Spam Detection Using Machine "
    "Learning and Natural Language Processing"
)

st.caption(
    "Built using Python, NLP, TF-IDF, "
    "Scikit-learn and Streamlit."
)

st.divider()

# ========================================
# 4. CREATE NAVIGATION TABS
# ========================================

tab1, tab2, tab3 = st.tabs([
    "📧 Email Analyzer",
    "📊 Model Comparison",
    "🧪 Evaluation Dashboard"
])

# ========================================
# TAB 1: EMAIL ANALYZER
# ========================================

with tab1:

    st.header("Email Spam Analyzer")

    st.write(
        "Paste an email to classify it "
        "as Spam or Ham using our trained SVM model."
    )

    with st.form("email_form"):

        email_text = st.text_area(
            "Email Content",
            height=220,
            placeholder="Enter or paste an email..."
        )

        analyze = st.form_submit_button(
            "🔍 Analyze Email",
            type="primary",
            width="stretch"        )

    if analyze:

        if not email_text.strip():

            st.warning("Please enter an email first.")

        else:

            cleaned = preprocess_text(email_text)

            if not cleaned:

                st.warning(
                    "Please enter an email containing text."
                )

            else:

                prediction = model.predict([cleaned])[0]

                if prediction == 1:

                    st.error("🚨 SPAM DETECTED")

                    st.write(
                        "The model classified this "
                        "email as spam."
                    )

                else:

                    st.success("✅ HAM DETECTED")

                    st.write(
                        "The model classified this email "
                        "as not spam. This does not "
                        "guarantee that it is safe."
                    )

                st.info(
                    "This is an ML prediction, not "
                    "independent verification of sender "
                    "identity, URLs, or attachments."
                )

    st.divider()

    st.subheader("About the Classification System")

    st.write(
        "The system cleans incoming text, "
        "converts it to TF-IDF features, "
        "and classifies it using SVM."
    )

    st.metric(
        "Revised Held-out Test Accuracy",
        f"{final_results['accuracy'] * 100:.2f}%"
    )

# ========================================
# TAB 2: MODEL COMPARISON
# ========================================

with tab2:

    st.header("Machine Learning Model Comparison")

    st.write(
        "Performance of three classification "
        "algorithms in our initial development experiment."
    )

    metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]

    display_df = comparison.copy()

    display_df[metrics] = (
        display_df[metrics] * 100
    ).round(2)

    st.subheader("Performance Metrics (%)")

    st.dataframe(
        display_df,
        hide_index=True,
        width="stretch"
    )

    st.subheader("Accuracy Comparison")

    st.bar_chart(
        display_df.set_index("Model")["Accuracy"],
        y_label="Accuracy (%)"
    )

    st.subheader("F1-Score Comparison")

    st.bar_chart(
        display_df.set_index("Model")["F1 Score"],
        y_label="F1 Score (%)"
    )

    best = comparison.loc[
        comparison["F1 Score"].idxmax()
    ]

    st.success(
        f"Best model in initial comparison: "
        f"{best['Model']}"
    )

    st.caption(
        "This section uses our original 80/20 "
        "development experiment."
    )

# ========================================
# TAB 3: FINAL EVALUATION
# ========================================

with tab3:

    st.header("Final Model Evaluation")

    st.write(
        "Results from the revised "
        "70/15/15 evaluation procedure."
    )

    st.subheader(
        f"Selected Model: {final_results['model']}"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Accuracy",
            f"{final_results['accuracy'] * 100:.2f}%"
        )

        st.metric(
            "Spam Recall",
            f"{final_results['recall'] * 100:.2f}%"
        )

    with col2:

        st.metric(
            "Spam Precision",
            f"{final_results['precision'] * 100:.2f}%"
        )

        st.metric(
            "Spam F1-score",
            f"{final_results['f1_score'] * 100:.2f}%"
        )

    st.subheader("Final Confusion Matrix")

    cm = np.array(
        final_results["confusion_matrix"]
    )

    fig, ax = plt.subplots(figsize=(6, 4))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Ham", "Spam"]
    )

    display.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        colorbar=False
    )

    ax.set_title("Final SVM Confusion Matrix")

    st.pyplot(fig)
    plt.close(fig)

    col1, col2 = st.columns(2)

    with col1:
        st.metric("False Positives", int(cm[0, 1]))

    with col2:
        st.metric("False Negatives", int(cm[1, 0]))

    st.caption(
        "The held-out test results come from the "
        "revised evaluation procedure. Earlier "
        "experiments used another split of the "
        "same dataset."
    )

# ========================================
# 5. FOOTER
# ========================================

st.divider()

st.caption(
    "MailGuard AI | Academic Machine Learning Project "
    "| Predictions may be incorrect."
)
