
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import ConfusionMatrixDisplay
from src.text_utils import preprocess_text
from src.explainability import explain_prediction

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
        BASE_DIR / "models" / "validation_comparison.csv"
    )


@st.cache_data
def load_cv_results():
    return pd.read_csv(
        BASE_DIR / "models" / "cross_validation_results.csv"
    )


@st.cache_data
def load_external_results():
    results_path = (
        BASE_DIR / "models" / "external_csv_results.json"
    )

    with open(results_path, "r", encoding="utf-8") as file:
        return json.load(file)




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
cv_results = load_cv_results()
external_results = load_external_results()

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
                
                # Explainable AI section
                st.divider()

                with st.expander(
                    "🧠 Why did the model make this prediction?",
                    expanded=True
                ):

                    spam_words, ham_words, score = (
                        explain_prediction(model, cleaned)
                    )

                    st.write(
                        "This explanation shows how words "
                        "in the email contributed to the "
                        "Linear SVM decision."
                    )

                    st.metric(
                        "SVM Decision Score",
                        f"{score:.4f}"
                    )

                    st.caption(
                        "Positive scores favor Spam. "
                        "Negative scores favor Ham. "
                        "This is not a probability."
                    )

                    col1, col2 = st.columns(2)

                    with col1:

                        st.subheader("Spam Contributions")

                        if not spam_words.empty:

                            chart_data = (
                                spam_words
                                .set_index("Word")["Contribution"]
                            )

                            st.bar_chart(chart_data)

                        else:
                            st.info(
                                "No positive word "
                                "contributions found."
                            )

                    with col2:

                        st.subheader("Ham Contributions")

                        if not ham_words.empty:

                            chart_data = (
                                ham_words
                                .assign(
                                    Magnitude=lambda d:
                                    -d["Contribution"]
                                )
                                .set_index("Word")["Magnitude"]
                            )

                            st.bar_chart(chart_data)

                            st.caption(
                                "Bars show absolute magnitudes "
                                "of negative contributions."
                            )

                        else:
                            st.info(
                                "No negative word "
                                "contributions found."
                            )

                    st.caption(
                        "Feature contributions explain "
                        "the model's mathematical decision, "
                        "not whether an email is actually safe."
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
        "Comparison of Naive Bayes, Logistic Regression "
    "and SVM on the validation portion of our "
    "revised 70/15/15 evaluation."
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
         "All three models used the same training and "
    "validation split. SVM was selected using "
    "validation F1-score."
    )

    
    # =====================================
    # STRATIFIED CROSS-VALIDATION RESULTS
    # =====================================

    st.divider()

    st.subheader("5-Fold Cross-Validation")

    st.write(
        "We evaluated all three models using "
        "the same five stratified folds on the "
        "development dataset."
    )

    # Create a readable comparison table
    cv_display = pd.DataFrame()

    cv_display["Model"] = cv_results["Model"]

    for metric in [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]:

        cv_display[metric] = [
            f"{mean * 100:.2f}% ± {std * 100:.2f}%"
            for mean, std in zip(
                cv_results[metric + " Mean"],
                cv_results[metric + " Std"]
            )
        ]

    st.dataframe(
        cv_display,
        hide_index=True,
        width="stretch"
    )

    st.caption(
        "Values represent mean ± sample standard "
        "deviation across five folds, not "
        "95% confidence intervals."
    )

    # Visualize average F1-score
    st.subheader("Mean Cross-Validation F1-Score")

    f1_chart = cv_results.set_index("Model")[
        "F1 Score Mean"
    ] * 100

    st.bar_chart(
        f1_chart,
        y_label="Mean F1-score (%)"
    )

    # Identify the best CV model
    best_cv = cv_results.loc[
        cv_results["F1 Score Mean"].idxmax()
    ]

    st.success(
        f"Best model by mean CV F1-score: "
        f"{best_cv['Model']}"
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
    
    # =====================================
    # VALIDATION THRESHOLD EXPERIMENT
    # =====================================

    st.divider()

    st.header("Decision Threshold Optimization")

    st.write(
        "We evaluated an experimental Linear SVM "
        "using different decision thresholds on "
        "the validation dataset."
    )

    threshold_file = (
        BASE_DIR / "models" / "threshold_experiment.json"
    )

    with open(
        threshold_file, "r", encoding="utf-8"
    ) as file:
        threshold_results = json.load(file)

    selected_threshold = threshold_results[
        "selected_threshold"
    ]

    default_cm = np.array(
        threshold_results["default_confusion_matrix"]
    )

    tuned_cm = np.array(
        threshold_results["tuned_confusion_matrix"]
    )

    st.metric(
        "Selected Experimental Threshold",
        f"{selected_threshold:.4f}"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Default Threshold")

        st.metric(
            "False Positives",
            int(default_cm[0, 1])
        )

        st.metric(
            "False Negatives",
            int(default_cm[1, 0])
        )

    with col2:
        st.subheader("Tuned Threshold")

        st.metric(
            "False Positives",
            int(tuned_cm[0, 1])
        )

        st.metric(
            "False Negatives",
            int(tuned_cm[1, 0])
        )

    st.subheader("Precision–Recall Curve")

    curve_path = (
        BASE_DIR / "models" / "threshold_pr_curve.png"
    )

    st.image(
        str(curve_path),
        caption="Precision–Recall curve on validation data",
        width="stretch"
    )

    st.info(
        "The tuned threshold is experimental and "
        "was selected using validation data. "
        "The Email Analyzer continues using the "
        "saved final SVM model with its default threshold."
    )


# ========================================
# 5. FOOTER
# ========================================

st.divider()

st.caption(
    "MailGuard AI | Academic Machine Learning Project "
    "| Predictions may be incorrect."
)
