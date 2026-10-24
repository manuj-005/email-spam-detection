
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

st.markdown("### 🛡️ MailGuard AI")

st.write(
    "### Email Spam Detection Using Machine "
    "Learning and Natural Language Processing"
)

st.caption(
    "Built using Python, NLP, TF-IDF, "
    "Scikit-learn and Streamlit."
)

st.divider()


# =========================================
# V4 SIDEBAR NAVIGATION
# =========================================

with st.sidebar:

    st.title("🛡️ MailGuard AI")

    st.caption("Intelligent Email Security")

    st.divider()

    st.subheader("Navigation")
    # Short labels for cleaner sidebar navigation
    short_labels = {
        "📧 Email Analyzer": "📧 Analyzer",
        "📊 Model Comparison": "📊 Models",
        "🧪 Evaluation Dashboard": "🧪 Evaluation",
        "🌍 External Testing": "🌍 External Tests"
    }

    page = st.radio(
         "Choose a section",
        list(short_labels.keys()),
        format_func=lambda option: short_labels[option],
        label_visibility="collapsed"
    )

    st.divider()

    st.caption("MODEL INFORMATION")

    st.write("**Classifier:** Linear SVM")

    st.write(
        f"**Held-out Accuracy:** "
        f"{final_results['accuracy'] * 100:.2f}%"
    )

    st.caption(
        "Predictions are automated and "
        "may be incorrect."
    )

# ========================================
# TAB 1: EMAIL ANALYZER
# ========================================

if page == "📧 Email Analyzer":

    st.header("Email Spam Analyzer")

    st.write(
        "Paste an email to classify it "
        "as Spam or Ham using our trained SVM model."
    )

    with st.form("email_form"):

        email_text = st.text_area(
            "Email Content",
            height=150,
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
                    expanded=False
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

                            st.bar_chart(
    chart_data,
    horizontal=True,
    height=230,
    width="stretch"
)

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

                            st.bar_chart(
    chart_data,
    horizontal=True,
    height=230,
    width="stretch"
)

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

elif page == "📊 Model Comparison":

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

elif page == "🧪 Evaluation Dashboard":

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

    
    # =====================================
    # COMPACT FINAL MODEL EVALUATION
    # =====================================

    st.subheader("Final Test Confusion Matrix")

    final_cm = np.array(
        final_results["confusion_matrix"]
    )

    matrix_col, stats_col = st.columns(
        [1.15, 1],
        gap="large"
    )

    with matrix_col:

        fig_final, ax_final = plt.subplots(
            figsize=(5, 3.5)
        )

        display_final = ConfusionMatrixDisplay(
            confusion_matrix=final_cm,
            display_labels=["Ham", "Spam"]
        )

        display_final.plot(
            ax=ax_final,
            cmap="Blues",
            values_format="d",
            colorbar=False
        )

        ax_final.set_title(
            "Final SVM Confusion Matrix",
            fontsize=11
        )

        fig_final.tight_layout()

        st.pyplot(fig_final, width=500)

        plt.close(fig_final)

    with stats_col:

        st.markdown("#### Classification Errors")

        fp = int(final_cm[0, 1])
        fn = int(final_cm[1, 0])

        ham_total = int(final_cm[0].sum())

        fpr = (
            fp / ham_total * 100
            if ham_total > 0 else 0
        )

        st.metric("False Positives", fp)

        st.metric("False Negatives", fn)

        st.metric(
            "False Positive Rate",
            f"{fpr:.2f}%"
        )

    st.caption(
        "Results from the revised 70/15/15 evaluation. "
        "The held-out test set contains 749 emails."
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
    width=570
    )
    st.info(
        "The tuned threshold is experimental and "
        "was selected using validation data. "
        "The Email Analyzer continues using the "
        "saved final SVM model with its default threshold."
    )


# =========================================
# TAB 4: EXTERNAL DATASET EVALUATION
# =========================================

elif page == "🌍 External Testing":

    st.header("🌍 External Dataset Generalization")

    st.write(
        "This section evaluates the same trained "
        "SVM model on emails from different datasets "
        "without additional training."
    )

    st.info(
        "The purpose is to measure how well our "
        "classifier generalizes beyond its "
        "original training source."
    )

    # -------------------------------------
    # 1. Build dataset comparison table
    # -------------------------------------

    st.subheader("Dataset Performance Comparison")

    rows = []

    # Original held-out test results
    original_cm = np.array(
        final_results["confusion_matrix"]
    )

    original_fpr = (
        original_cm[0, 1] / original_cm[0].sum()
    ) * 100

    rows.append({
        "Dataset": "Original Held-out Test",
        "Accuracy (%)": final_results["accuracy"] * 100,
        "Spam F1 (%)": final_results["f1_score"] * 100,
        "False Positive Rate (%)": original_fpr
    })

    # Read the external dataset results
    for name, result in external_results.items():

        metrics = result["metrics"]

        cm = np.array(metrics["confusion_matrix"])

        fpr = (cm[0, 1] / cm[0].sum()) * 100

        rows.append({
            "Dataset": name,
            "Accuracy (%)": metrics["accuracy"] * 100,
            "Spam F1 (%)": metrics["spam_f1"] * 100,
            "False Positive Rate (%)": fpr
        })

    comparison_df = pd.DataFrame(rows)

    numeric_columns = [
        "Accuracy (%)",
        "Spam F1 (%)",
        "False Positive Rate (%)"
    ]

    comparison_df[numeric_columns] = (
        comparison_df[numeric_columns].round(2)
    )

    st.dataframe(
        comparison_df,
        hide_index=True,
        width="stretch"
    )

    # -------------------------------------
    # 2. Comparison visualization
    # -------------------------------------

    st.subheader("Accuracy and F1-Score Comparison")

    chart_data = comparison_df.set_index("Dataset")[
        ["Accuracy (%)", "Spam F1 (%)"]
    ]

   
    st.bar_chart(
    chart_data,
    horizontal=True,
    stack=False,
    sort=False,
    x_label="Performance (%)",
    height=300,
    width="stretch"
    )


    st.caption(
        "The original held-out test uses the "
        "training source. External datasets have "
        "different vocabulary and class distributions."
    )

    # -------------------------------------
    # 3. Individual dataset analysis
    # -------------------------------------

    st.divider()

    st.subheader("Inspect an External Dataset")

    selected_dataset = st.selectbox(
        "Choose a dataset:",
        list(external_results.keys()),
        key="external_dataset_selector"
    )

    selected_results = external_results[selected_dataset]

    metrics = selected_results["metrics"]
    quality = selected_results["quality_checks"]

    cm = np.array(metrics["confusion_matrix"])

    st.write(f"### {selected_dataset}")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Emails Evaluated",
            f"{quality['evaluated_rows']:,}"
        )

    with col2:
        st.metric(
            "Accuracy",
            f"{metrics['accuracy'] * 100:.2f}%"
        )

    with col3:
        st.metric(
            "Balanced Accuracy",
            f"{metrics['balanced_accuracy'] * 100:.2f}%"
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Spam Precision",
            f"{metrics['spam_precision'] * 100:.2f}%"
        )

    with col2:
        st.metric(
            "Spam Recall",
            f"{metrics['spam_recall'] * 100:.2f}%"
        )

    with col3:
        st.metric(
            "Spam F1-Score",
            f"{metrics['spam_f1'] * 100:.2f}%"
        )


    
    
    # =====================================
    # COMPACT EXTERNAL CONFUSION MATRIX
    # =====================================

    st.subheader("External Confusion Matrix")

    matrix_col, stats_col = st.columns(
        [1.15, 1],
        gap="large"
    )

    with matrix_col:

        fig_ext, ax_ext = plt.subplots(
            figsize=(5, 3.5)
        )

        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Ham", "Spam"]
        )

        display.plot(
            ax=ax_ext,
            cmap="Blues",
            values_format="d",
            colorbar=False
        )

        ax_ext.set_title(
            f"{selected_dataset} - Confusion Matrix",
            fontsize=11
        )

        fig_ext.tight_layout()

        st.pyplot(
            fig_ext,
            width=500
        )

        plt.close(fig_ext)

    with stats_col:

        st.markdown("#### Classification Errors")

        false_positives = int(cm[0, 1])
        false_negatives = int(cm[1, 0])

        ham_total = int(cm[0].sum())

        false_positive_rate = (
            false_positives / ham_total * 100
            if ham_total > 0 else 0
        )

        st.metric(
            "False Positives",
            f"{false_positives:,}"
        )

        st.metric(
            "False Negatives",
            f"{false_negatives:,}"
        )

        st.metric(
            "False Positive Rate",
            f"{false_positive_rate:.2f}%"
        )

    


   
    # -------------------------------------
    # 6. Data quality information
    # -------------------------------------

    with st.expander("View Dataset Quality Checks"):

        quality_df = pd.DataFrame(
            list(quality.items()),
            columns=["Check", "Count"]
        )

        st.dataframe(
            quality_df,
            hide_index=True,
            width="stretch"
        )

    # -------------------------------------
    # 7. Research interpretation
    # -------------------------------------

    st.divider()

    st.subheader("Research Findings")

    st.warning(
        "The SVM performs well on Enron-related "
        "emails but has substantially lower "
        "accuracy on SpamAssassin and LingSpam. "
        "False positives are particularly common "
        "on these external sources."
    )

    st.write(
        "These results suggest dataset-specific "
        "learning and possible distribution shift. "
        "Differences in vocabulary, preprocessing, "
        "and email formatting require further study."
    )

    st.caption(
        "The Enron subset is a related-source evaluation, "
        "not fully independent evidence. Exact duplicates "
        "were removed, but near-duplicate overlap and "
        "dataset provenance remain limitations. "
        "No external dataset was used to retrain "
        "the saved SVM."
    )

# ========================================
# 5. FOOTER
# ========================================

st.divider()

st.caption(
    "MailGuard AI | Academic Machine Learning Project "
    "| Predictions may be incorrect."
)
