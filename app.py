
import streamlit as st
import joblib
import numpy as np

from pathlib import Path
from src.text_utils import preprocess_text

# Step 1: Configure the web page
st.set_page_config(
    page_title="MailGuard AI",
    page_icon="🛡️",
    layout="centered"
)

# Step 2: Load our trained model
@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "models" / "final_best_model.pkl"
    return joblib.load(model_path)

model = load_model()

# Step 3: Application heading
st.title("🛡️ MailGuard AI")

st.subheader("Email Spam Detection System")

st.write(
    "Detect spam emails using Machine Learning "
    "and Natural Language Processing."
)

st.divider()

# Step 4: Email input
st.subheader("📧 Enter Email Content")

email_text = st.text_area(
    "Paste your email below:",
    height=200,
    placeholder="Enter or paste an email message..."
)

# Step 5: Analyze button
if st.button(
    "🔍 Analyze Email",
    type="primary",
    use_container_width=True
):

    # Check for empty input
    if not email_text.strip():
        st.warning("Please enter an email first.")

    else:
        # Preprocess email
        cleaned_email = preprocess_text(email_text)

        if not cleaned_email:
            st.warning("Please enter an email containing text.")

        else:
            # Predict using SVM
            prediction = model.predict([cleaned_email])[0]

            st.divider()

            # Display prediction
            if prediction == 1:
                st.error("🚨 SPAM DETECTED")

                st.write(
                    "This email has been classified as spam "
                    "by our machine-learning model."
                )

            else:
                st.success("✅ LEGITIMATE EMAIL (HAM)")

                st.write(
                    "The model classified this email as HAM "
                    "(not spam). However, this prediction does "
                    "not guarantee that the email is safe."
               )

                st.caption(
                    "Never share passwords or verification codes "
                    "through untrusted emails. Verify suspicious "
                    "requests through official channels."
                )

# Step 6: Information section
st.divider()

st.subheader("ℹ️ About This Project")

st.write(
    "This project uses Natural Language Processing (NLP), "
    "TF-IDF feature extraction, and a trained "
    "Support Vector Machine (SVM) classifier."
)

st.info(
    "Model: Support Vector Machine (SVM) | "
    "Revised held-out test accuracy: 98.53% | "
    "Spam F1-score: 97.51%"
)

st.caption(
    "Predictions are automated and may be incorrect. "
    "This system does not independently verify links "
    "or detect every phishing attempt."
)


# Model Comparison Dashboard

import pandas as pd

st.divider()

st.header("📊 Initial Model Comparison")

st.write(
    "We trained and evaluated three classification "
    "algorithms using TF-IDF features."
)

# Load comparison results
comparison_path = (
    Path(__file__).parent
    / "models"
    / "model_comparison.csv"
)

results = pd.read_csv(comparison_path)

# Display model performance table
st.subheader("Performance Metrics")

display_results = results.copy()

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score"
]

display_results[metrics] = (
    display_results[metrics] * 100
).round(2)

st.dataframe(
    display_results,
    hide_index=True,
    use_container_width=True
)

# Compare model accuracy
st.subheader("Accuracy Comparison")

st.bar_chart(
    display_results.set_index("Model")["Accuracy"],
    y_label="Accuracy (%)",
    use_container_width=True
)

# Compare F1 scores
st.subheader("F1-Score Comparison")

st.bar_chart(
    display_results.set_index("Model")["F1 Score"],
    y_label="F1 Score (%)",
    use_container_width=True
)

# Display best-performing model by F1 score
best_model = results.loc[
    results["F1 Score"].idxmax()
]

st.success(
    f"🏆 Best Model by F1-Score: "
    f"{best_model['Model']}"
)

st.write(
    f"F1-Score: {best_model['F1 Score'] * 100:.2f}%"
)

st.caption(
    "These metrics come from our development comparison. "
    "A separate final evaluation is still recommended."
)


# Confusion Matrix Visualization

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay
)

st.divider()

st.header("🧪 Initial Model Evaluation")

st.write(
    "Select a machine learning model to view "
    "its confusion matrix and classification errors."
)

# Step 1: Create a model selection dropdown
selected_model = st.selectbox(
    "Choose a model:",
    ["Naive Bayes", "Logistic Regression", "SVM"]
)

# Step 2: Map model names to saved files
model_files = {
    "Naive Bayes": "naive_bayes.pkl",
    "Logistic Regression": "logistic_regression.pkl",
    "SVM": "svm.pkl"
}

# Step 3: Load the selected trained model
selected_path = (
    Path(__file__).parent
    / "models"
    / model_files[selected_model]
)

selected_pipeline = joblib.load(selected_path)

# Step 4: Load the processed dataset
dataset_path = (
    Path(__file__).parent
    / "data"
    / "processed_emails.csv"
)

df_eval = pd.read_csv(dataset_path)

X = df_eval["cleaned_text"].fillna("")
y = df_eval["label_num"]

# Step 5: Recreate the same development test split
_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Step 6: Predict testing emails
predictions = selected_pipeline.predict(X_test)

# Step 7: Calculate confusion matrix
cm = confusion_matrix(
    y_test,
    predictions,
    labels=[0, 1]
)

# Step 8: Create confusion matrix visualization
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

ax.set_title(f"{selected_model} - Confusion Matrix")

st.pyplot(fig)
plt.close(fig)

# Step 9: Display classification errors
false_positives = cm[0, 1]
false_negatives = cm[1, 0]

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "False Positives",
        int(false_positives)
    )

with col2:
    st.metric(
        "False Negatives",
        int(false_negatives)
    )


# Final Model Evaluation Dashboard

import json

st.divider()

st.header("🏆 Final Model Evaluation")

# Load saved final results
results_path = (
    Path(__file__).parent
    / "models"
    / "final_results.json"
)

with open(results_path, "r", encoding="utf-8") as file:
    final_results = json.load(file)

st.write(
    "This section presents the results of our "
    "revised 70/15/15 evaluation procedure."
)

st.subheader(
    f"Selected Model: {final_results['model']}"
)

# Display performance metrics
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
        "Spam F1-Score",
        f"{final_results['f1_score'] * 100:.2f}%"
    )

# Display final confusion matrix
st.subheader("Final Test Confusion Matrix")

cm_final = np.array(final_results["confusion_matrix"])

fig_final, ax_final = plt.subplots(figsize=(6, 4))

display_final = ConfusionMatrixDisplay(
    confusion_matrix=cm_final,
    display_labels=["Ham", "Spam"]
)

display_final.plot(
    ax=ax_final,
    cmap="Blues",
    values_format="d",
    colorbar=False
)

ax_final.set_title("Final SVM Confusion Matrix")

st.pyplot(fig_final)
plt.close(fig_final)

# Display classification errors
fp = cm_final[0, 1]
fn = cm_final[1, 0]

col1, col2 = st.columns(2)

with col1:
    st.metric("False Positives", fp)

with col2:
    st.metric("False Negatives", fn)

st.caption(
    "Results from the revised held-out evaluation. "
    "Earlier experiments used a different split "
    "of the same dataset."
)
