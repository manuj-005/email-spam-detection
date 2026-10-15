
import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

# Import our three ML algorithms
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# Import evaluation metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Step 1: Load the cleaned dataset
df = pd.read_csv("data/processed_emails.csv")

X = df["cleaned_text"].fillna("")
y = df["label_num"]

# Step 2: Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training emails:", len(X_train))
print("Testing emails:", len(X_test))

# Step 3: Define the ML algorithms
algorithms = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),
    "SVM": LinearSVC(
        max_iter=5000
    )
}

# Step 4: Create folder for saved models
Path("models").mkdir(exist_ok=True)

results = []

# Step 5: Train and evaluate each model
for name, algorithm in algorithms.items():

    print(f"\n===== Training {name} =====")

    # Create TF-IDF + ML pipeline
    model = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000)),
        ("classifier", algorithm)
    ])

    # Train on training emails
    model.fit(X_train, y_train)

    # Predict on testing emails
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Display performance
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save metrics for comparison
    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    # Save trained pipeline
    filename = name.lower().replace(" ", "_")
    joblib.dump(
        model,
        f"models/{filename}.pkl"
    )

    print(f"{name} model saved successfully!")

# Step 6: Compare all three models
results_df = pd.DataFrame(results)

print("\n===== MODEL COMPARISON =====")
print(results_df.to_string(index=False))

# Step 7: Save comparison results
results_df.to_csv(
    "models/model_comparison.csv",
    index=False
)

print("\nComparison results saved!")

# Step 8: Identify model with highest F1-score
best_model = results_df.loc[
    results_df["F1 Score"].idxmax()
]

print("\n===== BEST MODEL BY F1-SCORE =====")
print("Model:", best_model["Model"])
print("F1 Score:", round(best_model["F1 Score"], 4))
