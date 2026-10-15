
import pandas as pd
import joblib
import json

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# 1. Load cleaned dataset
df = pd.read_csv("data/processed_emails.csv")

X = df["cleaned_text"].fillna("")
y = df["label_num"]

# 2. Reserve 15% for final testing
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y,
    test_size=0.15,
    random_state=42,
    stratify=y
)

# 3. Split remaining data into training and validation
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.15 / 0.85,
    random_state=42,
    stratify=y_temp
)

print("===== DATASET SPLIT =====")
print("Training:", len(X_train))
print("Validation:", len(X_val))
print("Testing:", len(X_test))

# 4. Define model constructors
algorithms = {
    "Naive Bayes": lambda: MultinomialNB(),
    "Logistic Regression": lambda: LogisticRegression(
        max_iter=1000
    ),
    "SVM": lambda: LinearSVC(
        max_iter=5000
    )
}

# Function to build a fresh model pipeline
def build_pipeline(name):
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000)),
        ("classifier", algorithms[name]())
    ])

# 5. Compare models using validation data
results = []

for name in algorithms:

    print(f"\nTraining: {name}")

    model = build_pipeline(name)

    model.fit(X_train, y_train)

    predictions = model.predict(X_val)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_val, predictions),
        "Precision": precision_score(
            y_val, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_val, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y_val, predictions, zero_division=0
        )
    })

# 6. Display validation comparison
results_df = pd.DataFrame(results)

print("\n===== VALIDATION COMPARISON =====")
print(results_df.to_string(index=False))

# 7. Select best model by validation F1-score
best_name = results_df.loc[
    results_df["F1 Score"].idxmax(), "Model"
]

print("\nSelected model:", best_name)

# 8. Combine training and validation data
X_train_final = pd.concat([X_train, X_val])
y_train_final = pd.concat([y_train, y_val])

# 9. Train a fresh selected model
final_model = build_pipeline(best_name)

final_model.fit(
    X_train_final,
    y_train_final
)

# 10. Evaluate on untouched final test data
final_predictions = final_model.predict(X_test)

print("\n===== FINAL TEST RESULTS =====")

print(
    "Accuracy:",
    accuracy_score(y_test, final_predictions)
)
print(
    "Precision:",
    precision_score(
        y_test, final_predictions, zero_division=0
    )
)
print(
    "Recall:",
    recall_score(
        y_test, final_predictions, zero_division=0
    )
)
print(
    "F1 Score:",
    f1_score(
        y_test, final_predictions, zero_division=0
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(
    y_test,
    final_predictions,
    labels=[0, 1]
))

print("\nClassification Report:")
print(classification_report(
    y_test,
    final_predictions,
    labels=[0, 1],
    target_names=["Ham", "Spam"],
    zero_division=0
))

# 11. Save final model and validation comparison
Path("models").mkdir(exist_ok=True)

joblib.dump(
    final_model,
    "models/final_best_model.pkl"
)

results_df.to_csv(
    "models/validation_comparison.csv",
    index=False
)

print("\nFinal model saved successfully!")
print("Selected algorithm:", best_name)



# Save final evaluation results to JSON

final_results = {
    "model": best_name,
    "test_samples": len(X_test),
    "accuracy": accuracy_score(
        y_test, final_predictions
    ),
    "precision": precision_score(
        y_test, final_predictions,
        zero_division=0
    ),
    "recall": recall_score(
        y_test, final_predictions,
        zero_division=0
    ),
    "f1_score": f1_score(
        y_test, final_predictions,
        zero_division=0
    ),
    "confusion_matrix": confusion_matrix(
        y_test,
        final_predictions,
        labels=[0, 1]
    ).tolist()
}

with open(
    "models/final_results.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(final_results, file, indent=4)

print("\nFinal evaluation results saved!")
