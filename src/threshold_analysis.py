
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# 1. Load dataset
BASE_DIR = Path(__file__).resolve().parents[1]

df = pd.read_csv(
    BASE_DIR / "data" / "processed_emails.csv"
)

X = df["cleaned_text"].fillna("")
y = df["label_num"]

# 2. Recreate 70/15/15 split
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y,
    test_size=0.15,
    random_state=42,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.15 / 0.85,
    random_state=42,
    stratify=y_temp
)

# 3. Train a fresh experimental SVM
model = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000)),
    ("classifier", LinearSVC(max_iter=5000))
])

model.fit(X_train, y_train)

# 4. Calculate validation decision scores
scores = model.decision_function(X_val)

# 5. Precision–Recall curve
precision, recall, thresholds = precision_recall_curve(
    y_val, scores
)

# The last precision/recall point has no threshold
candidate_precision = precision[:-1]
candidate_recall = recall[:-1]

# 6. Require at least 96% validation spam recall
minimum_recall = 0.96

valid_indices = np.where(
    candidate_recall >= minimum_recall
)[0]

if len(valid_indices) == 0:
    raise ValueError(
        "No candidate threshold meets the recall requirement."
    )

# Choose highest precision among eligible thresholds
best_index = max(
    valid_indices,
    key=lambda i: (
        candidate_precision[i],
        candidate_recall[i]
    )
)

best_threshold = float(thresholds[best_index])

# 7. Evaluate default versus tuned threshold
default_predictions = (scores >= 0).astype(int)
tuned_predictions = (
    scores >= best_threshold
).astype(int)

def evaluate(name, predictions):

    print(f"\n===== {name} =====")

    print(
        "Precision:",
        round(precision_score(y_val, predictions), 4)
    )

    print(
        "Recall:",
        round(recall_score(y_val, predictions), 4)
    )

    print(
        "F1 Score:",
        round(f1_score(y_val, predictions), 4)
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_val, predictions))

evaluate("DEFAULT THRESHOLD", default_predictions)
evaluate("TUNED THRESHOLD", tuned_predictions)

print("\nSelected threshold:", round(best_threshold, 4))

# 8. Save experimental threshold information
results = {
    "selected_threshold": best_threshold,
    "minimum_validation_recall": minimum_recall,
    "default_confusion_matrix": confusion_matrix(
        y_val, default_predictions
    ).tolist(),
    "tuned_confusion_matrix": confusion_matrix(
        y_val, tuned_predictions
    ).tolist()
}

output_dir = BASE_DIR / "models"

with open(
    output_dir / "threshold_experiment.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(results, file, indent=4)

# 9. Plot precision–recall curve
plt.figure(figsize=(8, 5))

plt.plot(recall, precision)

plt.xlabel("Spam Recall")
plt.ylabel("Spam Precision")
plt.title("SVM Validation Precision–Recall Curve")
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    output_dir / "threshold_pr_curve.png",
    dpi=150
)

plt.show()

print("\nThreshold analysis completed!")
