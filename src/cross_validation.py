
import pandas as pd

from pathlib import Path
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# Project paths
BASE_DIR = Path(__file__).resolve().parents[1]

# 1. Load cleaned data
df = pd.read_csv(
    BASE_DIR / "data" / "processed_emails.csv"
)

X = df["cleaned_text"].fillna("")
y = df["label_num"]

# 2. Recreate our final evaluation's 15% holdout
# Do not use holdout data for cross-validation
X_dev, X_test, y_dev, y_test = train_test_split(
    X,
    y,
    test_size=0.15,
    random_state=42,
    stratify=y
)

print("Development samples:", len(X_dev))
print("Excluded final test samples:", len(X_test))

# 3. Set up five stratified folds
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# 4. Define the three algorithms
algorithms = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),
    "SVM": LinearSVC(max_iter=5000)
}

metrics = {
    "Accuracy": "accuracy",
    "Precision": "precision",
    "Recall": "recall",
    "F1 Score": "f1"
}

results = []

# 5. Evaluate each model on identical folds
for name, algorithm in algorithms.items():

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000)),
        ("classifier", algorithm)
    ])

    scores = cross_validate(
        pipeline,
        X_dev,
        y_dev,
        cv=cv,
        scoring=metrics,
        n_jobs=-1
    )

    row = {"Model": name}

    for metric in metrics:
        values = scores["test_" + metric]

        row[metric + " Mean"] = values.mean()
        row[metric + " Std"] = values.std(ddof=1)

    results.append(row)

# 6. Display results
results_df = pd.DataFrame(results)

print("\n===== 5-FOLD CROSS-VALIDATION =====")

for _, row in results_df.iterrows():

    print("\nModel:", row["Model"])

    for metric in metrics:
        mean = row[metric + " Mean"] * 100
        std = row[metric + " Std"] * 100

        print(f"{metric}: {mean:.2f}% ± {std:.2f}%")

# 7. Save for dashboard and project report
output = BASE_DIR / "models" / "cross_validation_results.csv"

results_df.to_csv(output, index=False)

print("\nResults saved:", output)
