"""Evaluate the saved MailGuard AI classifier on three uploaded CSV datasets.

Run from the project root:
    python src/external_csv_evaluation.py

Place input CSVs under data/external/ first. This file does NOT retrain any model.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from text_utils import preprocess_text

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "external"
MODELS_DIR = BASE_DIR / "models"
MAX_EMAIL_LENGTH = 100_000

FILES = {
    "SpamAssassin": "completeSpamAssassin.csv",
    "Enron subset (related source)": "enronSpamSubset.csv",
    "LingSpam": "lingSpam.csv",
}


def load_original_texts():
    """Conservatively exclude exact normalized matches to the original dataset."""
    path = BASE_DIR / "data" / "processed_emails.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing original dataset for overlap check: {path}")
    original = pd.read_csv(path)
    if "cleaned_text" not in original.columns:
        raise ValueError("Original CSV must contain 'cleaned_text'.")

    # Include stored clean text and text re-cleaned with today's inference function.
    texts = set(original["cleaned_text"].fillna("").astype(str))
    if "text" in original.columns:
        texts.update(original["text"].fillna("").astype(str).map(preprocess_text))
    texts.discard("")
    return texts


def clean_external(path, original_texts):
    data = pd.read_csv(path, usecols=["Body", "Label"])
    counts = {"original_rows": int(len(data))}

    # Check labels and remove empty or unusually large bodies.
    data["Label"] = pd.to_numeric(data["Label"], errors="coerce")
    data["Body"] = data["Body"].fillna("").astype(str)
    valid = (
        data["Label"].isin([0, 1])
        & data["Body"].str.strip().ne("")
        & data["Body"].str.len().le(MAX_EMAIL_LENGTH)
    )
    counts["invalid_empty_or_oversized_removed"] = int((~valid).sum())
    data = data.loc[valid, ["Body", "Label"]].copy()
    data["cleaned"] = data["Body"].map(preprocess_text)
    counts["empty_after_cleaning_removed"] = int(data["cleaned"].eq("").sum())
    data = data.loc[data["cleaned"].ne("")].copy()

    # Remove all instances of texts assigned conflicting labels.
    conflicts = data.groupby("cleaned")["Label"].transform("nunique") > 1
    counts["conflicting_label_rows_removed"] = int(conflicts.sum())
    data = data.loc[~conflicts].copy()

    counts["duplicates_removed"] = int(data.duplicated("cleaned").sum())
    data = data.drop_duplicates(subset="cleaned")

    # No external sample should be an exact processed duplicate of the source data.
    overlap = data["cleaned"].isin(original_texts)
    counts["original_dataset_overlaps_removed"] = int(overlap.sum())
    data = data.loc[~overlap].copy()

    counts["evaluated_rows"] = int(len(data))
    counts["ham_rows"] = int((data["Label"] == 0).sum())
    counts["spam_rows"] = int((data["Label"] == 1).sum())
    return data, counts


def main():
    model_file = MODELS_DIR / "final_best_model.pkl"
    if not model_file.exists():
        raise FileNotFoundError(f"Missing trained model: {model_file}")
    # Only load your own trusted pickle file; joblib is not safe for untrusted models.
    model = joblib.load(model_file)
    original_texts = load_original_texts()
    results = {}

    for name, filename in FILES.items():
        file_path = DATA_DIR / filename
        if not file_path.exists():
            raise FileNotFoundError(
                f"Place {filename} in {DATA_DIR} and rerun."
            )
        data, counts = clean_external(file_path, original_texts)
        if data.empty or data["Label"].nunique() < 2:
            print(f"\n{name}: insufficient clean labeled examples; skipping.")
            results[name] = {"quality_checks": counts, "status": "skipped"}
            continue

        # IMPORTANT: evaluate only, never call fit() or modify the saved model.
        actual = data["Label"].astype(int)
        predicted = model.predict(data["cleaned"])
        metrics = {
            "accuracy": float(accuracy_score(actual, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
            "spam_precision": float(precision_score(actual, predicted, zero_division=0)),
            "spam_recall": float(recall_score(actual, predicted, zero_division=0)),
            "spam_f1": float(f1_score(actual, predicted, zero_division=0)),
            "confusion_matrix": confusion_matrix(actual, predicted, labels=[0, 1]).tolist(),
        }
        results[name] = {"quality_checks": counts, "metrics": metrics}
        print(f"\n===== {name} =====")
        print("Data quality:", counts)
        for metric, value in metrics.items():
            if metric == "confusion_matrix":
                print("Confusion matrix [ham, spam]:", value)
            else:
                print(f"{metric}: {value * 100:.2f}%")
        print(classification_report(actual, predicted, labels=[0, 1], target_names=["Ham", "Spam"], zero_division=0))

    MODELS_DIR.mkdir(exist_ok=True)
    report_file = MODELS_DIR / "external_csv_results.json"
    with report_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nSaved aggregate results to:", report_file)
    print("Note: 'Enron subset' is a related-source check, not independent-domain proof.")


if __name__ == "__main__":
    main()
