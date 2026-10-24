
import numpy as np
import joblib

from external_csv_evaluation import (
    BASE_DIR,
    DATA_DIR,
    clean_external,
    load_original_texts
)

model = joblib.load(
    BASE_DIR / "models" / "final_best_model.pkl"
)

tfidf = model.named_steps["tfidf"]
classifier = model.named_steps["classifier"]
feature_names = tfidf.get_feature_names_out()

original_texts = load_original_texts()

datasets = {
    "SpamAssassin": "completeSpamAssassin.csv",
    "LingSpam": "lingSpam.csv"
}

for name, filename in datasets.items():

    data, quality = clean_external(
        DATA_DIR / filename,
        original_texts
    )

    X = data["cleaned"]
    y = data["Label"].astype(int).to_numpy()

    features = tfidf.transform(X)
    predictions = model.predict(X)
    scores = model.decision_function(X)

    # Identify correctly and incorrectly classified Ham
    correct_ham = (y == 0) & (predictions == 0)
    false_positive = (y == 0) & (predictions == 1)

    # Number of recognized TF-IDF features per email
    recognized = np.diff(features.indptr)

    print(f"\n===== {name} DIAGNOSIS =====")

    print("Correct Ham:", correct_ham.sum())
    print("False Positives:", false_positive.sum())

    if correct_ham.any():
        print(
            "Median recognized features in correct Ham:",
            np.median(recognized[correct_ham])
        )

    if false_positive.any():
        print(
            "Median recognized features in false positives:",
            np.median(recognized[false_positive])
        )

        print(
            "Average false-positive SVM score:",
            round(float(scores[false_positive].mean()), 4)
        )

        # Calculate average positive word contributions
        fp_features = features[false_positive]

        weighted = fp_features.multiply(
            classifier.coef_[0]
        )

        average = np.asarray(
            weighted.mean(axis=0)
        ).ravel()

        top_indices = np.argsort(average)[-10:][::-1]

        print("\nTop Spam-pushing terms in false positives:")

        for index in top_indices:
            if average[index] > 0:
                print(
                    feature_names[index],
                    round(float(average[index]), 5)
                )
