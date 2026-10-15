
import joblib
import numpy as np
from pathlib import Path

from text_utils import preprocess_text

# Load the final SVM model
project_dir = Path(__file__).resolve().parents[1]

model = joblib.load(
    project_dir / "models" / "final_best_model.pkl"
)

# Test different emails
test_emails = [
    "Congratulations! You won $50000. "
    "Click here to claim your free reward.",

    "Dear students, your project review meeting "
    "is scheduled for tomorrow at 10 AM.",

    "URGENT! Your account will be suspended today. "
    "Verify your password immediately at the link provided."
]

print("===== EMAIL PREDICTION ANALYSIS =====")

for email in test_emails:

    cleaned = preprocess_text(email)

    prediction = model.predict([cleaned])[0]

    score = model.decision_function([cleaned])[0]

    print("\nOriginal Email:", email)
    print("Cleaned Email:", cleaned)

    print(
        "Prediction:",
        "SPAM" if prediction == 1 else "HAM"
    )

    print("Decision Score:", round(float(score), 4))

# Inspect important learned features
tfidf = model.named_steps["tfidf"]
classifier = model.named_steps["classifier"]

feature_names = tfidf.get_feature_names_out()
weights = classifier.coef_[0]

top_spam_indices = np.argsort(weights)[-15:][::-1]
top_ham_indices = np.argsort(weights)[:15]

print("\n===== TOP SPAM FEATURES =====")
for index in top_spam_indices:
    print(
        feature_names[index],
        round(float(weights[index]), 4)
    )

print("\n===== TOP HAM FEATURES =====")
for index in top_ham_indices:
    print(
        feature_names[index],
        round(float(weights[index]), 4)
    )
