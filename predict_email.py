
import joblib

from src.text_utils import preprocess_text

# Step 1: Load our trained SVM model
model = joblib.load("models/svm.pkl")

print("Email Spam Detection System")
print("Model: Support Vector Machine (SVM)")
print("--------------------------------")

# Step 2: Allow the user to enter emails
while True:

    email = input(
        "\nEnter an email (or type quit to exit):\n"
    )

    if email.lower().strip() == "quit":
        print("Exiting application...")
        break

    if not email.strip():
        print("Please enter a valid email.")
        continue

    # Step 3: Apply NLP preprocessing
    cleaned_email = preprocess_text(email)

    # Step 4: Predict using the trained pipeline
    prediction = model.predict([cleaned_email])[0]

    # Step 5: Display the result
    if prediction == 1:
        print("\nPrediction: SPAM")
        print("This email appears to be spam.")
    else:
        print("\nPrediction: HAM")
        print("This email appears to be legitimate.")
