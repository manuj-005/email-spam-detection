
import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Step 1: Load the preprocessed dataset
df = pd.read_csv("data/processed_emails.csv")

# Step 2: Define input and output
X = df["cleaned_text"].fillna("")
y = df["label_num"]

# Step 3: Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# Step 4: Create a pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000)),
    ("classifier", MultinomialNB())
])

# Step 5: Train the model
print("\nTraining Naive Bayes model...")

model.fit(X_train, y_train)

print("Model training completed!")

# Step 6: Make predictions on testing data
y_pred = model.predict(X_test)

# Step 7: Evaluate model performance
print("\n===== MODEL PERFORMANCE =====")

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall:", recall_score(y_test, y_pred, zero_division=0))
print("F1 Score:", f1_score(y_test, y_pred, zero_division=0))

# Step 8: Display confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

# Step 9: Display detailed classification report
print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["Ham", "Spam"],
    zero_division=0
))

# Step 10: Save trained model
Path("models").mkdir(exist_ok=True)

joblib.dump(model, "models/naive_bayes.pkl")

print("\nModel saved successfully!")
print("Location: models/naive_bayes.pkl")
