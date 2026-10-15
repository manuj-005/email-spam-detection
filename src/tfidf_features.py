
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Step 1: Load cleaned dataset
df = pd.read_csv("data/processed_emails.csv")

# Step 2: Separate input and output
X = df["cleaned_text"].fillna("")
y = df["label_num"]

# Step 3: Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training emails:", len(X_train))
print("Testing emails:", len(X_test))

# Step 4: Create TF-IDF vectorizer
tfidf = TfidfVectorizer(max_features=5000)

# Step 5: Learn vocabulary from training emails
X_train_tfidf = tfidf.fit_transform(X_train)

# Step 6: Convert testing emails using the same vocabulary
X_test_tfidf = tfidf.transform(X_test)

# Step 7: Display results
print("\nTraining TF-IDF shape:")
print(X_train_tfidf.shape)

print("\nTesting TF-IDF shape:")
print(X_test_tfidf.shape)

# Step 8: Display a sample of learned words
print("\nFirst 20 vocabulary features:")
print(tfidf.get_feature_names_out()[:20])

print("\nTF-IDF feature extraction completed!")
 