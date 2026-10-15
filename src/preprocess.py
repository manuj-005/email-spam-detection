
import pandas as pd
import re
from nltk.tokenize import RegexpTokenizer

# Step 1: Load dataset
df = pd.read_csv("data/spam_ham_dataset.csv")

print("Original dataset size:", len(df))

# Step 2: Remove unnecessary column
df = df.drop(columns=["Unnamed: 0"])

# Step 3: Check whether identical emails have conflicting labels
conflicts = (
    df.groupby("text")["label_num"]
    .nunique()
    .gt(1)
    .sum()
)

if conflicts > 0:
    raise ValueError(
        f"Found {conflicts} email texts with conflicting labels!"
    )

# Step 4: Remove duplicate emails
df = df.drop_duplicates(subset=["text"]).copy()

print("Dataset size after removing duplicates:", len(df))

# Step 5: Create tokenizer
tokenizer = RegexpTokenizer(r"[a-z0-9]+")

# Step 6: Define text preprocessing function
def preprocess_text(text):

    # Convert text into lowercase
    text = text.lower()

    # Replace website links with a special token
    text = re.sub(
        r"(https?://|www\.)\S+",
        " urltoken ",
        text
    )

    # Replace email addresses with a special token
    text = re.sub(
        r"\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b",
        " emailtoken ",
        text
    )

    # Break text into word tokens
    tokens = tokenizer.tokenize(text)

    # Join tokens back into a clean string
    cleaned_text = " ".join(tokens)

    return cleaned_text

# Step 7: Apply preprocessing to all emails
df["cleaned_text"] = df["text"].apply(preprocess_text)

# Step 8: Display example
print("\nOriginal email:")
print(df["text"].iloc[0][:300])

print("\nCleaned email:")
print(df["cleaned_text"].iloc[0][:300])

# Step 9: Save cleaned dataset
df.to_csv("data/processed_emails.csv", index=False)

print("\nPreprocessing completed successfully!")
print("Saved file: data/processed_emails.csv")
