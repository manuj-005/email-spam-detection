
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load dataset
df = pd.read_csv("data/spam_ham_dataset.csv")

print("===== DATASET INFORMATION =====")

# 2. Display dataset size
print("\nDataset shape:", df.shape)

# 3. Remove unnecessary index column
df = df.drop(columns=["Unnamed: 0"])

# 4. Display first five rows
print("\nFirst 5 rows:")
print(df[["label", "label_num"]].head())

# 5. Count spam and ham emails
print("\nEmail distribution:")
print(df["label"].value_counts())

# 6. Calculate percentages
print("\nEmail percentages:")
print(df["label"].value_counts(normalize=True) * 100)

# 7. Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# 8. Check duplicate emails
print("\nDuplicate emails:")
print(df["text"].duplicated().sum())

# 9. Calculate email lengths
df["email_length"] = df["text"].str.len()

print("\nEmail length statistics:")
print(df["email_length"].describe())

# 10. Create visualization
df["label"].value_counts().plot(
    kind="bar",
    title="Spam vs Ham Email Distribution"
)

plt.xlabel("Email Category")
plt.ylabel("Number of Emails")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
