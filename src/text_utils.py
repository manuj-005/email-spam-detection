
import re
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r"[a-z0-9]+")

def preprocess_text(text):

    # Convert email to lowercase
    text = text.lower()

    # Replace website URLs
    text = re.sub(
        r"(https?://|www\.)\S+",
        " urltoken ",
        text
    )

    # Replace email addresses
    text = re.sub(
        r"\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b",
        " emailtoken ",
        text
    )

    # Break text into tokens
    tokens = tokenizer.tokenize(text)

    # Join tokens into cleaned text
    cleaned_text = " ".join(tokens)

    return cleaned_text
