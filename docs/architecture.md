
# MailGuard AI — System Architecture

## 1. Model Training Workflow

```mermaid
flowchart TD
    A["Raw Email Dataset: 5,171 emails"]
    B["Remove Duplicates: 4,993 emails"]
    C["NLP Text Preprocessing"]
    D["Train / Validation / Test Split"]
    E["TF-IDF Feature Extraction"]
    F["Naive Bayes"]
    G["Logistic Regression"]
    H["Linear SVM"]
    I["Validation and Model Selection"]
    J["Retrain Selected SVM"]
    K["Held-out Test Evaluation"]
    L["Save Trained Model"]

    A --> B --> C --> D --> E
    E --> F
    E --> G
    E --> H
    F --> I
    G --> I
    H --> I
    I --> J
    J --> K
    J --> L
```

### Training Explanation

1. Load the labeled spam and ham dataset.
2. Remove duplicate email texts.
3. Perform NLP preprocessing.
4. Divide the dataset into training, validation, and testing sets.
5. Learn TF-IDF features from training data.
6. Train three classification algorithms.
7. Compare their validation performance.
8. Select and retrain the best-performing model.
9. Evaluate it using held-out test data.
10. Save the trained pipeline for future predictions.



## 2. Email Prediction Workflow

```mermaid
flowchart TD
    A["User Enters Email"]
    B["Streamlit Web Interface"]
    C["NLP Text Preprocessing"]
    D["Saved TF-IDF Vectorizer"]
    E["Trained SVM Classifier"]
    F{"Prediction"}
    G["Spam"]
    H["Ham"]

    A --> B --> C --> D --> E --> F
    F -->|1| G
    F -->|0| H
```

### Prediction Explanation

1. The user enters an email in the Streamlit interface.
2. The email is cleaned using the preprocessing function.
3. The saved TF-IDF vectorizer transforms the text.
4. The trained SVM model classifies the email.
5. Streamlit displays either SPAM or HAM.

The model is loaded from the saved pipeline.
It does not need to be retrained for each prediction.
