
# MailGuard AI

## Email Spam Detection Using Machine Learning and NLP

MailGuard AI is a machine-learning application that
classifies email text into two categories:

- Spam
- Ham (Legitimate)

## Technologies Used

- Python
- Pandas and NumPy
- NLTK
- Scikit-learn
- TF-IDF
- Streamlit
- Matplotlib

## Machine Learning Algorithms

1. Multinomial Naive Bayes
2. Logistic Regression
3. Support Vector Machine (Linear SVM)

## Selected Model

Support Vector Machine (SVM)

### Revised Evaluation Results

- Accuracy: 98.53%
- Spam Precision: 96.85%
- Spam Recall: 98.17%
- Spam F1-score: 97.51%

## Features

- Email Spam/Ham Classification
- NLP-Based Text Preprocessing
- TF-IDF Feature Extraction
- Machine Learning Model Comparison
- Performance Visualization
- Confusion Matrix Evaluation
- Interactive Streamlit Dashboard

## Run the Application

Install the required libraries:

    python -m pip install -r requirements.txt

Start the application:

    python -m streamlit run app_v2.py

Note: The trained model file must be available
inside the models directory.

## Limitations

The system performs text-based spam classification.
It does not independently verify senders, links,
attachments, or all phishing attempts.
