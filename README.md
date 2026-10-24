
# MailGuard AI
## Email Spam Detection Using Machine Learning and NLP

MailGuard AI is a Python-based email spam classification
system developed using Machine Learning and Natural
Language Processing (NLP).

It classifies email text as:
- SPAM: Unwanted or potentially suspicious email
- HAM: Email classified as not spam

## Features

- Spam/Ham classification
- NLP text preprocessing
- TF-IDF feature extraction
- Comparison of three ML algorithms
- Interactive Streamlit interface
- Performance metrics and confusion matrix

## Technologies

- Python 3.14
- Pandas and NumPy
- NLTK
- Scikit-learn
- Matplotlib
- Streamlit
- Joblib

## Machine Learning Models

1. Multinomial Naive Bayes
2. Logistic Regression
3. Linear Support Vector Machine (SVM)

## Dataset

Source:
https://github.com/kairess/toy-datasets/blob/master/spam_ham_dataset.csv

Original emails: 5,171
Emails after duplicate removal: 4,993

The dataset contains email text and Spam/Ham labels.

The dataset is not included in this repository.
Download it separately if you wish to retrain the models.

## Selected Model: SVM

Revised held-out evaluation:

| Metric | Result |
|---|---|
| Accuracy | 98.53% |
| Spam Precision | 96.85% |
| Spam Recall | 98.17% |
| Spam F1-score | 97.51% |

Final confusion matrix:

[[523, 7],
 [4, 215]]

The revised evaluation used a 70/15/15 split.
Earlier development experiments used a different split
of the same dataset.

## Installation

Clone the repository:

```powershell
git clone https://github.com/manuj-005/email-spam-detection.git
cd email-spam-detection
```

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Run the Web Application

The repository includes a saved trained model.

Start Streamlit:

```powershell
python -m streamlit run app_v4.py
```

Open the local URL displayed in the terminal.

## Retrain the Models (Optional)

To reproduce the training workflow:

1. Download spam_ham_dataset.csv from the dataset link.
2. Create a folder named data in the project directory.
3. Place the CSV inside data.

Then execute:

```powershell
python src/preprocess.py
python src/compare_models.py
python src/final_evaluation.py
```

This regenerates the processed dataset, model
comparison files, and final trained model.

## System Architecture

See [Architecture Documentation](docs/architecture.md).

## Project Limitations

- The model can make incorrect predictions.
- It does not verify email senders.
- It does not independently inspect malicious URLs.
- It does not scan attachments.
- It is not a complete phishing detection system.
- Performance on modern emails may differ from the
  historical dataset evaluation.

## Future Enhancements

- Modern phishing email datasets
- URL and attachment analysis
- Multilingual spam detection
- Real-time email integration
- Improved model generalization
```


## MailGuard AI V4 Features

MailGuard AI V4 includes:

- Cybersecurity-themed Streamlit dashboard
- Email Spam/Ham classification
- Explainable AI using Linear SVM feature contributions
- Comparison of three ML algorithms
- Stratified 5-fold cross-validation
- Decision-threshold optimization experiment
- External dataset generalization evaluation
- Confusion matrices and classification metrics

## External Dataset Evaluation

The saved SVM was evaluated on three additional datasets
without retraining.

| Dataset | Accuracy | Spam F1 |
|---|---:|---:|
| Original held-out test | 98.53% | 97.51% |
| Enron subset | 95.25% | 95.61% |
| SpamAssassin | 57.68% | 52.70% |
| LingSpam | 44.28% | 34.22% |

The results reveal substantial generalization limitations,
particularly high false-positive rates on SpamAssassin
and LingSpam.

The Enron subset is related to the original training source.
It should not be considered fully independent evidence.

## Reproducibility Notes

The repository contains trained models and aggregate
evaluation results.

Raw datasets are excluded.

The external evaluation dashboard displays saved results
without requiring the original external email files.

The experimental tuned threshold is not applied to
the final deployed classifier.
