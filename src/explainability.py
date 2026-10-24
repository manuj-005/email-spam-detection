
import pandas as pd


def explain_prediction(model, cleaned_text, top_n=8):

    # Get TF-IDF and SVM from our trained pipeline
    tfidf = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]

    # Convert cleaned text into TF-IDF features
    vector = tfidf.transform([cleaned_text])

    # Get vocabulary and learned feature weights
    feature_names = tfidf.get_feature_names_out()
    weights = classifier.coef_[0]

    # Calculate each present feature's contribution
    contributions = []

    for index, value in zip(
        vector.indices,
        vector.data
    ):
        contribution = float(value * weights[index])

        contributions.append({
            "Word": feature_names[index],
            "Contribution": contribution
        })

    # Create DataFrame
    df = pd.DataFrame(
        contributions,
        columns=["Word", "Contribution"]
    )

    # Split into positive and negative contributions
    spam_words = (
        df[df["Contribution"] > 0]
        .sort_values("Contribution", ascending=False)
        .head(top_n)
    )

    ham_words = (
        df[df["Contribution"] < 0]
        .sort_values("Contribution", ascending=True)
        .head(top_n)
    )

    # Calculate full SVM decision score
    score = float(
        classifier.decision_function(vector)[0]
    )

    return spam_words, ham_words, score
